import pytest
import os
import sys
import time
import json
import subprocess
import numpy as np

CLI_SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cli_main.py")
)
TEST_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_dbs", "e2e_cli_perf.db")
)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "onnx", "xfeat_static_320.onnx")
)
TEST_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_dataset", "drone")
)


class TestE2ECLIPerformance:
    @classmethod
    def setup_class(cls):
        if not os.path.exists(CLI_SCRIPT_PATH):
            pytest.skip(f"CLI script missing at {CLI_SCRIPT_PATH}")
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"ONNX model missing at {MODEL_PATH}")
        if not os.path.exists(TEST_DATASET_DIR):
            pytest.skip(f"Test dataset missing at {TEST_DATASET_DIR}")

    def test_nfr_cli_cold_start_latency(self):
        image_files = [
            f
            for f in os.listdir(TEST_DATASET_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not image_files:
            pytest.skip("No images found for cold start test.")

        single_image_path = os.path.join(TEST_DATASET_DIR, image_files[0])

        command = [
            sys.executable,
            CLI_SCRIPT_PATH,
            "-i",
            single_image_path,
            "--db",
            TEST_DB_PATH,
            "--model",
            MODEL_PATH,
        ]

        start_time = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000.0

        assert process.returncode == 0, (
            f"CLI crashed during cold start: {process.stderr}"
        )

        try:
            output_data = json.loads(process.stdout.strip())
            assert output_data["status"] in ["success", "failed"], (
                "Unexpected JSON status."
            )
        except json.JSONDecodeError:
            pytest.fail(f"CLI did not output valid JSON: {process.stdout}")

        print("\n--- CLI COLD START RESULTS ---")
        print(f"Total Boot + 1 Frame Inference: {execution_time_ms:.2f} ms")

        assert execution_time_ms < 5000.0, (
            f"Cold start latency {execution_time_ms:.2f} ms exceeded 5 second limit."
        )

    def test_nfr_cli_batch_throughput(self):
        command = [
            sys.executable,
            CLI_SCRIPT_PATH,
            "-d",
            TEST_DATASET_DIR,
            "--db",
            TEST_DB_PATH,
            "--model",
            MODEL_PATH,
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # Bypasses the OS pipe buffer limit
            text=True,
            bufsize=1,
        )

        frame_timestamps = []
        parsed_results = 0

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if "status" in data:
                    frame_timestamps.append(time.perf_counter())
                    parsed_results += 1
            except json.JSONDecodeError:
                pass

        process.wait()

        assert process.returncode == 0, (
            f"CLI exited with error code {process.returncode}."
        )
        assert parsed_results > 10, (
            "Not enough files processed to measure sustained throughput."
        )

        inter_frame_latencies_ms = np.diff(frame_timestamps) * 1000.0
        mean_latency = np.mean(inter_frame_latencies_ms)
        p99_latency = np.percentile(inter_frame_latencies_ms, 99)

        total_batch_time = frame_timestamps[-1] - frame_timestamps[0]
        fps = (parsed_results - 1) / total_batch_time if total_batch_time > 0 else 0

        print("\n--- CLI BATCH THROUGH THROUGHPUT RESULTS ---")
        print(f"Frames processed: {parsed_results}")
        print(f"Sustained Pipeline FPS: {fps:.2f} FPS")
        print(f"Mean inter-frame latency: {mean_latency:.2f} ms")
        print(f"P99 inter-frame latency:  {p99_latency:.2f} ms")
        #min 10 fps
        assert p99_latency < 200.0, (
            f"P99 pipeline latency {p99_latency:.2f} ms exceeds limit."
        )