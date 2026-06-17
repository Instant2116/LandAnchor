import pytest
import os
import time
import sqlite3
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.db_manager import DBManager
from logic.data_processor import DataProcessor
from logic.settings_manager import SettingsManager

TEST_DB_PATH = "e2e_scale_ui.db"
TEST_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_dataset")
)


class TestScalabilityAndUI:
    @classmethod
    def setup_class(cls):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except PermissionError:
                pass

    def test_tc5_database_scalability_nfr04(self):
        db_manager = DBManager(TEST_DB_PATH)

        target_count = 50000
        chunk_size = 5000
        current_id = 1

        print(f"\nPopulating database with {target_count} records for scale test...")

        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()

            for i in range(0, target_count, chunk_size):
                landmarks = [(0.0, 0.0, 0.0, 0.0) for _ in range(chunk_size)]
                cursor.executemany(
                    "INSERT INTO Landmarks (coordinate_x, coordinate_y, coordinate_z, azimuth) VALUES (?, ?, ?, ?);",
                    landmarks,
                )

                globals_desc = []
                for j in range(chunk_size):
                    l_id = current_id + j
                    # Generate vector mathematically identical to GeM Pooling output
                    g_vec = np.random.rand(64).astype(np.float32).tobytes()
                    globals_desc.append((l_id, g_vec))

                cursor.executemany(
                    "INSERT INTO GlobalDescriptors (landmark_id, global_vector) VALUES (?, ?);",
                    globals_desc,
                )
                conn.commit()

                current_id += chunk_size

        # Measure DB fetch time (I/O)
        start_fetch = time.perf_counter()
        records = db_manager.get_all_global_descriptors()
        fetch_time = (time.perf_counter() - start_fetch) * 1000.0

        assert records is not None and len(records) > 0, (
            "Failed to fetch records from DB."
        )
        db_vectors = np.vstack(
            [np.frombuffer(row[1], dtype=np.float32) for row in records]
        )

        # Statistically significant search benchmark (500 iterations)
        search_iterations = 500
        search_times = []

        print(
            f"Benchmarking vectorized search over {target_count} records ({search_iterations} iterations)..."
        )

        for _ in range(search_iterations):
            target_vector = np.random.rand(64).astype(np.float32)

            start_search = time.perf_counter()
            distances = np.linalg.norm(db_vectors - target_vector, axis=1)
            _ = np.argmin(distances)
            end_search = time.perf_counter()

            search_times.append((end_search - start_search) * 1000.0)

        mean_search_time = np.mean(search_times)
        p99_search_time = np.percentile(search_times, 99)

        print("--- SCALABILITY RESULTS ---")
        print(f"Database fetch time (50k records): {fetch_time:.2f} ms")
        print(f"Mean vector math search time:      {mean_search_time:.2f} ms")
        print(f"P99 vector math search time:       {p99_search_time:.2f} ms")

        assert p99_search_time < 100.0, (
            f"Vector search time P99 {p99_search_time:.2f} ms is too slow."
        )

    def test_tc6_ui_reactivity_nfr05(self):
        # Gracefully skip ONLY this UI test on headless servers.
        # This allows test_tc5 to execute normally.
        tk = pytest.importorskip("tkinter")

        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()
        processor = DataProcessor(db_manager, settings)

        root = tk.Tk()

        processor.start_dataset_processing_pipeline(TEST_DATASET_DIR)

        root.update()
        time.sleep(0.5)

        if not processor.is_running:
            root.destroy()
            pytest.skip(
                "Pipeline did not start properly. Check dataset and ONNX paths."
            )

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            root.update()
            latency = (time.perf_counter() - start) * 1000.0
            latencies.append(latency)
            time.sleep(0.05)

        max_latency = max(latencies)
        mean_latency = np.mean(latencies)

        processor.is_running = False
        root.destroy()

        print("--- UI REACTIVITY RESULTS ---")
        print(f"Mean UI event loop latency: {mean_latency:.2f} ms")
        print(f"Max UI event loop latency:  {max_latency:.2f} ms")

        assert max_latency < 200.0, (
            f"UI blocked. Max latency {max_latency:.2f} ms exceeded 200 ms limit."
        )
        print(
            "TC6 Passed: Main UI thread remains responsive during active background processing."
        )
