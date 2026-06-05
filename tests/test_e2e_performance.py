import pytest
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.location_consumer import LocationConsumer
from logic.settings_manager import SettingsManager
from db.db_manager import DBManager
from logic.xfeat_core import XFeatCore

TEST_DB_PATH = "e2e_real_test.db"
MODEL_PATH = "onnx/xfeat_static_320.onnx"


class TestE2EPerformance:
    def test_nfr01_real_inference_performance(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"ONNX model missing at {MODEL_PATH}")

        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()
        model = XFeatCore(model_path=MODEL_PATH)
        consumer = LocationConsumer(db_manager, model, settings)

        dummy_frame = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)

        consumer.localize(dummy_frame)

        iterations = 100
        execution_times = []

        print(
            f"Starting performance benchmark with {iterations} real ONNX iterations..."
        )

        for i in range(iterations):
            start = time.perf_counter()
            consumer.localize(dummy_frame)
            end = time.perf_counter()

            execution_times.append((end - start) * 1000.0)

        p99_time = np.percentile(execution_times, 99)
        mean_time = np.mean(execution_times)

        print("--- BENCHMARK RESULTS ---")
        print(f"Mean execution time: {mean_time:.2f} ms")
        print(f"P99 execution time:  {p99_time:.2f} ms")

        assert p99_time < 100.0, (
            f"Performance failed: P99 time {p99_time:.2f} ms exceeds 100 ms limit."
        )
        print("NFR-01 Passed: System maintains required FPS limit.")
