import pytest
import os
import sys
import time
import numpy as np
import cv2
import itertools

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.location_consumer import LocationConsumer
from logic.settings_manager import SettingsManager
from db.db_manager import DBManager
from logic.xfeat_core import XFeatCore

TEST_DB_PATH = "e2e_real_test.db"
MODEL_PATH = "onnx/xfeat_static_320.onnx"
TEST_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_dataset", "drone")
)


class TestE2EPerformance:
    def test_nfr01_real_inference_performance(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"ONNX model missing at {MODEL_PATH}")
        if not os.path.exists(TEST_DATASET_DIR):
            pytest.skip(f"Real dataset directory missing at {TEST_DATASET_DIR}")

        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()
        model = XFeatCore(model_path=MODEL_PATH)
        consumer = LocationConsumer(db_manager, model, settings)

        # 1. Завантаження РЕАЛЬНИХ даних для тестування
        image_files = [
            f
            for f in os.listdir(TEST_DATASET_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not image_files:
            pytest.skip(
                "No images found in the dataset directory to perform real performance testing."
            )

        # Кешуємо до 50 реальних кадрів, щоб ізолювати час I/O диска від часу інференсу
        print(f"Loading {min(len(image_files), 50)} real frames for benchmark...")
        real_frames = []
        for file in image_files[:50]:
            img_path = os.path.join(TEST_DATASET_DIR, file)
            frame = cv2.imread(img_path)
            if frame is not None:
                real_frames.append(frame)

        # 2. Прогрів рушія ONNX (Warm-up)
        print("Warming up ONNX Runtime engine...")
        for frame in real_frames[:5]:
            consumer.localize(frame)

        # 3. Бенчмаркінг на великій вибірці (500 ітерацій)
        iterations = 500
        execution_times = []
        frame_iterator = itertools.cycle(real_frames)

        print(
            f"Starting performance benchmark with {iterations} real ONNX iterations..."
        )

        for i in range(iterations):
            current_frame = next(frame_iterator)

            start = time.perf_counter()
            consumer.localize(current_frame)
            end = time.perf_counter()

            execution_times.append((end - start) * 1000.0)

        p99_time = np.percentile(execution_times, 99)
        mean_time = np.mean(execution_times)

        print("--- BENCHMARK RESULTS (REAL DATA) ---")
        print(f"Sample size: {iterations} iterations")
        print(f"Mean execution time: {mean_time:.2f} ms")
        print(f"P99 execution time:  {p99_time:.2f} ms")

        assert p99_time < 100.0, (
            f"Performance failed: P99 time {p99_time:.2f} ms exceeds 100 ms limit."
        )
        print("NFR-01 Passed: System maintains required FPS limit on real visual data.")
