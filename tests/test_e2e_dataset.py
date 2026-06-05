import pytest
import os
import threading

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.data_processor import DataProcessor
from logic.settings_manager import SettingsManager
from db.db_manager import DBManager

TEST_DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_dataset")
)
TEST_DB_PATH = "e2e_real_test.db"


class HeadlessUIObserver:
    def __init__(self):
        self.completion_event = threading.Event()
        self.last_update_data = None
        self.started = False

    def winfo_exists(self):
        return True

    def after(self, ms, func, *args):
        func(*args)

    def ui_signal_process_start(self):
        self.started = True

    def ui_signal_process_update(self, data):
        self.last_update_data = data

    def ui_signal_process_complete(self):
        self.completion_event.set()


class TestRealDatasetEndToEnd:
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

    def test_end_to_end_pipeline_with_real_dataset(self):
        main_xlsx = os.path.join(TEST_DATASET_DIR, "test_dataset.xlsx")
        dem_xlsx = os.path.join(TEST_DATASET_DIR, "location_with_dem.xlsx")
        drone_dir = os.path.join(TEST_DATASET_DIR, "drone")

        assert os.path.exists(TEST_DATASET_DIR), (
            f"Directory missing: {TEST_DATASET_DIR}"
        )
        assert os.path.exists(main_xlsx), f"Main Excel file missing: {main_xlsx}"
        assert os.path.exists(dem_xlsx), f"DEM Excel file missing: {dem_xlsx}"
        assert os.path.exists(drone_dir), f"Drone images folder missing: {drone_dir}"

        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()

        settings.update_cv_param("globalDistanceThreshold", 0.06)
        settings.update_cv_param("temporalDeduplicationThreshold", 0.05)
        settings.update_cv_param("xfeatConfidenceThreshold", 0.001)

        processor = DataProcessor(db_manager=db_manager, settings_manager=settings)

        model_path = "onnx/xfeat_static_320.onnx"
        assert os.path.exists(model_path), (
            f"ONNX model missing at: {os.path.abspath(model_path)}"
        )

        headless_ui = HeadlessUIObserver()

        print(f"\\nStarting processing for directory: {TEST_DATASET_DIR}")
        processor.start_dataset_processing_pipeline(
            target_dir=TEST_DATASET_DIR, view_callback=headless_ui
        )

        assert headless_ui.started is True, (
            "Pipeline failed to start. Check DataProcessor logs."
        )

        timeout_seconds = 1200
        finished = headless_ui.completion_event.wait(timeout=timeout_seconds)

        assert finished is True, (
            f"Processing did not complete within {timeout_seconds} seconds."
        )

        final_metrics = headless_ui.last_update_data
        assert final_metrics is not None, (
            "UI received no updates. Worker thread likely crashed."
        )

        total_processed = final_metrics["current_index"]
        inserted_keyframes = final_metrics["total_keyframes"]

        print("\\n--- RESULTS ---")
        print(f"Total files in Excel: {final_metrics['total_items']}")
        print(f"Processed images: {total_processed}")
        print(f"Extracted keypoints: {final_metrics['total_features']}")
        print(f"Unique keyframes inserted into DB: {inserted_keyframes}")

        db_report = db_manager.get_active_database_metrics_report()
        db_landmarks_count = int(db_report["landmarks_count"])

        assert db_landmarks_count > 0, "No landmarks were inserted into the database."
        assert db_landmarks_count == inserted_keyframes, (
            "Mismatch between DataProcessor report and DB state."
        )

        assert db_landmarks_count <= total_processed, (
            "Logic error: more frames saved than processed."
        )

        if db_landmarks_count < total_processed:
            print(
                f"Deduplication confirmed. System discarded {total_processed - db_landmarks_count} identical frames."
            )
        else:
            print("All frames were unique. No discarding occurred.")
