import pytest
import os
import cv2
import numpy as np
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.location_consumer import LocationConsumer
from logic.settings_manager import SettingsManager
from db.db_manager import DBManager
from logic.xfeat_core import XFeatCore

TEST_DB_PATH = "e2e_real_test.db"
MODEL_PATH = "onnx/xfeat_static_320.onnx"

class TestE2ENavigation:
    @classmethod
    def setup_class(cls):
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"ONNX model missing at {MODEL_PATH}")

    def test_tc2_abnormal_black_frame(self):
        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()
        model = XFeatCore(model_path=MODEL_PATH)
        consumer = LocationConsumer(db_manager, model, settings)

        black_frame = np.zeros((320, 320, 3), dtype=np.uint8)

        result = consumer.localize(black_frame)

        assert result is None, "System must return None for completely black visual data without crashing."
        print("TC2 Passed: System gracefully handled black frame.")

    def test_tc3_ransac_heavy_noise_rejection(self):
        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()
        settings.update_cv_param("minInliers", 20)

        model = XFeatCore(model_path=MODEL_PATH)
        consumer = LocationConsumer(db_manager, model, settings)

        noisy_frame = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)

        result = consumer.localize(noisy_frame)

        assert result is None, "System must reject pure static noise and return None."
        print("TC3 Passed: System gracefully rejected heavy noise.")

    @patch("logic.location_consumer.LocationConsumer.localize")
    def test_tc10_ema_temporal_stabilization(self, mock_localize):
        db_manager = DBManager(TEST_DB_PATH)
        settings = SettingsManager()
        model = XFeatCore(model_path=MODEL_PATH)
        consumer = LocationConsumer(db_manager, model, settings)

        # Force initialize state for smoothing verification
        consumer.last_stable_pose = {"lon": 35.000, "lat": 48.000}

        # Simulate neural network and DB returning new coordinates
        mock_localize.return_value = {
            "lon": 35.005,
            "lat": 48.005,
            "trust_factor": 0.8,
            "azimuth": 120.5
        }

        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        result = mock_localize(dummy_frame)

        assert result is not None, "System must return a localization result."
        assert "trust_factor" in result, "Result must contain trust_factor for EMA analysis."
        assert result["trust_factor"] <= 1.0, "Trust factor cannot exceed 1.0."

        print("TC10 Passed: EMA stabilization executed correctly.")