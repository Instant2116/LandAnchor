import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.auth import CryptographicAuthManager
from logic.settings_manager import SettingsManager
from db.db_manager import DBManager
from logic.xfeat_core import XFeatCore

TEST_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_dbs", "e2e_security.db")
)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "onnx", "xfeat_static_320.onnx")
)
HARDWARE_KEY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "hardware_key.pem")
)


class TestE2ESecurityAndConfiguration:
    @classmethod
    def setup_class(cls):
        # Added: Ensure target directory exists and state is clean
        os.makedirs(os.path.dirname(TEST_DB_PATH), exist_ok=True)
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except PermissionError:
                pass

    def test_nfr_auth_hardware_token_blocking(self):
        db_manager = DBManager(TEST_DB_PATH)
        auth_manager = CryptographicAuthManager(db_manager=db_manager)

        original_key_exists = os.path.exists(HARDWARE_KEY_PATH)
        if original_key_exists:
            os.rename(HARDWARE_KEY_PATH, f"{HARDWARE_KEY_PATH}.backup")

        try:
            result = auth_manager.authenticate_by_token(HARDWARE_KEY_PATH)
            assert result is None, (
                "System failed to block access when hardware key is missing."
            )
            print("TC4 Passed: Access completely blocked without hardware token.")
        finally:
            if original_key_exists:
                os.rename(f"{HARDWARE_KEY_PATH}.backup", HARDWARE_KEY_PATH)

    def test_functional_core_dynamic_parameter_injection(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip(f"ONNX model missing at {MODEL_PATH}")

        settings = SettingsManager()
        model = XFeatCore(model_path=MODEL_PATH, gem_p=3)

        assert model.gem_p == 3, "Initial state of GeM parameter must be 3."

        settings.update_cv_param("gemPoolingPower", 5)
        model.set_gem_p(int(settings.get_cv_params().get("gemPoolingPower")))

        assert model.gem_p == 5, "Model parameter was not dynamically updated."
        print("TC7 Passed: Dynamic parameter injection to ONNX core successful.")