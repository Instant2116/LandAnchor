import json
import os
from typing import Any

class SettingsManager:
    def __init__(self, config_path: str = "system_preferences.json"):
        self.config_path = config_path
        self.data = self._load()

    def _get_hardcoded_defaults(self) -> dict:
        """ Returns the structural baseline to prevent KeyErrors. """
        return {
            "theme": {
                "bg_primary": "#020617",
                "bg_secondary": "#0f172a",
                "bg_tertiary": "#1e293b",
                "bg_accent": "#2563eb",
                "bg_success": "#022c22",
                "bg_error": "#450a0a",
                "text_primary": "#f8fafc",
                "text_secondary": "#cbd5e1",
                "text_muted": "#64748b",
                "text_status": "#94a3b8",
                "accent_green": "#4ade80",
                "accent_blue": "#60a5fa",
                "accent_red": "#ef4444",
                "accent_yellow": "#eab308",
                "accent_success": "#10b981",
                "border_color": "#334155"
            },
            "cv_defaults": {
                "xfeatMaxFeatures": 500,
                "xfeatConfidenceThreshold": 0.005,
                "gemPoolingPower": 3,
                "matchRatio": 0.75,
                "ransacThreshold": 3.0,
                "minInliers": 15,
                "topKCandidates": 5,
                "globalDistanceThreshold": 0.4,
                "featureDetector": "XFeat (Local ONNX Engine)",
                "descriptorMatcher": "MNN Matcher (Vectorized Core)",
                "outlierFilter": "RANSAC (OpenCV Matrix)",
                "debugVisualization": True
            },
            "system": {
                "app_title": "Drone Security System - LandAnchor",
                "version": "v2.5.0",
                "default_key_name": "hardware_key.pem"
            }
        }

    def _load(self) -> dict:
        """ Reads the JSON file and performs a deep merge with defaults to prevent missing keys. """
        default_config = self._get_hardcoded_defaults()

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)

                    # Perform a deep merge for the nested dictionaries
                    merged_config = default_config.copy()

                    for section in ["theme", "cv_defaults", "system"]:
                        if section in file_data:
                            # Update the specific section with saved data, keeping defaults for missing keys
                            merged_config[section].update(file_data[section])

                    return merged_config
            except Exception:
                return default_config
        return default_config

    def get_cv_params(self) -> dict:
        """ Returns the current computer vision parameter dictionary. """
        return self.data.get("cv_defaults", {})

    def update_cv_param(self, key: str, value: Any) -> None:
        """ Updates a specific computer vision parameter in memory. """
        if "cv_defaults" not in self.data:
            self.data["cv_defaults"] = {}
        self.data["cv_defaults"][key] = value

    def save(self) -> None:
        """ Saves the current dictionary to system_preferences.json. """
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def reset_to_defaults(self) -> dict:
        """ Restores and returns the default configuration. """
        default_state = self._get_hardcoded_defaults()
        self.data["cv_defaults"] = default_state["cv_defaults"]
        return self.data["cv_defaults"]

    def get_theme(self) -> dict:
        """ Retrieves theme settings. """
        return self.data.get("theme", {})