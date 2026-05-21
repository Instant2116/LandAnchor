import json
import os
from typing import Any

class SettingsManager:
    def __init__(self, config_path: str = "system_preferences.json"):
        self.config_path = config_path
        self.data = self._load()

    def _load(self) -> dict:
        """ Reads the JSON file or returns default settings if the file is missing. """
        default_config = {
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
            "cv_params": {
                "xfeatMaxFeatures": 500,
                "matchRatio": 0.75,
                "ransacThreshold": 3.0,
                "minInliers": 15,
                "featureDetector": "XFeat (Local ONNX Engine)",
                "descriptorMatcher": "MNN Matcher (Vectorized Core)",
                "outlierFilter": "RANSAC (OpenCV Matrix)",
                "debugVisualization": True
            }
        }

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default_config
        return default_config

    def get_cv_params(self) -> dict:
        """ Returns the current computer vision parameter dictionary. """
        return self.data.get("cv_params", {})

    def update_cv_param(self, key: str, value: Any) -> None:
        """ Updates a specific computer vision parameter in memory. """
        if "cv_params" not in self.data:
            self.data["cv_params"] = {}
        self.data["cv_params"][key] = value

    def save(self) -> None:
        """ Saves the current dictionary to system_preferences.json. """
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def reset_to_defaults(self) -> dict:
        """ Restores and returns the default configuration. """
        default_state = self._load()
        self.data["cv_params"] = default_state.get("cv_params", {})
        return self.data["cv_params"]

    def get_theme(self) -> dict:
        """ Retrieves theme settings. """
        return self.data.get("theme", {})