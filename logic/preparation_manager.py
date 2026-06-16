import os
from typing import Any, Dict

from logic.data_processor import DataProcessor
from db.db_manager import DBManager


class PreparationManager:
    """
    Controller layer for the Preparation View.
    Bridges the Tkinter UI thread with the background SQLite/ONNX extraction pipeline.
    """

    def __init__(self, db_manager: DBManager, settings_manager: Any, app_config: dict):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.app_config = app_config

        self.data_processor = DataProcessor(db_manager, settings_manager)
        self.last_payload = None
        self.data_processor.register_hooks(
            progress_callback=self._handle_processor_update,
            completion_callback=self._handle_processor_complete,
        )

        self.view = None

    def register_preparation_view(self, view: Any) -> None:
        """Stores a reference to the active view to push thread-safe UI updates."""
        self.view = view
        if self.data_processor.is_running:
            self.view.ui_signal_process_start()
            if self.last_payload:
                self.view.ui_signal_process_update(self.last_payload)

    def get_cv_params(self) -> Dict[str, Any]:
        """Provides the current computer vision thresholds to the UI."""
        return self.settings_manager.get_cv_params()

    def get_database_filename_node(self) -> str:
        """Retrieves the active database filename for the UI status card."""
        return os.path.basename(self.db_manager.db_path)

    def get_active_database_metrics_report(self) -> Dict[str, str]:
        """Queries the actual SQLite file footprint to populate the metrics pane."""
        try:
            db_size_bytes = os.path.getsize(self.db_manager.db_path)
            if db_size_bytes < 1024 * 1024:
                size_str = f"{db_size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{db_size_bytes / (1024 * 1024):.2f} MB"

            stats = self.db_manager.get_database_statistics()

            return {
                "landmarks_count": str(stats.get("landmarks_count", 0)),
                "global_descriptors_count": str(
                    stats.get("global_descriptors_count", 0)
                ),
                "local_features_count": str(stats.get("local_features_count", 0)),
                "disk_size_string": size_str,
            }
        except Exception:
            return {
                "landmarks_count": "0",
                "global_descriptors_count": "0",
                "local_features_count": "0",
                "disk_size_string": "Error",
            }

    def start_dataset_processing_pipeline(
        self, target_dir: str, view_callback: Any
    ) -> None:
        """Triggers the background extraction pipeline and locks the UI."""
        self.register_preparation_view(view_callback)

        if self.view and self.view.winfo_exists():
            self.view.ui_signal_process_start()

        self.data_processor.start_dataset_processing_pipeline(target_dir)

    def _handle_processor_update(self, payload: dict) -> None:
        """
        Thread-safe bridge. Receives the generic payload from the background thread
        and uses Tkinter's 'after' method to safely update the UI.
        """
        if self.view and self.view.winfo_exists():
            self.view.after(0, self.view.ui_signal_process_update, payload)

    def _handle_processor_complete(self) -> None:
        """Thread-safe bridge to unlock the UI once the pipeline terminates."""
        if self.view and self.view.winfo_exists():
            self.view.after(0, self.view.ui_signal_process_complete)
