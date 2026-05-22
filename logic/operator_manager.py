import os
import time
import logging
import threading
import cv2
from PIL import Image, ImageTk
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from gui.views.operator_view import OperatorDashboardView

from logic.xfeat_core import XFeatCore
from logic.location_consumer import LocationConsumer


class OperatorManager:
    """
    Manages the visual localization pipeline for the operator dashboard.
    Handles background image streaming, machine learning inference via LocationConsumer,
    and thread-safe state synchronization with the Tkinter UI.
    """

    def __init__(self, db_manager: Any, settings_manager: Any):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.active_view: Optional["OperatorDashboardView"] = None
        self.is_running: bool = False
        self.logger = logging.getLogger("OperatorManager")

        # --- STATE CACHE ---
        # Moving volatile data out of the UI to survive tab switching
        self.persistent_flight_path: list = []
        self.last_t_data: Dict[str, Any] = {"confidence": 0.0}
        self.last_map_data: Optional[Dict[str, Any]] = None

    def register_view(self, view: "OperatorDashboardView") -> None:
        """
        Stores a reference to the active view and instantly restores its state.
        This allows the operator to switch notebook tabs without losing flight data.
        """
        self.active_view = view

        # 1. Share the memory reference so the UI appends directly to the Manager's list
        self.active_view.flight_path = self.persistent_flight_path

        # 2. Force an immediate UI redraw using the last known cached data
        if (
            self.last_map_data is not None
            or self.last_t_data.get("confidence", 0.0) > 0.0
        ):
            self.active_view.ui_update_telemetry(self.last_t_data)
            self.active_view.ui_update_map_canvas(self.last_map_data)

            if self.is_running:
                self.active_view.ui_update_status(True, "VISUAL NAVIGATION ACTIVE")

    def start_dataset_simulation(self, dataset_dir: str) -> None:
        """Initiates the dataset image parsing in a background thread."""
        if not self.active_view:
            self.logger.error("Simulation Start Failed: No active UI view registered.")
            return

        if self.is_running:
            self.logger.warning(
                "Simulation is already running! Ignoring new start request."
            )
            return

        self.logger.info(f"Initiating simulation for dataset: {dataset_dir}")
        self.is_running = True

        # Execute data processing on a separate thread to keep Tkinter responsive
        worker = threading.Thread(target=self._dataset_loop, args=(dataset_dir,))
        worker.daemon = True
        worker.start()

    def stop_simulation(self) -> None:
        """Halts the ongoing simulation safely by toggling the execution flag."""
        self.is_running = False

    def _dataset_loop(self, dataset_dir: str) -> None:
        """
        Background process: Streams images from the dataset and localizes them purely via DB.
        Executes purely mathematical operations to prevent cross-thread Tkinter deadlocks.
        """
        try:
            drone_dir = os.path.join(dataset_dir, "drone")

            if not os.path.exists(drone_dir):
                self.logger.error(f"ABORTING: Drone directory not found at {drone_dir}")
                return

            valid_extensions = (".jpg", ".jpeg", ".png")
            image_files = sorted(
                [
                    f
                    for f in os.listdir(drone_dir)
                    if f.lower().endswith(valid_extensions)
                ]
            )

            if not image_files:
                self.logger.error(f"ABORTING: No valid images found in {drone_dir}")
                return

            self.logger.info(f"Found {len(image_files)} images. Loading AI Core...")

            # Verify ONNX model existence
            model_path = "onnx/xfeat_static_320.onnx"
            if not os.path.exists(model_path):
                self.logger.error(f"CRITICAL: ONNX model not found at {model_path}")
                return

            model = XFeatCore(model_path)
            consumer = LocationConsumer(self.db_manager, model, self.settings_manager)

            # Thread-safe UI status update via .after
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(
                    0,
                    self.active_view.ui_update_status,
                    True,
                    "VISUAL NAVIGATION ACTIVE",
                )

            for img_name in image_files:
                if not self.is_running:
                    self.logger.info("Simulation halted by user/system flag.")
                    break

                img_path = os.path.join(drone_dir, img_name)
                map_data = None
                t_data = {"confidence": 0.0}

                if os.path.exists(img_path):
                    # 1. Read matrix into memory once
                    cv_img = cv2.imread(img_path)

                    if cv_img is not None:
                        # 2. Convert matrix directly to PIL (No stretching/resizing)
                        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        t_data["frame_pil"] = Image.fromarray(cv_img_rgb)

                        # 3. Execute purely visual localization directly on the matrix
                        loc_result = consumer.localize(cv_img)

                        if loc_result:
                            inliers = loc_result["inliers"]
                            t_data["confidence"] = min(100.0, (inliers / 50.0) * 100.0)

                            match_coords = loc_result["coordinates"]
                            map_data = {
                                "lat": match_coords["lat"],
                                "lon": match_coords["lon"],
                            }

                            t_data["alt"] = match_coords["alt"]
                            t_data["hdg"] = match_coords["azimuth"]

                # --- STATE CACHING ---
                # Cache the exact state into the manager before pushing to the UI
                self.last_t_data = t_data.copy()
                if map_data:
                    self.last_map_data = map_data.copy()

                # Delegate UI updates entirely to the main Tkinter thread
                if self.active_view and self.active_view.winfo_exists():
                    self.active_view.after(0, self._update_ui_sync, t_data, map_data)

                # Throttle playback to allow UI digestion and mimic real-time streaming
                time.sleep(0.2)

        except Exception as e:
            self.logger.error(
                f"Thread crashed due to an unexpected error: {e}", exc_info=True
            )

        finally:
            # Ensures flags are reset preventing permanent UI lock-outs
            self.logger.info("Simulation loop finished or terminated. Resetting flags.")
            self.is_running = False
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(0, self.active_view.ui_update_status, False, "")

    def _update_ui_sync(self, t_data: dict, map_data: Optional[dict]) -> None:
        """
        Safely updates the UI on the main Tkinter thread without stretching.
        Includes safeguards against TclErrors if the user closes the app mid-update.
        """
        # Abort if the view reference is gone or the underlying UI is destroyed
        if not self.active_view or not self.active_view.winfo_exists():
            self.is_running = False  # Signal the background thread to shut down
            return

        try:
            if t_data.get("frame_pil"):
                # Pushes the original 320x320 image as requested
                t_data["frame_tk"] = ImageTk.PhotoImage(t_data["frame_pil"])

            self.active_view.ui_update_telemetry(t_data)
            self.active_view.ui_update_map_canvas(map_data)

            # Explicitly force Tkinter to flush the draw queue and paint the screen NOW
            self.active_view.update()

        except Exception as e:
            # Catch Tkinter TclErrors if the widget is abruptly destroyed
            self.logger.warning(f"UI update aborted due to closed widget: {e}")
            self.is_running = False
