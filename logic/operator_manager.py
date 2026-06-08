import os
import time
import threading
import cv2
from PIL import Image, ImageTk
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from gui.views.operator_view import OperatorView

from logic.xfeat_core import XFeatCore
from logic.location_consumer import LocationConsumer
from logic.logger import SystemLogger


class OperatorManager:
    """
    Manages the visual localization pipeline for the operator dashboard.
    Handles background image streaming, machine learning inference via LocationConsumer,
    and thread-safe state synchronization with the Tkinter UI.
    """

    def __init__(self, db_manager: Any, settings_manager: Any):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.active_view: Optional["OperatorView"] = None
        self.is_running: bool = False
        self.logger = SystemLogger()

        self.persistent_flight_path: list = []
        self.last_t_data: Dict[str, Any] = {"confidence": 0.0}
        self.last_map_data: Optional[Dict[str, Any]] = None

    def register_view(self, view: "OperatorView") -> None:
        self.active_view = view
        self.active_view.flight_path = self.persistent_flight_path

        if (
            self.last_map_data is not None
            or self.last_t_data.get("confidence", 0.0) > 0.0
        ):
            self.active_view.ui_update_telemetry(self.last_t_data)
            self.active_view.ui_update_map_canvas(self.last_map_data)

            if self.is_running:
                self.active_view.ui_update_status(True, "VISUAL NAVIGATION ACTIVE")

    def start_dataset_simulation(self, dataset_dir: str) -> None:
        if not self.active_view:
            self.logger.error("Localization simulation aborted: UI synchronization view missing.")
            return

        if self.is_running:
            self.logger.warn(
                "Telemetry stream is already active. Ignoring secondary start request."
            )
            return

        self.logger.info(f"Initializing visual telemetry simulation stream for dataset: {dataset_dir}")
        self.is_running = True

        try:
            worker = threading.Thread(target=self._dataset_loop, args=(dataset_dir,))
            worker.daemon = True
            worker.start()
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to allocate background thread for telemetry stream: {e}")

    def stop_simulation(self) -> None:
        self.is_running = False

    def _dataset_loop(self, dataset_dir: str) -> None:
        try:
            drone_dir = os.path.join(dataset_dir, "drone")

            if not os.path.exists(drone_dir):
                self.logger.error(f"Simulation aborted: Drone capture directory unavailable at {drone_dir}")
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
                self.logger.error(f"Simulation aborted: No valid frame buffers found in {drone_dir}")
                return

            model_path = "onnx/xfeat_static_320.onnx"
            if not os.path.exists(model_path):
                self.logger.error(f"CRITICAL: Feature extraction ONNX weights missing at {model_path}")
                return

            model = XFeatCore(model_path)
            consumer = LocationConsumer(self.db_manager, model, self.settings_manager)

            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(
                    0,
                    self.active_view.ui_update_status,
                    True,
                    "VISUAL NAVIGATION ACTIVE",
                )

            # metrics tracking
            skipped_frames = 0
            lost_frames = 0
            match_count = 0
            confidence_accumulator = 0.0

            for img_name in image_files:
                if not self.is_running:
                    break

                img_path = os.path.join(drone_dir, img_name)
                map_data = None
                t_data = {"confidence": 0.0}

                if os.path.exists(img_path):
                    cv_img = cv2.imread(img_path)

                    if cv_img is not None:
                        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        t_data["frame_pil"] = Image.fromarray(cv_img_rgb)

                        loc_result = consumer.localize(cv_img)

                        if loc_result:
                            inliers = loc_result["inliers"]
                            current_conf = min(100.0, (inliers / 50.0) * 100.0)
                            t_data["confidence"] = current_conf

                            confidence_accumulator += current_conf
                            match_count += 1

                            match_coords = loc_result["coordinates"]
                            map_data = {
                                "lat": match_coords.get("lat", 0.0),
                                "lon": match_coords.get("lon", 0.0),
                            }
                            t_data["alt"] = match_coords.get(
                                "alt", match_coords.get("rel_alt", 0.0)
                            )
                            t_data["hdg"] = match_coords.get(
                                "azimuth", match_coords.get("yaw", 0.0)
                            )
                        else:
                            # Frame processed, but no location matched
                            lost_frames += 1
                    else:
                        # Image file corrupted or unreadable
                        skipped_frames += 1
                else:
                    # Path does not exist
                    skipped_frames += 1

                # Calculate averages and attach to payload
                avg_conf = (
                    (confidence_accumulator / match_count) if match_count > 0 else 0.0
                )
                t_data["avg_conf"] = avg_conf
                t_data["skipped_frames"] = skipped_frames
                t_data["lost_frames"] = lost_frames

                self.last_t_data = t_data.copy()
                if map_data:
                    self.last_map_data = map_data.copy()

                if self.active_view and self.active_view.winfo_exists():
                    self.active_view.after(0, self._update_ui_sync, t_data, map_data)

                time.sleep(0.2)

        except Exception as e:
            self.logger.error(
                f"Telemetry stream thread encountered a fatal exception: {e}"
            )

        finally:
            self.logger.info("Telemetry simulation loop finalized. Resource locks released.")
            self.is_running = False
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(0, self.active_view.ui_update_status, False, "")

    def _update_ui_sync(self, t_data: dict, map_data: Optional[dict]) -> None:
        if not self.active_view or not self.active_view.winfo_exists():
            self.is_running = False
            return

        try:
            if t_data.get("frame_pil"):
                t_data["frame_tk"] = ImageTk.PhotoImage(t_data["frame_pil"])

            self.active_view.ui_update_telemetry(t_data)
            self.active_view.ui_update_map_canvas(map_data)
            self.active_view.update()

        except Exception as e:
            self.logger.warn(f"UI synchronization dropped: Target widget destroyed or invalid. Error: {e}")
            self.is_running = False

    def reset_session(self) -> None:
        """
        Clears the current flight path and resets tracking statistics.
        Allows the user to start fresh or stack a new dataset without old data.
        """
        self.logger.info("Session reset initiated: Purging flight path and telemetry statistics.")


        self.persistent_flight_path.clear()

        # Reset telemetry state
        self.last_t_data = {
            "confidence": 0.0,
            "avg_conf": 0.0,
            "skipped_frames": 0,
            "lost_frames": 0
        }
        self.last_map_data = None

        #Synchronize with UI if active
        if self.active_view and self.active_view.winfo_exists():
            self.active_view.flight_path = self.persistent_flight_path
            self.active_view.ui_update_telemetry(self.last_t_data)
            self.active_view.ui_update_map_canvas(None)
            self.active_view.ui_update_status(self.is_running,
                                              "SESSION RESET" if not self.is_running else "VISUAL NAVIGATION ACTIVE")