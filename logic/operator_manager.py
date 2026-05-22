import os
import time
import threading
import cv2
from PIL import Image, ImageTk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui.views.operator_view import OperatorDashboardView

from logic.xfeat_core import XFeatCore
from logic.location_consumer import LocationConsumer


class OperatorManager:
    def __init__(self, db_manager, settings_manager):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.active_view: "OperatorDashboardView" = None
        self.is_running = False

    def register_view(self, view: "OperatorDashboardView") -> None:
        """Stores a typed reference to the active OperatorDashboardView."""
        self.active_view = view

    def start_dataset_simulation(self, dataset_dir: str) -> None:
        """Initiates the dataset image parsing in a background thread."""
        import logging
        logger = logging.getLogger("OperatorManager")

        if not self.active_view:
            logger.error("Simulation Start Failed: No active UI view registered.")
            return

        if self.is_running:
            logger.warning("Simulation is already running! Ignoring new start request.")
            return

        logger.info(f"Initiating simulation for dataset: {dataset_dir}")
        self.is_running = True
        worker = threading.Thread(target=self._dataset_loop, args=(dataset_dir,))
        worker.daemon = True
        worker.start()

    def stop_simulation(self) -> None:
        self.is_running = False

    def _dataset_loop(self, dataset_dir: str) -> None:
        """Background process: Streams images from the dataset and localizes them purely via DB."""
        import logging
        logger = logging.getLogger("OperatorManager")

        try:
            drone_dir = os.path.join(dataset_dir, "drone")

            if not os.path.exists(drone_dir):
                logger.error(f"ABORTING: Drone directory not found at {drone_dir}")
                return

            valid_extensions = (".jpg", ".jpeg", ".png")
            image_files = sorted(
                [f for f in os.listdir(drone_dir) if f.lower().endswith(valid_extensions)]
            )

            if not image_files:
                logger.error(f"ABORTING: No valid images (.jpg, .png) found in {drone_dir}")
                return

            logger.info(f"Found {len(image_files)} images. Loading AI Core...")

            # FIX: Ensure this path perfectly matches where your model actually is!
            # If your model is in the root folder, change this to just "xfeat_static_320.onnx"
            model_path = "xfeat_static_320.onnx"
            if not os.path.exists(model_path):
                logger.error(f"CRITICAL: ONNX model not found at {model_path}")
                return

            model = XFeatCore(model_path)
            consumer = LocationConsumer(self.db_manager, model, self.settings_manager)

            self.active_view.after(
                0, self.active_view.ui_update_status, True, "VISUAL NAVIGATION ACTIVE"
            )

            w = self.active_view.hud_canvas.winfo_width()
            h = self.active_view.hud_canvas.winfo_height()
            w = w if w > 10 else 640
            h = h if h > 10 else 480

            for img_name in image_files:
                if not self.is_running:
                    logger.info("Simulation halted by user/system flag.")
                    break

                img_path = os.path.join(drone_dir, img_name)
                pil_img = None
                map_data = None
                t_data = {"confidence": 0.0}

                if os.path.exists(img_path):
                    cv_img = cv2.imread(img_path)

                    if cv_img is not None:
                        # Convert directly to PIL without resizing to prevent Tkinter lag
                        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(cv_img_rgb)
                        t_data["frame_pil"] = pil_img

                        # Execute purely visual localization
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

                # Push UI update
                self.active_view.after(0, self._update_ui_sync, t_data, map_data)

                # Throttle playback
                time.sleep(0.2)

        except Exception as e:
            logger.error(f"Thread crashed due to an unexpected error: {e}", exc_info=True)

        finally:
            # THIS BLOCK ALWAYS RUNS - Prevents the UI button from getting permanently locked
            logger.info("Simulation loop finished or terminated. Resetting flags.")
            self.is_running = False
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(0, self.active_view.ui_update_status, False, "")


    def stop_simulation(self) -> None:
        self.is_running = False

    def _dataset_loop(self, dataset_dir: str) -> None:
        """Background process: Streams images from the dataset and localizes them purely via DB."""
        import logging
        logger = logging.getLogger("OperatorManager")

        try:
            drone_dir = os.path.join(dataset_dir, "drone")

            if not os.path.exists(drone_dir):
                logger.error(f"ABORTING: Drone directory not found at {drone_dir}")
                return

            valid_extensions = (".jpg", ".jpeg", ".png")
            image_files = sorted(
                [f for f in os.listdir(drone_dir) if f.lower().endswith(valid_extensions)]
            )

            if not image_files:
                logger.error(f"ABORTING: No valid images (.jpg, .png) found in {drone_dir}")
                return

            logger.info(f"Found {len(image_files)} images. Loading AI Core...")

            # Restored your correct path
            model_path = "onnx/xfeat_static_320.onnx"
            if not os.path.exists(model_path):
                logger.error(f"CRITICAL: ONNX model not found at {model_path}")
                return

            model = XFeatCore(model_path)
            consumer = LocationConsumer(self.db_manager, model, self.settings_manager)

            # Thread-safe UI delegation via .after
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(
                    0, self.active_view.ui_update_status, True, "VISUAL NAVIGATION ACTIVE"
                )

            for img_name in image_files:
                if not self.is_running:
                    logger.info("Simulation halted by user/system flag.")
                    break

                img_path = os.path.join(drone_dir, img_name)
                map_data = None
                t_data = {"confidence": 0.0}

                if os.path.exists(img_path):
                    cv_img = cv2.imread(img_path)

                    if cv_img is not None:
                        # Convert to PIL directly (no stretching/resizing)
                        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        t_data["frame_pil"] = Image.fromarray(cv_img_rgb)

                        # Execute purely visual localization
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

                # Delegate UI updates entirely to the main thread
                if self.active_view and self.active_view.winfo_exists():
                    self.active_view.after(0, self._update_ui_sync, t_data, map_data)

                time.sleep(0.2)

        except Exception as e:
            logger.error(f"Thread crashed due to an unexpected error: {e}", exc_info=True)

        finally:
            logger.info("Simulation loop finished or terminated. Resetting flags.")
            self.is_running = False
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(0, self.active_view.ui_update_status, False, "")


        valid_extensions = (".jpg", ".jpeg", ".png")
        image_files = sorted(
            [f for f in os.listdir(drone_dir) if f.lower().endswith(valid_extensions)]
        )

        if not image_files:
            self.is_running = False
            return

        # Initialize the Location Consumer and ONNX model
        model = XFeatCore("onnx/xfeat_static_320.onnx")
        consumer = LocationConsumer(self.db_manager, model, self.settings_manager)

        self.active_view.after(
            0, self.active_view.ui_update_status, True, "VISUAL NAVIGATION ACTIVE"
        )

        for img_name in image_files:
            if not self.is_running:
                break

            img_path = os.path.join(drone_dir, img_name)
            pil_img = None

            # Default empty states for a lost drone
            map_data = None
            t_data = {"confidence": 0.0}

            if os.path.exists(img_path):
                cv_img = cv2.imread(img_path)
                if cv_img is not None:
                    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv_img_rgb)
                    t_data["frame_pil"] = pil_img

                    # Execute purely visual localization
                    loc_result = consumer.localize(img_path)

                    if loc_result:
                        inliers = loc_result["inliers"]
                        t_data["confidence"] = min(100.0, (inliers / 50.0) * 100.0)

                        match_coords = loc_result["coordinates"]

                        # Populate map data solely from the database match
                        map_data = {
                            "lat": match_coords["lat"],
                            "lon": match_coords["lon"],
                        }

                        # Populate HUD telemetry solely from the database match
                        t_data["alt"] = match_coords["alt"]
                        t_data["hdg"] = match_coords["azimuth"]

            self.active_view.after(0, self._update_ui_sync, t_data, map_data)

            # Throttle the playback speed so the operator can actually see the UI update
            time.sleep(0.2)

        self.is_running = False
        self.active_view.after(0, self.active_view.ui_update_status, False, "")

    def _update_ui_sync(self, t_data: dict, map_data: dict) -> None:
        """Safely updates the UI on the main Tkinter thread without stretching."""
        if not self.active_view:
            return

        if t_data.get("frame_pil"):
            # Pushes the original 320x320 image as requested
            t_data["frame_tk"] = ImageTk.PhotoImage(t_data["frame_pil"])

        self.active_view.ui_update_telemetry(t_data)
        self.active_view.ui_update_map_canvas(map_data)

        # Explicitly force Tkinter to flush the draw queue and paint the screen NOW
        self.active_view.update()