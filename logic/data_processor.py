import os
import cv2
import logging
import threading
import datetime
import numpy as np
import pandas as pd
from typing import Any
from PIL import Image

from logic.xfeat_core import XFeatCore
from db.db_manager import DBManager


class DataProcessor:
    """
    Manages the ingestion, feature extraction, and database insertion of new datasets.
    Handles telemetry merging (Excel) and background ONNX processing while maintaining
    a thread-safe state synchronization with the Tkinter UI.
    """

    def __init__(self, db_manager: DBManager, settings_manager: Any):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.model = None
        self.logger = logging.getLogger("DataProcessor")

        # --- STATE CACHE ---
        # Moving volatile progress data out of the UI to survive tab switching
        self.active_view = None
        self.last_update_data = None
        self.is_running = False

    def register_view(self, view_callback: Any) -> None:
        """
        Registers the active UI view and instantly restores the progress state.
        Allows the operator to switch notebook tabs without losing the progress bar.
        """
        self.active_view = view_callback
        if (
            self.last_update_data
            and self.active_view
            and self.active_view.winfo_exists()
        ):
            self.active_view.after(
                0, self.active_view.ui_signal_process_update, self.last_update_data
            )

    def start_dataset_processing_pipeline(
        self, target_dir: str, view_callback: Any
    ) -> None:
        """
        Initializes the dataset ingestion sequence by merging telemetry Excel files,
        and spawns a background worker thread to process the images autonomously.
        """
        # Instantly register and bind the new view
        self.register_view(view_callback)

        if self.is_running:
            self.logger.warning(
                "Pipeline is already running. Ignoring new start request."
            )
            return

        self.logger.info(
            f"Starting data processing pipeline for directory: {target_dir}"
        )
        dataset_name = os.path.basename(os.path.normpath(target_dir))

        main_xlsx_path = os.path.join(target_dir, f"{dataset_name}.xlsx")
        dem_xlsx_path = os.path.join(target_dir, "location_with_dem.xlsx")
        model_path = "onnx/xfeat_static_320.onnx"

        if not os.path.exists(main_xlsx_path) or not os.path.exists(dem_xlsx_path):
            self.logger.error("Required telemetry files do not exist.")
            self.logger.error(f"Expected main xlsx: {main_xlsx_path}")
            self.logger.error(f"Expected dem xlsx: {dem_xlsx_path}")
            return

        if not os.path.exists(model_path):
            self.logger.error(f"CRITICAL: ONNX model not found at {model_path}")
            return

        if self.model is None:
            self.model = XFeatCore(model_path)

        processing_queue = []
        try:
            self.logger.info("Parsing and merging telemetry data...")
            # Read and merge the Excel files on the 'id' column
            df_main = pd.read_excel(main_xlsx_path)
            df_dem = pd.read_excel(dem_xlsx_path)
            df_merged = pd.merge(df_main, df_dem, on="id", how="inner")

            for index, row in df_merged.iterrows():
                # Ensure the filename has a .jpg extension
                img_id = str(row["id"])
                img_filename = (
                    img_id if img_id.lower().endswith(".jpg") else f"{img_id}.jpg"
                )

                # Route the target path to the 'drone' subdirectory
                relative_img_path = os.path.join("drone", img_filename)

                processing_queue.append(
                    {
                        "id": relative_img_path,
                        "lon": float(row["lon"]),
                        "lat": float(row["lat"]),
                        "rel_alt": float(row["relative_altitude"]),
                        "yaw": float(row["yaw"]) if pd.notna(row["yaw"]) else 0.0,
                    }
                )
        except Exception as e:
            self.logger.error(f"Exception during Excel processing: {e}", exc_info=True)
            return

        if not processing_queue:
            self.logger.error("Processing queue is empty. Aborting pipeline.")
            return

        self.is_running = True

        # Lock the UI via callback
        if self.active_view and self.active_view.winfo_exists():
            self.active_view.ui_signal_process_start()

        # Execute data processing on a separate thread to keep Tkinter responsive
        worker_thread = threading.Thread(
            target=self._pipeline_worker_thread,
            args=(target_dir, processing_queue),
        )
        worker_thread.daemon = True
        worker_thread.start()

    def _pipeline_worker_thread(self, target_dir: str, queue: list) -> None:
        """
        Executes ONNX processing and delegates database commits to the DBManager.
        Triggers thread-safe UI updates on the main thread via .after().
        """
        try:
            total_items = len(queue)
            total_features = 0
            total_landmarks = 0
            total_keyframes = 0

            cv_params = self.settings_manager.get_cv_params()
            max_features_limit = int(cv_params.get("xfeatMaxFeatures", 500))
            confidence_threshold = float(
                cv_params.get("xfeatConfidenceThreshold", 0.005)
            )

            self.logger.info(f"Initiating extraction for {total_items} queued images.")

            for current_index, node in enumerate(queue):
                if not self.is_running:
                    self.logger.info("Pipeline processing aborted by system flag.")
                    break

                image_target_path = os.path.join(target_dir, node["id"])
                kpts_count = 0
                img_res = None
                pil_frame = None

                if os.path.exists(image_target_path):
                    img = cv2.imread(image_target_path)

                    if img is not None:
                        img_res = cv2.resize(img, (320, 320))

                        inference_results = self.model.process(img_res)

                        # Flatten all dense tensors to 1D/2D arrays BEFORE slicing
                        scores = inference_results["scores"].reshape(-1)
                        desc = inference_results["desc"].reshape(-1, 64)
                        kpts = inference_results["kpts"].reshape(-1, 2)

                        # Utilize the parameterized confidence threshold
                        valid_indices = np.where(scores > confidence_threshold)[0]
                        if len(valid_indices) > max_features_limit:
                            valid_indices = np.argsort(scores)[-max_features_limit:]

                        filtered_descriptors = desc[valid_indices]
                        filtered_keypoints = kpts[valid_indices]
                        kpts_count = len(valid_indices)

                        if kpts_count > 0:
                            gem_vector = inference_results["global"]
                            current_timestamp = datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            coordinates = (
                                node["lon"],
                                node["lat"],
                                node["rel_alt"],
                                node["yaw"],
                            )

                            try:
                                # Delegate database insertion strictly to the Data Layer
                                self.db_manager.insert_landmark_transaction(
                                    coordinates=coordinates,
                                    global_desc=gem_vector.tobytes(),
                                    local_features=filtered_descriptors.tobytes(),
                                    keypoints=filtered_keypoints.tobytes(),
                                    timestamp=current_timestamp,
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"Database insertion failed for node {node['id']}: {e}"
                                )

                        # Convert OpenCV BGR matrix directly to PIL RGB for UI rendering
                        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(img_rgb)

                # Update operational metrics
                total_features += kpts_count
                if kpts_count > 0:
                    total_landmarks += 1
                if current_index % 3 == 0 and kpts_count > 0:
                    total_keyframes += 1

                progress_ratio = ((current_index + 1) / total_items) * 100

                # --- STATE CACHING ---
                # Save the exact state to the persistent manager before pushing to the UI
                self.last_update_data = {
                    "progress_ratio": progress_ratio,
                    "total_features": total_features,
                    "total_landmarks": total_landmarks,
                    "total_keyframes": total_keyframes,
                    "current_index": current_index + 1,
                    "total_items": total_items,
                    "frame_pil": pil_frame,
                }

                # Push UI update to the main Tkinter thread safely
                if self.active_view and self.active_view.winfo_exists():
                    self.active_view.after(
                        0,
                        self.active_view.ui_signal_process_update,
                        self.last_update_data,
                    )

        except Exception as e:
            self.logger.error(f"Pipeline worker thread crashed: {e}", exc_info=True)

        finally:
            self.logger.info(
                "Pipeline processing finished or terminated. Resetting flags."
            )
            self.is_running = False
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(0, self.active_view.ui_signal_process_complete)
