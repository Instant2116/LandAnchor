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
        try:
            total_items = len(queue)
            total_features = 0
            total_landmarks = 0
            total_keyframes = 0

            cv_params = self.settings_manager.get_cv_params()
            max_features_limit = int(cv_params.get("xfeatMaxFeatures", 500))
            confidence_threshold = float(cv_params.get("xfeatConfidenceThreshold", 0.005))
            deduplication_threshold = float(cv_params.get("temporalDeduplicationThreshold", 0.05))
            gem_p = int(cv_params.get("gemPoolingPower", 3))
            gem_distance_threshold = float(cv_params.get("globalDistanceThreshold", 0.06))

            if self.model:
                self.model.set_gem_p(gem_p)

            # --- ПУНКТ 5: ЧАСОВА СТАБІЛІЗАЦІЯ КАРТИ (LIFELONG MAPPING) ---
            # Завантажуємо історичну базу для уникнення дублікатів при повторних прогонах
            self.logger.info("Loading historical map state for temporal deduplication...")
            existing_globals_raw = self.db_manager.get_all_global_descriptors()
            historical_gems = []
            for l_id, gem_blob in existing_globals_raw:
                historical_gems.append((l_id, self.db_manager.blob_to_array(gem_blob, dtype=np.float32)))

            # Лічильник для відображення оновлених (а не доданих) орієнтирів
            total_updated = 0

            last_saved_gem = None
            self.logger.info(f"Initiating extraction for {total_items} queued images.")

            for current_index, node in enumerate(queue):
                if not self.is_running:
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

                        scores = inference_results["scores"].reshape(-1)
                        desc = inference_results["desc"].reshape(-1, 64)
                        kpts = inference_results["kpts"].reshape(-1, 2)
                        gem_vector = inference_results["global"].reshape(-1)

                        valid_indices = np.where(scores > confidence_threshold)[0]
                        if len(valid_indices) > max_features_limit:
                            valid_indices = np.argsort(scores)[-max_features_limit:]

                        filtered_descriptors = desc[valid_indices]
                        filtered_keypoints = kpts[valid_indices]
                        kpts_count = len(valid_indices)

                        if kpts_count > 0:
                            is_keyframe = False

                            if last_saved_gem is None:
                                is_keyframe = True
                            else:
                                gem_distance = np.linalg.norm(gem_vector - last_saved_gem)
                                if gem_distance > gem_distance_threshold:
                                    is_keyframe = True

                            if is_keyframe:
                                current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                # --- ЛОГІКА ДЕДУПЛІКАЦІЇ ---
                                is_duplicate = False
                                duplicate_id = None

                                if historical_gems:
                                    # Рахуємо відстані до всієї існуючої БД
                                    distances = [np.linalg.norm(gem_vector - db_gem) for _, db_gem in historical_gems]
                                    min_idx = np.argmin(distances)
                                    min_dist = distances[min_idx]

                                    if min_dist < deduplication_threshold:
                                        is_duplicate = True
                                        duplicate_id = historical_gems[min_idx][0]

                                if is_duplicate:
                                    # Оновлюємо історичний запис (стабілізація в часі)
                                    self.db_manager.touch_landmark_metadata(duplicate_id)
                                    total_updated += 1
                                    last_saved_gem = historical_gems[min_idx][1]
                                else:
                                    # Записуємо новий унікальний орієнтир
                                    coordinates = (node["lon"], node["lat"], node["rel_alt"], node["yaw"])
                                    try:
                                        new_id = self.db_manager.insert_landmark_transaction(
                                            coordinates=coordinates,
                                            global_desc=gem_vector.tobytes(),
                                            local_features=filtered_descriptors.tobytes(),
                                            keypoints=filtered_keypoints.tobytes(),
                                            timestamp=current_timestamp,
                                        )
                                        last_saved_gem = gem_vector
                                        historical_gems.append((new_id, gem_vector))  # Додаємо в локальний кеш
                                        total_keyframes += 1
                                    except Exception as e:
                                        self.logger.error(f"Database insertion failed for node {node['id']}: {e}")

                        total_features += kpts_count
                        if kpts_count > 0:
                            total_landmarks += 1

                        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(img_rgb)

                # --- STATE CACHING ---
                progress_ratio = ((current_index + 1) / total_items) * 100
                self.last_update_data = {
                    "progress_ratio": progress_ratio,
                    "total_features": total_features,
                    "total_landmarks": total_landmarks,
                    "total_keyframes": total_keyframes,
                    "current_index": current_index + 1,
                    "total_items": total_items,
                    "frame_pil": pil_frame,
                }

                if self.active_view and self.active_view.winfo_exists():
                    self.active_view.after(
                        0,
                        self.active_view.ui_signal_process_update,
                        self.last_update_data,
                    )

            self.logger.info(
                f"Dataset processed. Inserted: {total_keyframes}. Updated (Temporal Stability): {total_updated}")

        except Exception as e:
            self.logger.error(f"Pipeline worker thread crashed: {e}", exc_info=True)

        finally:
            self.logger.info("Pipeline processing finished or terminated. Resetting flags.")
            self.is_running = False
            if self.active_view and self.active_view.winfo_exists():
                self.active_view.after(0, self.active_view.ui_signal_process_complete)