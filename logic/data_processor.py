import os
import cv2
import threading
import datetime
import numpy as np
from typing import Any
from PIL import Image
from logic.xfeat_core import XFeatCore
from db.db_manager import DBManager


class DataProcessor:
    def __init__(self, db_manager: DBManager, settings_manager: Any):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.model = None

    def start_dataset_processing_pipeline(
            self, target_dir: str, view_callback: Any
    ) -> None:
        """
        Initializes the ingestion sequence by merging telemetry Excel files
        and spawns a background worker thread.
        """
        import pandas as pd

        print("Starting data processing pipeline...")
        print("target_dir: ", target_dir)

        # Dynamically determine the dataset name from the target directory path
        dataset_name = os.path.basename(os.path.normpath(target_dir))

        main_xlsx_path = os.path.join(target_dir, f"{dataset_name}.xlsx")
        dem_xlsx_path = os.path.join(target_dir, "location_with_dem.xlsx")
        model_path = "onnx/xfeat_static_320.onnx"

        if not os.path.exists(main_xlsx_path) or not os.path.exists(dem_xlsx_path):
            print("Required telemetry files do not exist.")
            print(f"Expected main xlsx: {main_xlsx_path}")
            print(f"Expected dem xlsx: {dem_xlsx_path}")
            return

        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            return

        if self.model is None:
            self.model = XFeatCore(model_path)

        processing_queue = []
        try:
            # Read and merge the Excel files on the 'id' column
            df_main = pd.read_excel(main_xlsx_path)
            df_dem = pd.read_excel(dem_xlsx_path)
            df_merged = pd.merge(df_main, df_dem, on="id", how="inner")

            for index, row in df_merged.iterrows():
                # Ensure the filename has a .jpg extension
                img_id = str(row["id"])
                img_filename = img_id if img_id.lower().endswith(".jpg") else f"{img_id}.jpg"

                # Route the target path to the 'drone' subdirectory
                relative_img_path = os.path.join("drone", img_filename)

                processing_queue.append(
                    {
                        "id": relative_img_path,  # Worker thread joins this with target_dir
                        "lon": float(row["lon"]),
                        "lat": float(row["lat"]),
                        "rel_alt": float(row["relative_altitude"]),
                        "yaw": float(row["yaw"]) if pd.notna(row["yaw"]) else 0.0,
                    }
                )
        except Exception as e:
            print(f"Exception during Excel processing: {e}")
            return

        if not processing_queue:
            print("Processing queue is empty. Aborting pipeline.")
            return

        # Lock the UI via callback
        view_callback.ui_signal_process_start()

        # Execute data processing on a separate thread to keep Tkinter responsive
        worker_thread = threading.Thread(
            target=self._pipeline_worker_thread,
            args=(target_dir, processing_queue, view_callback),
        )
        worker_thread.daemon = True
        worker_thread.start()

    def _pipeline_worker_thread(
            self, target_dir: str, queue: list, view_callback: Any
    ) -> None:
        """
        Executes ONNX processing and delegates database commits to the DBManager.
        Triggers thread-safe UI updates on the main thread via .after().
        """
        total_items = len(queue)
        total_features = 0
        total_landmarks = 0
        total_keyframes = 0
        cv_params = self.settings_manager.get_cv_params()
        max_features_limit = int(cv_params.get("xfeatMaxFeatures", 500))
        # Parameterize the threshold, defaulting to 0.005 if not explicitly set in the config
        confidence_threshold = float(cv_params.get("xfeatConfidenceThreshold", 0.005))

        for current_index, node in enumerate(queue):
            image_target_path = os.path.join(target_dir, node["id"])
            kpts_count = 0

            if os.path.exists(image_target_path):
                img = cv2.imread(image_target_path)
                if img is not None:
                    img_res = cv2.resize(img, (320, 320))

                    inference_results = self.model.process(img_res)

                    # Flatten all tensors to 1D/2D arrays BEFORE slicing
                    scores = inference_results["scores"].reshape(-1)
                    desc = inference_results["desc"].reshape(-1, 64)
                    kpts = inference_results["kpts"].reshape(-1, 2)

                    valid_indices = np.where(scores > 0.005)[0]
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
                            node["yaw"]
                        )

                        try:
                            # Delegate database insertion strictly to the Data Layer
                            self.db_manager.insert_landmark_transaction(
                                coordinates=coordinates,
                                global_desc=gem_vector.tobytes(),
                                local_features=filtered_descriptors.tobytes(),
                                keypoints=filtered_keypoints.tobytes(),
                                timestamp=current_timestamp
                            )
                        except Exception as e:
                            # Expose the error instead of failing silently
                            print(f"Database insertion failed for node {node['id']}: {e}")

            total_features += kpts_count
            if kpts_count > 0:
                total_landmarks += 1
            if current_index % 3 == 0 and kpts_count > 0:
                total_keyframes += 1

            progress_ratio = ((current_index + 1) / total_items) * 100

            # Convert OpenCV BGR to PIL RGB for the UI
            pil_frame = None
            if img_res is not None:
                img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(img_rgb)

            update_data = {
                "progress_ratio": progress_ratio,
                "total_features": total_features,
                "total_landmarks": total_landmarks,
                "total_keyframes": total_keyframes,
                "current_index": current_index + 1,
                "total_items": total_items,
                "frame_pil": pil_frame
            }

            view_callback.after(0, view_callback.ui_signal_process_update, update_data)

        view_callback.after(0, view_callback.ui_signal_process_complete)