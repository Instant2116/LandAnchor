import os
import csv
import cv2
import threading
import datetime
import numpy as np
from typing import Any

from xfeat_core import XFeatCore
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
        Initializes the ingestion sequence and spawns a background worker thread
        to process the dataset asynchronously.
        """
        csv_file_path = os.path.join(target_dir, "flight_telemetry.csv")
        model_path = "xfeat_static_320.onnx"

        if not os.path.exists(csv_file_path) or not os.path.exists(model_path):
            # Abort if critical files are missing
            return

        if self.model is None:
            self.model = XFeatCore(model_path)

        processing_queue = []
        try:
            with open(csv_file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processing_queue.append(
                        {
                            "id": row["id"],
                            "lon": float(row["longitude"]),
                            "lat": float(row["latitude"]),
                            "rel_alt": float(row["altitude"]),
                            "yaw": float(row["yaw"]),
                        }
                    )
        except Exception:
            return

        if not processing_queue:
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
        Executes ONNX processing and commits results to the local database container.
        Triggers thread-safe UI updates on the main thread via .after().
        """
        total_items = len(queue)
        total_features = 0
        total_landmarks = 0
        total_keyframes = 0

        cv_params = self.settings_manager.get_cv_params()
        max_features_limit = int(cv_params.get("xfeatMaxFeatures", 500))

        for current_index, node in enumerate(queue):
            image_target_path = os.path.join(target_dir, node["id"])
            kpts_count = 0

            if os.path.exists(image_target_path):
                img = cv2.imread(image_target_path)
                if img is not None:
                    img_res = cv2.resize(img, (320, 320))

                    # Leverage the uploaded xfeat_core.py logic
                    inference_results = self.model.process(img_res)

                    desc = inference_results["desc"]
                    kpts = inference_results["kpts"]
                    scores = inference_results["scores"]

                    # Filter based on project parameter limit constraints
                    valid_indices = np.where(scores > 0.005)[0]
                    if len(valid_indices) > max_features_limit:
                        valid_indices = np.argsort(scores)[-max_features_limit:]

                    filtered_descriptors = desc[valid_indices]
                    filtered_keypoints = kpts[valid_indices]
                    kpts_count = len(valid_indices)

                    if kpts_count > 0:
                        gem_vector = inference_results["global"]

                        try:
                            with self.db_manager._get_connection() as conn:
                                cursor = conn.cursor()

                                cursor.execute(
                                    """
                                               INSERT INTO Landmarks (coordinate_x, coordinate_y, coordinate_z, azimuth)
                                               VALUES (?, ?, ?, ?);
                                               """,
                                    (
                                        node["lon"],
                                        node["lat"],
                                        node["rel_alt"],
                                        node["yaw"],
                                    ),
                                )
                                landmark_id = cursor.lastrowid

                                cursor.execute(
                                    """
                                               INSERT INTO GlobalDescriptors (landmark_id, global_vector)
                                               VALUES (?, ?);
                                               """,
                                    (landmark_id, gem_vector.tobytes()),
                                )

                                cursor.execute(
                                    """
                                               INSERT INTO LocalFeatures (landmark_id, local_features, keypoints)
                                               VALUES (?, ?, ?);
                                               """,
                                    (
                                        landmark_id,
                                        filtered_descriptors.tobytes(),
                                        filtered_keypoints.tobytes(),
                                    ),
                                )

                                current_timestamp = datetime.datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                cursor.execute(
                                    """
                                               INSERT INTO LandmarkMetadata (landmark_id, created_at)
                                               VALUES (?, ?);
                                               """,
                                    (landmark_id, current_timestamp),
                                )

                                conn.commit()
                        except Exception:
                            pass

            total_features += kpts_count
            if kpts_count > 0:
                total_landmarks += 1
            if current_index % 3 == 0 and kpts_count > 0:
                total_keyframes += 1

            progress_ratio = ((current_index + 1) / total_items) * 100

            update_data = {
                "progress_ratio": progress_ratio,
                "total_features": total_features,
                "total_landmarks": total_landmarks,
                "total_keyframes": total_keyframes,
                "current_index": current_index + 1,
                "total_items": total_items,
            }

            # Safely schedule the UI update to run on the main Tkinter thread
            view_callback.after(0, view_callback.ui_signal_process_update, update_data)

        # Notify completion
        view_callback.after(0, view_callback.ui_signal_process_complete)
