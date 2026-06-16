import os
import cv2
import threading
import datetime
import time
import numpy as np
import pandas as pd
from typing import Any, Callable, Optional
from PIL import Image

from logic.xfeat_core import XFeatCore
from logic.geometry_utils import verify_matches_ransac
from db.db_manager import DBManager
from logic.logger import SystemLogger


class DataProcessor:
    def __init__(self, db_manager: DBManager, settings_manager: Any):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.model = None
        self.logger = SystemLogger()

        self.is_running = False

        self.progress_hook: Optional[Callable[[dict], None]] = None
        self.completion_hook: Optional[Callable[[], None]] = None

    def register_hooks(self, progress_callback: Callable, completion_callback: Callable) -> None:
        """Registers agnostic callbacks to emit telemetry without coupling to a specific UI framework."""
        self.progress_hook = progress_callback
        self.completion_hook = completion_callback

    def start_dataset_processing_pipeline(self, target_dir: str) -> None:
        if self.is_running:
            self.logger.warn("Processing pipeline is currently locked by an active thread.")
            return

        self.logger.info(f"Initiating feature extraction pipeline for target directory: {target_dir}")
        telemetry_xlsx_path = os.path.join(target_dir, "telemetry.xlsx")
        model_path = "onnx/xfeat_static_320.onnx"

        if not os.path.exists(telemetry_xlsx_path):
            self.logger.error("Telemetry metadata files missing from target directory.")
            return

        if not os.path.exists(model_path):
            self.logger.error(f"CRITICAL: ONNX model weights not found at {model_path}")
            return

        if self.model is None:
            self.model = XFeatCore(model_path)

        processing_queue = []
        try:
            df_telemetry = pd.read_excel(telemetry_xlsx_path)
            required_cols = ["id", "lon", "lat", "relative_altitude", "yaw"]
            missing_cols = [col for col in required_cols if col not in df_telemetry.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns in telemetry file: {missing_cols}")
                raise ValueError(f"Missing required columns in telemetry file: {missing_cols}")

            for index, row in df_telemetry.iterrows():
                img_id = str(row["id"]).strip()
                # Add .jpg extension if missing
                img_filename = img_id if img_id.lower().endswith(".jpg") else f"{img_id}.jpg"
                relative_img_path = os.path.join("drone", img_filename)

                processing_queue.append({
                    "id": relative_img_path,
                    "lon": float(row["lon"]) if pd.notna(row.get("lon")) else 0.0,
                    "lat": float(row["lat"]) if pd.notna(row.get("lat")) else 0.0,
                    "rel_alt": float(row["relative_altitude"]) if pd.notna(row.get("relative_altitude")) else 0.0,
                    "yaw": float(row["yaw"]) if pd.notna(row.get("yaw")) else 0.0,
                })

        except Exception as e:
            self.logger.error(f"Telemetry data parsing failed: {e}")
            return

        if not processing_queue:
            self.logger.error("processing_queue is empty")
            return

        self.is_running = True

        worker_thread = threading.Thread(
            target=self._pipeline_worker_thread, args=(target_dir, processing_queue)
        )
        worker_thread.daemon = True
        worker_thread.start()

    def _pipeline_worker_thread(self, target_dir: str, queue: list) -> None:
        try:
            total_items = len(queue)

            # Neural network extraction counters
            total_features = 0
            total_landmarks = 0

            # Database hard-drive commit counters
            total_keyframes = 0
            total_updated = 0
            saved_features = 0
            saved_landmarks = 0

            skipped_frames = 0
            confidence_accumulator = 0.0

            local_matcher = cv2.BFMatcher(cv2.NORM_L2)

            self.logger.info("Loading historical generalized mean (GeM) descriptor states...")
            existing_globals_raw = self.db_manager.get_all_global_descriptors()
            historical_gems = []

            for l_id, gem_blob in existing_globals_raw:
                db_gem = self.db_manager.blob_to_array(gem_blob, dtype=np.float32).reshape(-1)
                db_gem = db_gem / (np.linalg.norm(db_gem) + 1e-8)
                historical_gems.append((l_id, db_gem))

            last_saved_desc = None
            last_saved_kpts = None

            for current_index, node in enumerate(queue):
                if not self.is_running:
                    break

                cv_params = self.settings_manager.get_cv_params()
                max_features_limit = int(cv_params.get("xfeatMaxFeatures", 2000))
                confidence_threshold = float(cv_params.get("xfeatConfidenceThreshold", 0.005))
                deduplication_threshold = float(cv_params.get("temporalDeduplicationThreshold", 0.6))
                match_ratio = float(cv_params.get("matchRatio", 0.75))
                ransac_thresh = float(cv_params.get("ransacThreshold", 3.0))
                min_inliers = int(cv_params.get("minInliers", 15))
                global_dist_thresh = float(cv_params.get("globalDistanceThreshold", 0.4))

                if hasattr(self.model, "set_gem_p"):
                    self.model.set_gem_p(int(cv_params.get("gemPoolingPower", 3)))

                image_target_path = os.path.join(target_dir, node["id"])
                kpts_count = 0
                img_res = None
                pil_frame = None

                if os.path.exists(image_target_path):
                    img = cv2.imread(image_target_path)

                    if img is not None:
                        img_res = img
                        #img_res = cv2.resize(img, (320, 320)) #used when images were sub standard
                        inference_results = self.model.process(img_res)

                        scores = inference_results["scores"].reshape(-1)
                        desc = inference_results["desc"].reshape(-1, 64)
                        kpts = inference_results["kpts"].reshape(-1, 2)

                        confidence_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
                        confidence_accumulator += confidence_score

                        gem_vector = inference_results["global"].reshape(-1)
                        gem_vector = gem_vector / (np.linalg.norm(gem_vector) + 1e-8)

                        valid_indices = np.where(scores > confidence_threshold)[0]
                        if len(valid_indices) > max_features_limit:
                            valid_indices = np.argsort(scores)[-max_features_limit:]

                        filtered_descriptors = desc[valid_indices]
                        filtered_keypoints = kpts[valid_indices]
                        kpts_count = len(valid_indices)

                        if kpts_count > 0:
                            is_keyframe = False

                            if last_saved_desc is None:
                                is_keyframe = True
                            else:
                                sim_matrix = np.dot(filtered_descriptors, last_saved_desc.T)
                                dist_matrix = np.sqrt(np.clip(2.0 - 2.0 * sim_matrix, 0, None))

                                sorted_db_idx = np.argsort(dist_matrix, axis=1)
                                best_live_idx_for_db = np.argmin(dist_matrix, axis=0)
                                matches = []

                                for live_idx in range(dist_matrix.shape[0]):
                                    if dist_matrix.shape[1] > 1:
                                        best_db_idx = sorted_db_idx[live_idx, 0]
                                        second_best_db_idx = sorted_db_idx[live_idx, 1]

                                        if dist_matrix[live_idx, second_best_db_idx] == 0:
                                            continue

                                        if (dist_matrix[live_idx, best_db_idx] / dist_matrix[live_idx, second_best_db_idx]) < match_ratio:
                                            if best_live_idx_for_db[best_db_idx] == live_idx:
                                                matches.append([live_idx, best_db_idx])

                                inlier_count = 0
                                if len(matches) >= min_inliers:
                                    inlier_count, _, _, _ = verify_matches_ransac(
                                        filtered_keypoints, last_saved_kpts, matches, ransac_thresh, min_inliers
                                    )
                                    overlap_tolerance = kpts_count * 0.80
                                    if inlier_count < overlap_tolerance:
                                        is_keyframe = True
                                else:
                                    is_keyframe = True

                            #  filter
                            if is_keyframe:
                                current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                is_duplicate = False
                                duplicate_id = None

                                if historical_gems:
                                    distances = [
                                        (db_id, np.linalg.norm(gem_vector - db_gem))
                                        for db_id, db_gem in historical_gems
                                    ]
                                    distances.sort(key=lambda x: x[1])
                                    top_candidates = distances[:5]

                                    for cand_id, cand_dist in top_candidates:
                                        # Apply global distance limit symmetrically with consumer
                                        if cand_dist > global_dist_thresh:
                                            continue

                                        payload = self.db_manager.get_landmark_payload(cand_id)
                                        if not payload:
                                            continue

                                        db_desc = self.db_manager.blob_to_array(payload["local_features"], dtype=np.float32).reshape(-1, 64)
                                        db_kpts = self.db_manager.blob_to_array(payload["keypoints"], dtype=np.float32).reshape(-1, 2)

                                        local_matches = local_matcher.knnMatch(np.float32(filtered_descriptors), np.float32(db_desc), k=2)
                                        good_matches = [
                                            [m.queryIdx, m.trainIdx]
                                            for m, n in local_matches
                                            if m.distance < match_ratio * n.distance
                                        ]

                                        if len(good_matches) >= min_inliers:
                                            inlier_count, _, _, _ = verify_matches_ransac(
                                                filtered_keypoints, db_kpts, good_matches, ransac_thresh, min_inliers
                                            )

                                            # Apply absolute inlier minimum symmetrically with consumer
                                            if inlier_count >= min_inliers:
                                                overlap_ratio = inlier_count / max(1, kpts_count)
                                                if overlap_ratio >= deduplication_threshold:
                                                    is_duplicate = True
                                                    duplicate_id = cand_id
                                                    break

                                if is_duplicate:
                                    self.db_manager.touch_landmark_metadata(duplicate_id)
                                    total_updated += 1
                                else:
                                    coordinates = (node["lon"], node["lat"], node["rel_alt"], node["yaw"])
                                    try:
                                        new_id = self.db_manager.insert_landmark_transaction(
                                            coordinates=coordinates,
                                            global_desc=gem_vector.tobytes(),
                                            local_features=filtered_descriptors.tobytes(),
                                            keypoints=filtered_keypoints.tobytes(),
                                            timestamp=current_timestamp,
                                        )
                                        last_saved_desc = filtered_descriptors
                                        last_saved_kpts = filtered_keypoints
                                        historical_gems.append((new_id, gem_vector))

                                        # Track verified disk commits
                                        total_keyframes += 1
                                        saved_features += kpts_count
                                        saved_landmarks += 1

                                    except Exception as e:
                                        self.logger.error(f"Database transaction failed during landmark insertion for tensor node {node['id']}: {e}")

                        total_features += kpts_count
                        if kpts_count > 0:
                            total_landmarks += 1

                        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(img_rgb)
                    else:
                        skipped_frames += 1
                else:
                    skipped_frames += 1

                progress_ratio = ((current_index + 1) / total_items) * 100
                avg_confidence = (confidence_accumulator / (current_index + 1)) * 100

                payload = {
                    "progress_ratio": progress_ratio,
                    "total_features": total_features,
                    "total_landmarks": total_landmarks,
                    "saved_features": saved_features,
                    "saved_landmarks": saved_landmarks,
                    "total_keyframes": total_keyframes,
                    "skipped_frames": skipped_frames,
                    "avg_confidence": avg_confidence,
                    "current_index": current_index + 1,
                    "total_items": total_items,
                    "frame_pil": pil_frame,
                }

                if self.progress_hook:
                    self.progress_hook(payload)

                time.sleep(0.005)

            self.logger.info(f"Pipeline execution finalized. Keyframes inserted: {total_keyframes}. Temporal duplicates deduplicated: {total_updated}.")

        except Exception as e:
            self.logger.error(f"Pipeline worker thread encountered a fatal exception: {e}")

        finally:
            self.logger.info("Pipeline resource locks released.")
            self.is_running = False
            if self.completion_hook:
                self.completion_hook()