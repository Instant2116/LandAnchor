import cv2
import numpy as np
from typing import Any, Dict, Optional

# Assuming utilities are imported
from db_utils import blob_to_array
from geometry_utils import verify_matches_ransac


class LocationConsumer:
    def __init__(self, db_manager: Any, xfeat_model: Any, settings_manager: Any):
        self.db_manager = db_manager
        self.model = xfeat_model

        params = settings_manager.get_cv_params()
        self.max_features = int(params.get("xfeatMaxFeatures", 500))
        self.match_ratio = float(params.get("matchRatio", 0.75))
        self.ransac_thresh = float(params.get("ransacThreshold", 3.0))
        self.min_inliers = int(params.get("minInliers", 15))

        # How many candidates to verify locally after GeM search
        self.top_k_candidates = 5

    def localize(self, img_path: str) -> Optional[Dict[str, Any]]:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img_res = cv2.resize(img, (320, 320))

        # 1. Inference and Confidence Filtering
        inference_results = self.model.process(img_res)
        scores = inference_results["scores"]

        valid_indices = np.where(scores > 0.005)[0]
        if len(valid_indices) > self.max_features:
            valid_indices = np.argsort(scores)[-self.max_features :]

        live_desc = inference_results["desc"][valid_indices]
        live_kpts = inference_results["kpts"][valid_indices]
        live_gem = inference_results["global"]

        # 2. Fast Global (GeM) Search
        db_global_data = self.db_manager.get_all_global_descriptors()
        if not db_global_data:
            return None

        candidates = []
        for l_id, gem_blob in db_global_data:
            db_gem = blob_to_array(gem_blob, dtype=np.float32)
            # L2 distance between normalized GeM vectors
            dist = np.linalg.norm(live_gem - db_gem)
            candidates.append((l_id, dist))

        # Sort by lowest distance and take Top K
        candidates.sort(key=lambda x: x[1])
        top_candidates = candidates[: self.top_k_candidates]

        # 3. Local Feature Matching & RANSAC Verification
        best_match = None
        highest_inliers = 0

        for landmark_id, _ in top_candidates:
            payload = self.db_manager.get_landmark_payload(landmark_id)
            if not payload:
                continue

            db_desc = blob_to_array(payload["local_features"], shape=(-1, 64))
            db_kpts = blob_to_array(payload["keypoints"], shape=(-1, 2))

            # Mutual Nearest Neighbor
            raw_matches = self._match_descriptors(live_desc, db_desc)

            # RANSAC
            inlier_count, _ = verify_matches_ransac(
                live_kpts, db_kpts, raw_matches, self.ransac_thresh, self.min_inliers
            )

            if inlier_count > highest_inliers and inlier_count >= self.min_inliers:
                highest_inliers = inlier_count
                best_match = {
                    "landmark_id": landmark_id,
                    "inliers": inlier_count,
                    "coordinates": {
                        "lon": payload["lon"],
                        "lat": payload["lat"],
                        "alt": payload["alt"],
                        "yaw": payload["yaw"],
                    },
                }

        # 4. Finalizing
        if best_match:
            # Update the last matched timestamp via Data Layer
            self.db_manager.touch_landmark_metadata(best_match["landmark_id"])
            return best_match

        return None

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        dist_matrix = np.linalg.norm(desc1[:, np.newaxis] - desc2, axis=2)
        idx1 = np.argmin(dist_matrix, axis=1)
        min_dist1 = np.min(dist_matrix, axis=1)
        idx2 = np.argmin(dist_matrix, axis=0)

        matches = []
        for i, j in enumerate(idx1):
            if idx2[j] == i and min_dist1[i] < self.match_ratio:
                matches.append([i, j])

        return matches
