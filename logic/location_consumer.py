import cv2
import numpy as np
import logging
from typing import Any, Dict, Optional

from db.db_utils import blob_to_array
from logic.geometry_utils import verify_matches_ransac, extract_relative_rotation

# Initialize the logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Localization")


class LocationConsumer:
    def __init__(self, db_manager: Any, xfeat_model: Any, settings_manager: Any):
        self.db_manager = db_manager
        self.model = xfeat_model
        self.settings_manager = settings_manager  # Save reference to reload dynamically

        # Initial parameter load
        self._reload_parameters()

        # Step 1: In-memory cache for Global Descriptors
        self.global_cache = []
        self._load_global_cache()

    def _reload_parameters(self):
        """Dynamically pulls the latest parameters from the UI context."""
        cv_params = self.settings_manager.get_cv_params()
        self.max_features = int(cv_params.get("xfeatMaxFeatures", 500))
        self.conf_threshold = float(cv_params.get("xfeatConfidenceThreshold", 0.005))
        self.match_ratio = float(cv_params.get("matchRatio", 0.75))
        self.ransac_thresh = float(cv_params.get("ransacThreshold", 3.0))
        self.min_inliers = int(cv_params.get("minInliers", 15))
        self.top_k = int(cv_params.get("topKCandidates", 5))
        self.global_dist_thresh = float(cv_params.get("globalDistanceThreshold", 0.4))

        if hasattr(self.model, 'set_gem_p'):
            self.model.set_gem_p(int(cv_params.get("gemPoolingPower", 3)))

    def _load_global_cache(self) -> None:
        raw_globals = self.db_manager.get_all_global_descriptors()
        for l_id, gem_blob in raw_globals:
            db_gem = blob_to_array(gem_blob, dtype=np.float32)
            self.global_cache.append((l_id, db_gem))
        logger.info(f"Loaded {len(self.global_cache)} landmarks into Global Cache.")

    def localize(self, img_input: Any) -> Optional[Dict[str, Any]]:
        # Reload parameters to instantly apply UI slider changes
        self._reload_parameters()

        if isinstance(img_input, str):
            img = cv2.imread(img_input)
        else:
            img = img_input

        if img is None:
            return None

        img_res = cv2.resize(img, (320, 320))

        # Step 2: Extract Live Features
        inference = self.model.process(img_res)
        scores = inference["scores"].reshape(-1)
        desc = inference["desc"].reshape(-1, 64)
        kpts = inference["kpts"].reshape(-1, 2)
        live_gem = inference["global"].reshape(-1)

        valid_indices = np.where(scores > self.conf_threshold)[0]
        if len(valid_indices) > self.max_features:
            valid_indices = np.argsort(scores)[-self.max_features:]

        live_desc = desc[valid_indices]
        live_kpts = kpts[valid_indices]

        # Step 3: Fast Global Search (L2 Distance)
        if not self.global_cache:
            return None

        distances = []
        for l_id, db_gem in self.global_cache:
            dist = np.linalg.norm(live_gem - db_gem)
            distances.append((l_id, dist))

        distances.sort(key=lambda x: x[1])
        top_candidates = distances[: self.top_k]

        # --- THE FIX: GLOBAL DISTANCE THRESHOLD FILTER ---
        # If the ABSOLUTE best global match is further away mathematically
        # than the UI slider allows, abort early. This saves massive CPU time.
        if top_candidates[0][1] > self.global_dist_thresh:
            logger.warning(
                f"<<< EARLY EXIT: Best candidate GeM Distance ({top_candidates[0][1]:.4f}) exceeded threshold ({self.global_dist_thresh}).")
            return None

        best_match_data = None
        highest_inliers = 0

        # Step 4: Local Verification
        for rank, (landmark_id, dist) in enumerate(top_candidates):
            payload = self.db_manager.get_landmark_payload(landmark_id)
            if not payload:
                continue

            db_desc = blob_to_array(payload["local_features"], shape=(-1, 64))
            db_kpts = blob_to_array(payload["keypoints"], shape=(-1, 2))

            raw_matches = self._match_descriptors(live_desc, db_desc)

            inlier_count, _, pts_live, pts_db = verify_matches_ransac(
                live_kpts, db_kpts, raw_matches, self.ransac_thresh, self.min_inliers
            )

            # Step 5: Identify Best Match
            if inlier_count > highest_inliers and inlier_count >= self.min_inliers:
                highest_inliers = inlier_count

                yaw_delta = extract_relative_rotation(pts_live, pts_db)
                true_azimuth = (payload["yaw"] + yaw_delta) % 360.0

                best_match_data = {
                    "landmark_id": landmark_id,
                    "inliers": inlier_count,
                    "coordinates": {
                        "lon": payload["lon"],
                        "lat": payload["lat"],
                        "alt": payload["alt"],
                        "azimuth": round(true_azimuth, 2),
                    },
                }

        # Step 7: Return Coordinates and Rotation
        if best_match_data:
            self.db_manager.touch_landmark_metadata(best_match_data["landmark_id"])
            return best_match_data

        return None

    def _load_global_cache(self) -> None:
        raw_globals = self.db_manager.get_all_global_descriptors()
        for l_id, gem_blob in raw_globals:
            db_gem = blob_to_array(gem_blob, dtype=np.float32)
            self.global_cache.append((l_id, db_gem))

        logger.info(f"Loaded {len(self.global_cache)} landmarks into Global Cache.")

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        # Fast dot product for Cosine Similarity -> Euclidean Distance
        sim_matrix = np.dot(desc1, desc2.T)
        dist_matrix = np.sqrt(np.clip(2.0 - 2.0 * sim_matrix, 0, None))

        matches = []

        # 1. Find the top 2 nearest neighbors in DB for each live feature
        sorted_db_idx = np.argsort(dist_matrix, axis=1)

        # 2. Find the absolute best live feature for each DB feature (for MNN)
        best_live_idx_for_db = np.argmin(dist_matrix, axis=0)

        for live_idx in range(dist_matrix.shape[0]):
            best_db_idx = sorted_db_idx[live_idx, 0]
            second_best_db_idx = sorted_db_idx[live_idx, 1]

            dist_best = dist_matrix[live_idx, best_db_idx]
            dist_second = dist_matrix[live_idx, second_best_db_idx]

            # Protect against division by zero in perfectly identical edge cases
            if dist_second == 0:
                continue

            # 3. Lowe's Ratio Test: Reject ambiguous, repetitive features (like grass/water)
            if (dist_best / dist_second) < self.match_ratio:

                # 4. Mutual Nearest Neighbor Check: Ensure they both agree they are the best match
                if best_live_idx_for_db[best_db_idx] == live_idx:
                    matches.append([live_idx, best_db_idx])

        return matches