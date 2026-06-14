import cv2
import numpy as np
from typing import Any, Dict, Optional

from logic.geometry_utils import verify_matches_ransac, extract_relative_rotation
from logic.logger import SystemLogger


class LocationConsumer:
    def __init__(self, db_manager: Any, xfeat_model: Any, settings_manager: Any):
        self.db_manager = db_manager
        self.model = xfeat_model
        self.settings_manager = settings_manager
        self.logger = SystemLogger()

        self._reload_parameters()

        self.global_ids = []

        # Persistent FLANN index for global descriptors
        index_params = dict(algorithm=1, trees=4)
        search_params = dict(checks=128)
        self.global_flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Brute-force matcher for stateless local descriptor verification
        self.local_matcher = cv2.BFMatcher(cv2.NORM_L2)

        self._load_global_cache()

    def _reload_parameters(self):
        cv_params = self.settings_manager.get_cv_params()
        self.max_features = int(cv_params.get("xfeatMaxFeatures", 500))
        self.conf_threshold = float(cv_params.get("xfeatConfidenceThreshold", 0.005))
        self.match_ratio = float(cv_params.get("matchRatio", 0.75))
        self.ransac_thresh = float(cv_params.get("ransacThreshold", 3.0))
        self.min_inliers = int(cv_params.get("minInliers", 15))
        self.top_k = int(cv_params.get("topKCandidates", 5))
        self.global_dist_thresh = float(cv_params.get("globalDistanceThreshold", 0.4))

        if hasattr(self.model, "set_gem_p"):
            self.model.set_gem_p(int(cv_params.get("gemPoolingPower", 3)))

    def _load_global_cache(self) -> None:
        raw_globals = self.db_manager.get_all_global_descriptors()
        global_descs = []

        for l_id, gem_blob in raw_globals:

            db_gem = self.db_manager.blob_to_array(gem_blob, dtype=np.float32)
            db_gem = db_gem / (np.linalg.norm(db_gem) + 1e-8)

            self.global_ids.append(l_id)
            global_descs.append(db_gem)

        if global_descs:
            global_descs_array = np.vstack(global_descs).astype(np.float32)
            self.global_flann.add([global_descs_array])
            self.global_flann.train()

        self.logger.info(
            f"Loaded {len(self.global_ids)} spatial landmark payloads into FLANN index."
        )

    def localize(self, img_input: Any) -> Optional[Dict[str, Any]]:
        self._reload_parameters()

        if isinstance(img_input, str):
            img = cv2.imread(img_input)
        else:
            img = img_input

        if img is None:
            self.logger.warn(
                "Localization dropped a frame: Received null image buffer."
            )
            return None

        img_res = cv2.resize(img, (320, 320))
        inference = self.model.process(img_res)

        # normalization
        live_gem = inference["global"].reshape(-1)
        live_gem = live_gem / (np.linalg.norm(live_gem) + 1e-8)

        # Feature preparation
        valid_indices = np.where(inference["scores"].reshape(-1) > self.conf_threshold)[0]
        if len(valid_indices) > self.max_features:
            valid_indices = np.argsort(inference["scores"].reshape(-1))[
                -self.max_features :
            ]

        live_desc = inference["desc"].reshape(-1, 64)[valid_indices]
        live_kpts = inference["kpts"].reshape(-1, 2)[valid_indices]

        if not self.global_ids:
            return None

        # Global search via trained FLANN
        query_gem = np.float32(live_gem).reshape(1, -1)
        search_k = min(self.top_k, len(self.global_ids))

        matches = self.global_flann.knnMatch(query_gem, k=search_k)

        if not matches or not matches[0]:
            return None

        top_candidates = []
        for match in matches[0]:
            top_candidates.append((self.global_ids[match.trainIdx], match.distance))

        if top_candidates[0][1] > self.global_dist_thresh:
            return None

        # RANSAC and verification
        for landmark_id, dist in top_candidates:
            payload = self.db_manager.get_landmark_payload(landmark_id)
            if not payload:
                continue

            db_desc = self.db_manager.blob_to_array(payload["local_features"], shape=(-1, 64))
            db_kpts = self.db_manager.blob_to_array(payload["keypoints"], shape=(-1, 2))

            local_matches = self._match_descriptors(live_desc, db_desc)
            inlier_count, _, pts_live, pts_db = verify_matches_ransac(
                live_kpts, db_kpts, local_matches, self.ransac_thresh, self.min_inliers
            )

            if inlier_count >= self.min_inliers:
                yaw_delta = extract_relative_rotation(pts_live, pts_db)
                return {
                    "landmark_id": landmark_id,
                    "inliers": inlier_count,
                    "coordinates": {
                        "lon": payload["lon"],
                        "lat": payload["lat"],
                        "azimuth": round((payload["yaw"] + yaw_delta) % 360.0, 2),
                    },
                }

        return None

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        matches = self.local_matcher.knnMatch(np.float32(desc1), np.float32(desc2), k=2)
        return [
            [m.queryIdx, m.trainIdx]
            for m, n in matches
            if m.distance < self.match_ratio * n.distance
        ]
