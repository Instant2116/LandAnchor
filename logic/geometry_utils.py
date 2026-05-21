import cv2
import numpy as np
from typing import List, Tuple


def verify_matches_ransac(
    live_kpts: np.ndarray,
    db_kpts: np.ndarray,
    matches: List[List[int]],
    ransac_threshold: float,
    min_inliers: int,
) -> Tuple[int, List[List[int]]]:
    """
    Applies RANSAC Homography to filter out geometrically inconsistent matches.
    """
    if len(matches) < min_inliers:
        return 0, []

    # Extract the matched coordinates
    # MNN matches format: [live_index, db_index]
    pts_live = np.float32([live_kpts[m[0]] for m in matches]).reshape(-1, 1, 2)
    pts_db = np.float32([db_kpts[m[1]] for m in matches]).reshape(-1, 1, 2)

    # Calculate Homography using RANSAC
    H, mask = cv2.findHomography(pts_live, pts_db, cv2.RANSAC, ransac_threshold)

    if H is None:
        return 0, []

    inliers_mask = mask.ravel().tolist()
    inlier_matches = [m for i, m in enumerate(matches) if inliers_mask[i] == 1]

    return len(inlier_matches), inlier_matches
