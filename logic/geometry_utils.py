import cv2
import numpy as np
import math
from typing import List, Tuple

def extract_relative_rotation(
        pts_live: np.ndarray,
        pts_db: np.ndarray
) -> float:
    """
    Calculates the 2D rotation angle using a full 8-DOF Homography matrix.
    Extracts the yaw by projecting basis vectors to account for perspective distortion.
    """
    if pts_live.size == 0 or pts_db.size == 0 or len(pts_live) < 4 or len(pts_db) < 4:
        return 0.0

    matrix, _ = cv2.findHomography(pts_db, pts_live, 0)

    if matrix is None:
        return 0.0

    # Project standard basis vectors to account for perspective scaling
    p0 = np.array([0.0, 0.0, 1.0])
    p1 = np.array([1.0, 0.0, 1.0])

    p0_proj = matrix @ p0
    p1_proj = matrix @ p1

    # Normalize by homogeneous coordinate
    p0_proj /= p0_proj[2]
    p1_proj /= p1_proj[2]

    dx = p1_proj[0] - p0_proj[0]
    dy = p1_proj[1] - p0_proj[1]

    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def verify_matches_ransac(
        live_kpts: np.ndarray,
        db_kpts: np.ndarray,
        matches: List[List[int]],
        ransac_threshold: float,
        min_inliers: int
) -> Tuple[int, List[List[int]], np.ndarray, np.ndarray]:
    """
    Filters matches using RANSAC and returns the aligned point arrays for the inliers.
    Uses vectorized NumPy operations for coordinate extraction.
    """
    if len(matches) < min_inliers:
        return 0, [], np.array([]), np.array([])

    matches_arr = np.array(matches)
    query_idxs = matches_arr[:, 0]
    train_idxs = matches_arr[:, 1]

    # Advanced indexing to eliminate Python list comprehensions
    pts_live = live_kpts[query_idxs].astype(np.float32).reshape(-1, 1, 2)
    pts_db = db_kpts[train_idxs].astype(np.float32).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(pts_live, pts_db, cv2.RANSAC, ransac_threshold)

    if homography is None:
        return 0, [], np.array([]), np.array([])

    # Boolean masking for inlier extraction
    inliers_bool = mask.ravel() == 1

    inlier_matches = matches_arr[inliers_bool].tolist()
    inlier_pts_live = pts_live[inliers_bool]
    inlier_pts_db = pts_db[inliers_bool]

    return len(inlier_matches), inlier_matches, inlier_pts_live, inlier_pts_db
