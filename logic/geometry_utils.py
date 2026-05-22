import cv2
import numpy as np
import math
from typing import List, Tuple

def extract_relative_rotation(
        pts_live: np.ndarray,
        pts_db: np.ndarray
) -> float:
    """
    Calculates the 2D rotation angle between two sets of matched points.
    Assumes nadir view where transformation is primarily affine (scale, rotation, translation).
    """
    # Estimate a partial 2D affine transformation (4 DOF)
    matrix, _ = cv2.estimateAffinePartial2D(pts_db, pts_live)

    if matrix is None:
        return 0.0

    # Extract rotation from the matrix elements: atan2(M[1,0], M[0,0])
    angle_radians = math.atan2(matrix[1, 0], matrix[0, 0])
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
    """
    if len(matches) < min_inliers:
        return 0, [], np.array([]), np.array([])

    pts_live = np.float32([live_kpts[m[0]] for m in matches]).reshape(-1, 1, 2)
    pts_db = np.float32([db_kpts[m[1]] for m in matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(pts_live, pts_db, cv2.RANSAC, ransac_threshold)

    if homography is None:
        return 0, [], np.array([]), np.array([])

    inliers_mask = mask.ravel().tolist()
    inlier_matches = []
    inlier_pts_live = []
    inlier_pts_db = []

    for i, m in enumerate(matches):
        if inliers_mask[i] == 1:
            inlier_matches.append(m)
            inlier_pts_live.append(pts_live[i])
            inlier_pts_db.append(pts_db[i])

    return len(inlier_matches), inlier_matches, np.array(inlier_pts_live), np.array(inlier_pts_db)