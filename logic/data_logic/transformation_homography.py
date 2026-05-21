import cv2
import numpy as np
import math
import os


def simulate_uav_wind_tilt(input_path, output_path=None, pitch=0, roll=0, yaw=0):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: The image at {input_path} was not found.")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Error: File could not be decoded.")

    h, w = img.shape[:2]

    # 1. Simulate Camera Intrinsics
    # We estimate the focal length based on image width to simulate a standard FoV
    focal_length = w

    # 2. Convert angles to radians
    rx = math.radians(pitch)  # UAV tilting nose up/down
    ry = math.radians(yaw)  # UAV rotating left/right
    rz = math.radians(roll)  # UAV banking left/right

    # 3. Compute 3D Rotation Matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ], dtype=np.float64)

    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ], dtype=np.float64)

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # Combined Rotation Matrix
    R = Rz @ Ry @ Rx

    # 4. Define the four corners of the image plane in 3D space
    # Center the coordinates so rotation happens around the middle of the image
    corners_3d = np.array([
        [-w / 2, -h / 2, 0],
        [w / 2, -h / 2, 0],
        [w / 2, h / 2, 0],
        [-w / 2, h / 2, 0]
    ], dtype=np.float64)

    # 5. Rotate the corners in 3D space
    rotated_corners = corners_3d @ R.T

    # 6. Project back to 2D space using the camera focal length
    pts2d = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        # Translate the plane away from the camera by 'focal_length' 
        # so it doesn't clip through the lens
        z = rotated_corners[i, 2] + focal_length

        # Prevent division by zero if pitch is extreme
        if z <= 0: z = 0.0001

        # Perspective projection formula
        pts2d[i, 0] = (rotated_corners[i, 0] * focal_length / z) + w / 2
        pts2d[i, 1] = (rotated_corners[i, 1] * focal_length / z) + h / 2

    # Original 2D corners
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # 7. Calculate Homography Matrix
    M = cv2.getPerspectiveTransform(pts1, pts2d)

    # 8. Adjust Matrix and Canvas to prevent clipping
    x_coords = pts2d[:, 0]
    y_coords = pts2d[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    tx = -x_min if x_min < 0 else 0
    ty = -y_min if y_min < 0 else 0

    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    M_adjusted = translation_matrix @ M

    new_w = int(x_max - x_min) if x_min < 0 else int(x_max)
    new_h = int(y_max - y_min) if y_min < 0 else int(y_max)

    # 9. Apply the final warp
    result = cv2.warpPerspective(img, M_adjusted, (new_w, new_h))

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_uav_tilt{ext}"

    cv2.imwrite(output_path, result)
    print(f"Success: Image saved to {output_path}")

# Example Usage:
# Nose pitches down by 45 degrees, banking right by 10 degrees
simulate_uav_wind_tilt("../transformation_output/_target.jpg", pitch=45, roll=10)