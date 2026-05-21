import cv2
import numpy as np
import os
import math


def transform_image(input_path, output_path=None, rotation_deg=None, tilt_deg=None):
    # Validate input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Error: The image at {input_path} was not found.")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Error: File exists but could not be decoded as an image.")

    rows, cols = img.shape[:2]
    result = img.copy()

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_transformed{ext}"

    # Apply Rotation
    if rotation_deg is not None:
        center = (cols / 2, rows / 2)
        # Positive values are counter-clockwise
        matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        result = cv2.warpAffine(result, matrix, (cols, rows))

    # Apply Tilt (Shear)
    if tilt_deg is not None:
        # Calculate shear factor based on angle
        # tan(0) = 0 (no tilt), tan(45) = 1 (equal shift)
        shear_factor = math.tan(math.radians(tilt_deg))

        # Affine matrix for horizontal shear:
        # [[1, shear_factor, 0],
        #  [0, 1,            0]]
        shear_matrix = np.float32([[1, shear_factor, 0],
                                   [0, 1, 0]])

        # Adjust canvas size to prevent clipping if necessary,
        # or keep (cols, rows) to maintain original dimensions
        result = cv2.warpAffine(result, shear_matrix, (cols, rows))

    # Save result
    cv2.imwrite(output_path, result)
    print(f"Success: Image saved to {output_path}")

# Example Usage:
transform_image("../transformation_output/_target.jpg",
                output_path="transformation_output/rotated_270.jpg",
                rotation_deg=90)