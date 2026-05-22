import cv2
import numpy as np
import os


class ImageStandardizer:
    def __init__(self, target_size=(640, 480)):
        self.target_w, self.target_h = target_size
        self.input_dir = "raw_maps"  # Put your original screenshots here
        self.output_dir = "../../uavA_input"  # Standardized images go here

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"No files in {self.input_dir}. Please add screenshots.")
            return

        for filename in files:
            img = cv2.imread(os.path.join(self.input_dir, filename))
            if img is None:
                continue

            # 1. Calculate Scaling
            h, w = img.shape[:2]
            scale = min(self.target_w / w, self.target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # 2. Resize
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 3. Create Letterbox (Canvas)
            canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
            x_offset = (self.target_w - new_w) // 2
            y_offset = (self.target_h - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

            # 4. Save
            cv2.imwrite(os.path.join(self.output_dir, filename), canvas)
            print(f"Standardized: {filename} -> {self.output_dir}")


if __name__ == "__main__":
    standardizer = ImageStandardizer()
    standardizer.run()