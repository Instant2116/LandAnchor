import sqlite3

import cv2
import numpy as np
import onnxruntime as ort


class UAVB_Consumer:
    def __init__(
        self, model_path="xfeat_static_320.onnx", db_path="uav_map_database.db"
    ):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.db_path = db_path

        self.high = 320
        self.width = 320

    def _descriptors(self, desc1, desc2, threshold=0.8):

        dist_matrix = np.linalg.norm(desc1[:, np.newaxis] - desc2, axis=2)

        dist_matrix = np.linalg.norm(desc1[:, np.newaxis] - desc2, axis=2)
        idx1 = np.argmin(dist_matrix, axis=1)
        min_dist1 = np.min(dist_matrix, axis=1)
        idx2 = np.argmin(dist_matrix, axis=0)

        es = []
        for i, j in enumerate(idx1):
            if idx2[j] == i and min_dist1[i] < threshold:
                es.append([i, j])
        return np.array(es)

    def localize(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        # OpenCV resize: (width, height)
        img_res = cv2.resize(img, (self.width, self.high))
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)

        outputs = self.session.run(None, {"input": tensor})

        # 1. Отримуємо дескриптори
        live_desc = outputs[0][0] # Формат (64, 40, 40)

        # 2. Розгортаємо у (1600, 64)
        live_desc = live_desc.reshape(64, -1).T

        # 3. Додаємо L2 нормалізацію (робить вектори довжиною 1)
        live_desc = live_desc / (np.linalg.norm(live_desc, axis=1, keepdims=True) + 1e-6)

        return self._search_db(live_desc)

    def _search_db(self, live_desc):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT filename, descriptors FROM land_anchors")
        results = []
        for db_name, db_desc_raw in cursor:
            db_desc = np.frombuffer(db_desc_raw, dtype=np.float32).reshape(-1, 64)

            es = self._descriptors(live_desc, db_desc)
            results.append((db_name, len(es)))

        conn.close()
        return results
