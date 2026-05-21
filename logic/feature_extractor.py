import sqlite3
import cv2
import numpy as np
import onnxruntime as ort


class UAVA_Producer:
    def __init__(
        self, model_path="xfeat_static_320.onnx", db_path="uav_map_database.db"
    ):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.db_path = db_path

        # Параметризація
        self.high = 320
        self.width = 320
        self.score_threshold = 0.005  # Поріг надійності точки
        self.max_features = 800  # Ліміт дескрипторів для БД
        self.gem_p = 3  # Степінь для GeM Pooling

        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS telemetry")
        cursor.execute("DROP TABLE IF EXISTS coordinates")
        cursor.execute("DROP TABLE IF EXISTS files")

        # Таблиця 1: Файли та дескриптори
        cursor.execute("""CREATE TABLE files (
                                                 id TEXT PRIMARY KEY,
                                                 filename TEXT,
                                                 keypoints BLOB,
                                                 descriptors BLOB,
                                                 global_desc BLOB
                          )""")

        # Таблиця 2: Координати
        cursor.execute("""CREATE TABLE coordinates (
                                                       id TEXT PRIMARY KEY,
                                                       lon REAL,
                                                       lat REAL,
                                                       FOREIGN KEY(id) REFERENCES files(id)
                          )""")

        # Таблиця 3: Телеметрія (використовуємо relative_altitude)
        cursor.execute("""CREATE TABLE telemetry (
                                                     id TEXT PRIMARY KEY,
                                                     altitude REAL,
                                                     roll REAL,
                                                     pitch REAL,
                                                     yaw REAL,
                                                     FOREIGN KEY(id) REFERENCES files(id)
                          )""")
        conn.close()

    def extract_features(self, image_path, data_row):
        """
        data_row: словник з телеметрією (id, lon, lat, rel_alt, roll, pitch, yaw)
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Препроцесинг та інференс
        img_res = cv2.resize(img, (self.width, self.high))
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), axis=0)
        outputs = self.session.run(None, {"input": tensor})

        desc_raw = outputs[0][0].reshape(64, -1).T
        kpts = outputs[1][0]
        scores = outputs[2][0].flatten()

        # 1. Нормалізація та фільтрація за Reliability Map
        desc = desc_raw / (np.linalg.norm(desc_raw, axis=1, keepdims=True) + 1e-6)

        idx = np.where(scores > self.score_threshold)[0]
        if len(idx) > self.max_features:
            idx = np.argsort(scores)[-self.max_features :]

        f_desc = desc[idx]
        f_kpts = kpts[idx]

        # 2. Розрахунок GeM Pooling (Глобальний дескриптор)
        # f = (1/N * sum(x^p))^(1/p)
        gem_vector = np.power(
            np.mean(np.power(f_desc, self.gem_p), axis=0), 1.0 / self.gem_p
        )
        gem_vector /= np.linalg.norm(gem_vector) + 1e-6

        # 3. Збереження в 3 таблиці
        self._save_to_db(data_row, f_kpts, f_desc, gem_vector)
        return len(idx)

    def _save_to_db(self, row, kpts, desc, g_desc):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        img_id = row["id"]

        # Вставка у files
        cursor.execute(
            "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
            (img_id, f"{img_id}.jpg", kpts.tobytes(), desc.tobytes(), g_desc.tobytes()),
        )

        # Вставка у coordinates
        cursor.execute(
            "INSERT INTO coordinates VALUES (?, ?, ?)", (img_id, row["lon"], row["lat"])
        )

        # Вставка у telemetry
        cursor.execute(
            "INSERT INTO telemetry VALUES (?, ?, ?, ?, ?)",
            (img_id, row["rel_alt"], row["roll"], row["pitch"], row["yaw"]),
        )

        conn.commit()
        conn.close()
