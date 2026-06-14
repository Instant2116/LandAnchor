import sqlite3
from typing import Optional, Dict, Any
import numpy as np


class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_database(self) -> None:
        """
        Creates schema skeletons adhering to strict 3NF constraints.
        Formatted precisely to Tables 3.1 - 3.6 specifications.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Roles
                           (
                               id   INTEGER PRIMARY KEY AUTOINCREMENT,
                               name TEXT UNIQUE NOT NULL
                           );
                           """)

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Users
                           (
                               id            INTEGER PRIMARY KEY AUTOINCREMENT,
                               username      TEXT UNIQUE NOT NULL,
                               password_hash TEXT        NOT NULL,
                               role_id       INTEGER     NOT NULL,
                               FOREIGN KEY (role_id) REFERENCES Roles (id) ON DELETE RESTRICT
                           );
                           """)

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Landmarks
                           (
                               id           INTEGER PRIMARY KEY AUTOINCREMENT,
                               coordinate_x REAL NOT NULL,
                               coordinate_y REAL NOT NULL,
                               coordinate_z REAL NOT NULL,
                               azimuth      REAL NOT NULL
                           );
                           """)

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS GlobalDescriptors
                           (
                               landmark_id   INTEGER PRIMARY KEY,
                               global_vector BLOB NOT NULL,
                               FOREIGN KEY (landmark_id) REFERENCES Landmarks (id) ON DELETE CASCADE
                           );
                           """)

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS LocalFeatures
                           (
                               landmark_id    INTEGER PRIMARY KEY,
                               local_features BLOB NOT NULL,
                               keypoints      BLOB NOT NULL,
                               FOREIGN KEY (landmark_id) REFERENCES GlobalDescriptors (landmark_id) ON DELETE CASCADE
                           );
                           """)

            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS LandmarkMetadata
                           (
                               landmark_id INTEGER PRIMARY KEY,
                               created_at  TEXT NOT NULL,
                               FOREIGN KEY (landmark_id) REFERENCES LocalFeatures (landmark_id) ON DELETE CASCADE
                           );
                           """)

            # --- FIX: EXPLICIT VERIFICATION ---
            # Bypasses potential missing UNIQUE constraints in legacy database files
            for role in ['Operator', 'Technician']:
                cursor.execute("SELECT id FROM Roles WHERE name = ?;", (role,))
                if not cursor.fetchone():
                    cursor.execute("INSERT INTO Roles (name) VALUES (?);", (role,))

            conn.commit()

    def add_user_signature(
        self, username: str, hardware_hash: str, role_name: str
    ) -> bool:
        """
        Inserts or overwrites hardware registration tokens directly into local storage.
        Targeting the password_hash column as defined in Table 3.2.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM Roles WHERE name = ?;", (role_name,))
                role_row = cursor.fetchone()
                if not role_row:
                    return False

                role_id = role_row[0]

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO Users (username, password_hash, role_id)
                    VALUES (?, ?, ?);
                """,
                    (username, hardware_hash, role_id),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def verify_key_signature(
        self, calculated_fingerprint: str
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the database to find a user matching the provided hardware hash string.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT Users.username, Roles.name
                FROM Users
                         JOIN Roles ON Users.role_id = Roles.id
                WHERE Users.password_hash = ?;
                """,
                (calculated_fingerprint,),
            )
            row = cursor.fetchone()
            if row:
                return {"username": row[0], "role": row[1]}
            return None

    def get_database_filename_node(self) -> str:
        import os

        return os.path.basename(self.db_path)

    def get_active_database_metrics_report(self) -> Dict[str, Any]:
        """Compiles a statistical report of the current database state."""
        import os
        metrics = {
            "landmarks_count": "0",
            "global_descriptors_count": "0",
            "local_features_count": "0",
            "disk_size_string": "0.0 KB"
        }

        # Calculate live disk footprint
        if os.path.exists(self.db_path):
            size_kb = os.path.getsize(self.db_path) / 1024
            if size_kb > 1024:
                metrics["disk_size_string"] = f"{size_kb / 1024:.2f} MB"
            else:
                metrics["disk_size_string"] = f"{size_kb:.2f} KB"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM Landmarks;")
                metrics["landmarks_count"] = str(cursor.fetchone()[0])

                cursor.execute("SELECT COUNT(*) FROM GlobalDescriptors;")
                metrics["global_descriptors_count"] = str(cursor.fetchone()[0])

                cursor.execute("SELECT COUNT(*) FROM LocalFeatures;")
                metrics["local_features_count"] = str(cursor.fetchone()[0])
        except sqlite3.Error:
            pass  # Return defaults safely if tables haven't been created yet

        return metrics


    def get_all_global_descriptors(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT landmark_id, global_vector FROM GlobalDescriptors;")
            return cursor.fetchall()

    def get_landmark_payload(self, landmark_id: int) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                           SELECT L.coordinate_x,
                                  L.coordinate_y,
                                  L.coordinate_z,
                                  L.azimuth,
                                  LF.local_features,
                                  LF.keypoints
                           FROM Landmarks L
                                    JOIN LocalFeatures LF ON L.id = LF.landmark_id
                           WHERE L.id = ?;
                           """,
                (landmark_id,),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "lon": row[0],
                    "lat": row[1],
                    "alt": row[2],
                    "yaw": row[3],
                    "local_features": row[4],
                    "keypoints": row[5],
                }
            return {}

    def touch_landmark_metadata(self, landmark_id: int) -> None:
        import datetime

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                           UPDATE LandmarkMetadata
                           SET created_at = ?
                           WHERE landmark_id = ?;
                           """,
                (current_time, landmark_id),
            )
            conn.commit()

    @staticmethod
    def array_to_blob(arr: np.ndarray) -> bytes:
        """Converts a numpy array to bytes for SQLite BLOB storage."""
        return arr.tobytes()

    @staticmethod
    def blob_to_array(blob: bytes, dtype=np.float32, shape=None) -> np.ndarray:
        """Converts SQLite BLOB bytes back to a numpy array with an optional shape."""
        arr = np.frombuffer(blob, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    def insert_landmark_transaction(
        self,
        coordinates: tuple,
        global_desc: bytes,
        local_features: bytes,
        keypoints: bytes,
        timestamp: str,
    ) -> int:
        lon, lat, alt, azimuth = coordinates

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                           INSERT INTO Landmarks (coordinate_x, coordinate_y, coordinate_z, azimuth)
                           VALUES (?, ?, ?, ?);
                           """,
                (lon, lat, alt, azimuth),
            )

            landmark_id = cursor.lastrowid

            cursor.execute(
                """
                           INSERT INTO GlobalDescriptors (landmark_id, global_vector)
                           VALUES (?, ?);
                           """,
                (landmark_id, global_desc),
            )

            cursor.execute(
                """
                           INSERT INTO LocalFeatures (landmark_id, local_features, keypoints)
                           VALUES (?, ?, ?);
                           """,
                (landmark_id, local_features, keypoints),
            )

            cursor.execute(
                """
                           INSERT INTO LandmarkMetadata (landmark_id, created_at)
                           VALUES (?, ?);
                           """,
                (landmark_id, timestamp),
            )

            return landmark_id
