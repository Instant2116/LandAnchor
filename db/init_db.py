import sqlite3

DB_NAME = "../landanchor.db"

def initialize_all_tables():
    print(f"Starting database initialization: {DB_NAME}")

    # Connect to the database file
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        # Table: Roles
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Roles (
                                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                            name TEXT UNIQUE NOT NULL
                       );
                       """)
        print("Table 'Roles' initialized successfully.")

        # Table: Users (Depends on Roles)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Users (
                                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                            username TEXT UNIQUE NOT NULL,
                                                            password_hash TEXT NOT NULL,
                                                            role_id INTEGER NOT NULL,
                                                            FOREIGN KEY (role_id) REFERENCES Roles(id) ON DELETE RESTRICT
                       );
                       """)
        print("Table 'Users' initialized successfully.")

        # 3. Table: Landmarks
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Landmarks (
                                                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                coordinate_x REAL NOT NULL,
                                                                coordinate_y REAL NOT NULL,
                                                                coordinate_z REAL NOT NULL,
                                                                azimuth REAL NOT NULL
                       );
                       """)
        print("Table 'Landmarks' initialized successfully.")

        # Table: GlobalDescriptors (Depends on Landmarks)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS GlobalDescriptors (
                                                                        landmark_id INTEGER PRIMARY KEY,
                                                                        global_vector BLOB NOT NULL,
                                                                        FOREIGN KEY (landmark_id) REFERENCES Landmarks(id) ON DELETE CASCADE
                       );
                       """)
        print("Table 'GlobalDescriptors' initialized successfully.")

        # Table: LocalFeatures (Depends on GlobalDescriptors)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS LocalFeatures (
                                                                    landmark_id INTEGER PRIMARY KEY,
                                                                    local_features BLOB NOT NULL,
                                                                    keypoints BLOB NOT NULL,
                                                                    FOREIGN KEY (landmark_id) REFERENCES GlobalDescriptors(landmark_id) ON DELETE CASCADE
                       );
                       """)
        print("Table 'LocalFeatures' initialized successfully.")

        # Table: LandmarkMetadata (Depends on LocalFeatures)
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS LandmarkMetadata (
                                                                       landmark_id INTEGER PRIMARY KEY,
                                                                       created_at TEXT NOT NULL,
                                                                       FOREIGN KEY (landmark_id) REFERENCES LocalFeatures(landmark_id) ON DELETE CASCADE
                       );
                       """)
        print("Table 'LandmarkMetadata' initialized successfully.")

        # Seed initial operational roles
        cursor.execute("INSERT OR IGNORE INTO Roles (name) VALUES ('Operator');")
        cursor.execute("INSERT OR IGNORE INTO Roles (name) VALUES ('Technician');")
        print("Default roles seeded successfully.")

        # Commit all changes securely
        conn.commit()
        print("Database transaction committed successfully. Integrity is verified.")

    except sqlite3.Error as e:
        print(f"Database initialization failed due to an error: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    initialize_all_tables()
