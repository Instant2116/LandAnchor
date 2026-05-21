import os
import cv2
import pandas as pd
import numpy as np
from typing import Any

# Assume these are imported from the utilities we discussed earlier
from telemetry_utils import calculate_azimuth, format_timestamp
from db.db_utils import array_to_blob


class DatasetOrchestrator:
    def __init__(self, db_manager: Any, xfeat_model: Any, settings_manager: Any):
        self.db_manager = db_manager
        self.model = xfeat_model

        params = settings_manager.get_cv_params()
        self.max_features = int(params.get("xfeatMaxFeatures", 500))

    def process_dataset_directory(
        self, base_dir: str, dataset_name: str = "AnYangRiver"
    ):
        """
        Reads the specific dataset structure, merges Excel files, and orchestrates extraction.
        """
        dataset_path = os.path.join(base_dir, dataset_name)
        drone_dir = os.path.join(dataset_path, "drone")

        main_xlsx = os.path.join(dataset_path, f"{dataset_name}.xlsx")
        dem_xlsx = os.path.join(dataset_path, "location_with_dem.xlsx")

        if not os.path.exists(main_xlsx) or not os.path.exists(dem_xlsx):
            print("Error: Missing telemetry Excel files in the target directory.")
            return

        # Read and merge telemetry data based on the 'id' column
        df_main = pd.read_excel(main_xlsx)
        df_dem = pd.read_excel(dem_xlsx)
        df_merged = pd.merge(df_main, df_dem, on="id", how="inner")

        previous_point = None

        for index, row in df_merged.iterrows():
            # Handle image filename extension safely
            img_id = str(row["id"])
            img_filename = (
                img_id if img_id.lower().endswith(".jpg") else f"{img_id}.jpg"
            )
            img_path = os.path.join(drone_dir, img_filename)

            if not os.path.exists(img_path):
                continue

            # Load image (already 320x320)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. XFeat Inference
            inference_results = self.model.process(img)
            scores = inference_results["scores"]

            # Confidence threshold and max feature limitation
            valid_indices = np.where(scores > 0.005)[0]
            if len(valid_indices) > self.max_features:
                valid_indices = np.argsort(scores)[-self.max_features :]

            if len(valid_indices) == 0:
                continue

            # 2. Extract and Serialize Data
            filtered_desc = inference_results["desc"][valid_indices]
            filtered_kpts = inference_results["kpts"][valid_indices]
            global_gem = inference_results["global"]

            blob_local = array_to_blob(filtered_desc)
            blob_kpts = array_to_blob(filtered_kpts)
            blob_global = array_to_blob(global_gem)

            # 3. Telemetry Processing
            current_point = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "yaw": float(row["yaw"]) if pd.notna(row["yaw"]) else None,
            }

            azimuth = calculate_azimuth(current_point, previous_point)
            previous_point = current_point

            timestamp = format_timestamp()

            # Using 'relative_altitude' as Z coordinate based on your location_with_dem.xlsx
            coords = (
                current_point["lon"],
                current_point["lat"],
                float(row["relative_altitude"]),
                azimuth,
            )

            # 4. Data Layer Transaction
            try:
                self.db_manager.insert_landmark_transaction(
                    coords, blob_global, blob_local, blob_kpts, timestamp
                )
            except Exception as e:
                print(f"Failed to insert record {img_filename}: {e}")
