import os
import csv
import time
import threading
import cv2
from PIL import Image, ImageTk

class OperatorManager:
    def __init__(self, db_manager, settings_manager):
        self.db_manager = db_manager
        self.settings_manager = settings_manager
        self.active_view = None
        self.is_running = False

    def register_view(self, view) -> None:
        """ Зберігає посилання на активний OperatorDashboardView """
        self.active_view = view

    def start_dataset_simulation(self, dataset_dir: str) -> None:
        """ Запускає симуляцію польоту у фоновому потоці. """
        if not self.active_view or self.is_running:
            return

        self.is_running = True
        worker = threading.Thread(target=self._simulation_loop, args=(dataset_dir,))
        worker.daemon = True
        worker.start()

    def stop_simulation(self) -> None:
        """ Зупиняє симуляцію. """
        self.is_running = False

    def _simulation_loop(self, dataset_dir: str) -> None:
        """ Фоновий процес: читає CSV, дістає картинки та генерує телеметрію. """
        csv_path = os.path.join(dataset_dir, "flight_telemetry.csv")
        if not os.path.exists(csv_path):
            self.is_running = False
            return

        
        self.active_view.after(0, self.active_view.ui_update_status, True, "DATASET SIMULATION")

        battery = 100.0
        start_time = time.time()

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
        except Exception:
            self.is_running = False
            return

        for row in reader:
            if not self.is_running:
                break

            
            lat = float(row.get("latitude", 0.0))
            lon = float(row.get("longitude", 0.0))
            alt = float(row.get("altitude", 0.0))
            yaw = float(row.get("yaw", 0.0))

            
            elapsed = int(time.time() - start_time)
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
            battery = max(0, battery - 0.05)

            
            img_id = row.get("id", "")
            img_path = os.path.join(dataset_dir, img_id)
            confidence = 0.0
            pil_img = None

            if os.path.exists(img_path):
                cv_img = cv2.imread(img_path)
                if cv_img is not None:
                    
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv_img)

                    
                    
                    
                    
                    confidence = 88.5  

            t_data = {
                "alt": alt,
                "spd": 12.4, 
                "hdg": yaw,
                "battery": f"{int(battery)}%",
                "time": time_str,
                "wind": "4.1 m/s",
                "confidence": confidence,
                "frame_pil": pil_img 
            }

            map_data = {"lat": lat, "lon": lon}

            
            self.active_view.after(0, self._update_ui_sync, t_data, map_data)

            
            time.sleep(0.1)

        self.is_running = False
        self.active_view.after(0, self.active_view.ui_update_status, False, "")

    def _update_ui_sync(self, t_data: dict, map_data: dict) -> None:
        """ Безпечне оновлення UI у головному потоці Tkinter. """
        if not self.active_view: return

        
        if t_data.get("frame_pil"):
            w = self.active_view.hud_canvas.winfo_width()
            h = self.active_view.hud_canvas.winfo_height()
            if w > 10 and h > 10:
                resized_img = t_data["frame_pil"].resize((w, h), Image.Resampling.LANCZOS)
                t_data["frame_tk"] = ImageTk.PhotoImage(resized_img)

        self.active_view.ui_update_telemetry(t_data)
        self.active_view.ui_update_map_canvas(map_data)