import tkinter as tk
from tkinter import ttk


class OperatorDashboardView(tk.Frame):
    def __init__(self, parent, controller):
        t = controller.config["theme"]
        super().__init__(parent, bg=t["bg_primary"])
        self.controller = controller
        self.t = t

        self.flight_path = []
        self.anchor_points = []

        title_strip = tk.Frame(self, bg=t["bg_primary"])
        title_strip.pack(fill="x", padx=24, pady=20)
        tk.Label(
            title_strip,
            text="Dataset Processing Dashboard",
            fg=t["text_primary"],
            bg=t["bg_primary"],
            font=("Arial", 22, "bold"),
            anchor="w",
        ).pack(fill="x")

        self.mode_subtitle = tk.Label(
            title_strip,
            text="Awaiting Dataset Feed",
            fg=t["text_status"],
            bg=t["bg_primary"],
            font=("Arial", 10),
            anchor="w",
        )
        self.mode_subtitle.pack(fill="x")

        main_layout = tk.Frame(self, bg=t["bg_primary"])
        main_layout.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        main_layout.grid_columnconfigure(0, weight=2)
        main_layout.grid_columnconfigure(1, weight=1)
        main_layout.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(main_layout, bg=t["bg_primary"])
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right_col = tk.Frame(main_layout, bg=t["bg_primary"])
        right_col.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        video_card = tk.Frame(
            left_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        video_card.pack(fill="both", expand=True, pady=(0, 12))

        v_title_bar = tk.Frame(video_card, bg=t["bg_secondary"])
        v_title_bar.pack(fill="x", padx=16, pady=12)
        tk.Label(
            v_title_bar,
            text="Dataset Frame Feed",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        self.hud_canvas = tk.Canvas(
            video_card, bg=t["bg_primary"], bd=0, highlightthickness=0
        )
        self.hud_canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        map_card = tk.Frame(
            left_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        map_card.pack(fill="both", expand=True, pady=(12, 0))
        tk.Label(
            map_card,
            text="Coordinate Tracking Map",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=12)

        self.map_canvas = tk.Canvas(
            map_card, bg=t["bg_primary"], bd=0, highlightthickness=0
        )
        self.map_canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        conf_card = tk.Frame(
            right_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        conf_card.pack(fill="x", pady=(0, 12), ipady=10)
        tk.Label(
            conf_card,
            text="Localization Confidence",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        c_labels = tk.Frame(conf_card, bg=t["bg_secondary"])
        c_labels.pack(fill="x", padx=16)
        tk.Label(
            c_labels,
            text="Inlier Ratio",
            fg=t["text_status"],
            bg=t["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")
        self.conf_percent_lbl = tk.Label(
            c_labels,
            text="0%",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        )
        self.conf_percent_lbl.pack(side="right")

        self.conf_progress = ttk.Progressbar(
            conf_card, orient="horizontal", mode="determinate"
        )
        self.conf_progress.pack(fill="x", padx=16, pady=10)
        self.conf_progress["value"] = 0.0

        self.conf_status_frame = tk.Frame(conf_card, bg=t["bg_secondary"])
        self.conf_status_frame.pack(fill="x", padx=16)
        self.conf_status_txt = tk.Label(
            self.conf_status_frame,
            text="Awaiting data",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 9),
        )
        self.conf_status_txt.pack(side="left")

        self.hud_canvas.bind("<Configure>", lambda e: self.ui_update_hud_canvas({}))
        self.map_canvas.bind("<Configure>", lambda e: self.ui_update_map_canvas({}))

        self.controller.register_active_operator_view(self)

        conn_btn = tk.Button(
            title_strip,
            text="Load Dataset",
            bg=t["bg_tertiary"],
            fg=t["text_primary"],
            bd=1,
            relief="solid",
            font=("Arial", 9),
            cursor="hand2",
            command=self._on_connect_clicked,
        )
        conn_btn.pack(side="right", padx=16)

    def _on_connect_clicked(self) -> None:
        from tkinter import filedialog

        target_dir = filedialog.askdirectory(
            title="Select Flight Mission Dataset Root Folder"
        )
        if target_dir:
            self.controller.start_operator_simulation(target_dir)

    def ui_update_telemetry(self, t_data: dict) -> None:
        conf = float(t_data.get("confidence", 0.0))
        self.conf_percent_lbl.configure(text=f"{int(conf)}%")
        self.conf_progress["value"] = conf

        if conf >= 80:
            self.conf_status_txt.configure(
                text="High Confidence Match", fg=self.t["accent_green"]
            )
        elif conf >= 50:
            self.conf_status_txt.configure(
                text="Moderate Confidence Match", fg=self.t["accent_yellow"]
            )
        elif conf > 0:
            self.conf_status_txt.configure(
                text="Low Confidence Match", fg=self.t["accent_red"]
            )
        else:
            self.conf_status_txt.configure(
                text="No Match Found", fg=self.t["text_muted"]
            )

        self.ui_update_hud_canvas(t_data)

    def ui_update_status(self, is_active: bool, mode_text: str) -> None:
        if is_active:
            self.mode_subtitle.configure(
                text=f"Status: {mode_text}", fg=self.t["accent_blue"]
            )
        else:
            self.mode_subtitle.configure(text="Status: Idle", fg=self.t["text_status"])

    def ui_update_hud_canvas(self, t_data: dict) -> None:
        self.hud_canvas.delete("all")
        w, h = self.hud_canvas.winfo_width(), self.hud_canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if t_data and t_data.get("frame_tk"):
            self._current_frame = t_data["frame_tk"]
            self.hud_canvas.create_image(0, 0, anchor="nw", image=self._current_frame)
        elif not t_data:
            self.hud_canvas.create_text(
                w / 2,
                h / 2,
                text="AWAITING FRAMES",
                fill=self.t["text_muted"],
                font=("Courier", 10, "bold"),
            )
            return

        if "alt" in t_data or "hdg" in t_data:
            self.hud_canvas.create_text(
                20,
                20,
                text=f"ALT: {t_data.get('alt', 0.0):.1f}m\nYAW: {t_data.get('hdg', 0.0):.1f}°",
                fill=self.t["accent_green"],
                font=("Courier", 11, "bold"),
                anchor="nw",
            )

    def ui_update_map_canvas(self, map_data: dict) -> None:
        self.map_canvas.delete("all")
        w, h = self.map_canvas.winfo_width(), self.map_canvas.winfo_height()
        if w < 10 or h < 10:
            return

        # Draw grid background
        for x in range(0, w, 50):
            self.map_canvas.create_line(x, 0, x, h, fill=self.t["border_color"], width=1, dash=(1, 5))
        for y in range(0, h, 50):
            self.map_canvas.create_line(0, y, w, y, fill=self.t["border_color"], width=1, dash=(1, 5))

        # Handle successful localization
        if map_data and "lat" in map_data:
            lat, lon = float(map_data["lat"]), float(map_data["lon"])
            self.flight_path.append((lat, lon))
            if len(self.flight_path) > 1000:
                self.flight_path.pop(0)

        # Handle empty flight path (System just started)
        if not self.flight_path:
            self.map_canvas.create_text(
                w / 2, h / 2, text="AWAITING INITIAL FIX", fill=self.t["text_muted"], font=("Courier", 10)
            )
            return

        # Map Scaling Algorithm (Dynamic Auto-Zoom)
        all_lats = [p[0] for p in self.flight_path]
        all_lons = [p[1] for p in self.flight_path]

        if len(all_lats) > 1:
            min_lat, max_lat = min(all_lats), max(all_lats)
            min_lon, max_lon = min(all_lons), max(all_lons)

            lat_range = max(max_lat - min_lat, 0.00001)
            lon_range = max(max_lon - min_lon, 0.00001)

            padding = 40

            def scale_coords(plat, plon):
                x = padding + ((plon - min_lon) / lon_range) * (w - 2 * padding)
                y = h - (padding + ((plat - min_lat) / lat_range) * (h - 2 * padding))
                return x, y

            # Draw the estimated flight path
            points = [scale_coords(p[0], p[1]) for p in self.flight_path]
            for i in range(len(points) - 1):
                self.map_canvas.create_line(
                    points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                    fill=self.t["accent_green"], width=2
                )

            # Draw individual anchor waypoints
            for ax, ay in points:
                self.map_canvas.create_rectangle(
                    ax - 3, ay - 3, ax + 3, ay + 3,
                    fill=self.t["bg_primary"], outline=self.t["accent_green"], width=1
                )

            cx, cy = points[-1]
        else:
            cx, cy = w / 2, h / 2

        # Draw current estimated drone position blip
        self.map_canvas.create_oval(
            cx - 6, cy - 6, cx + 6, cy + 6,
            fill=self.t["accent_blue"], outline=self.t["text_primary"], width=2
        )

        # Coordinate Overlay Box
        last_lat, last_lon = self.flight_path[-1]
        self.map_canvas.create_rectangle(10, 10, 220, 50, fill=self.t["bg_primary"], outline=self.t["bg_tertiary"])

        # If map_data is None, tracking is lost for this frame
        if map_data is None:
            self.map_canvas.create_text(
                18, 30, text=f"LAST LAT: {last_lat:.6f}°\nLAST LON: {last_lon:.6f}°",
                fill=self.t["text_muted"], font=("Courier", 9), anchor="w"
            )
            self.map_canvas.create_text(
                w / 2, h / 2 - 20, text="TRACKING LOST",
                fill=self.t["accent_red"], font=("Courier", 14, "bold")
            )
            self.map_canvas.create_text(
                w / 2, h / 2 + 10, text="Attempting to re-acquire visual lock...",
                fill=self.t["accent_red"], font=("Courier", 10)
            )
        else:
            self.map_canvas.create_text(
                18, 30, text=f"EST LAT: {last_lat:.6f}°\nEST LON: {last_lon:.6f}°",
                fill=self.t["accent_green"], font=("Courier", 9), anchor="w"
            )