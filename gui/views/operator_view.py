import tkinter as tk
from tkinter import ttk


class OperatorDashboardView(tk.Frame):
    def __init__(self, parent, controller):
        t = controller.config["theme"]
        super().__init__(parent, bg=t["bg_primary"])
        self.controller = controller
        self.t = t

        title_strip = tk.Frame(self, bg=t["bg_primary"])
        title_strip.pack(fill="x", padx=24, pady=20)
        tk.Label(
            title_strip,
            text="Operator Dashboard",
            fg=t["text_primary"],
            bg=t["bg_primary"],
            font=("Arial", 22, "bold"),
            anchor="w",
        ).pack(fill="x")

        self.mode_subtitle = tk.Label(
            title_strip,
            text="Navigation Mode - Awaiting Hardware Link Connection",
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
            text="Live Video Stream",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        rec_indicator = tk.Frame(v_title_bar, bg=t["bg_secondary"])
        rec_indicator.pack(side="right")
        self.stream_dot = tk.Label(
            rec_indicator,
            text="●",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 10),
        )
        self.stream_dot.pack(side="left")
        self.stream_status = tk.Label(
            rec_indicator,
            text="STREAM IDLE",
            fg=t["text_status"],
            bg=t["bg_secondary"],
            font=("Arial", 9, "bold"),
        )
        self.stream_status.pack(side="left", padx=(4, 0))

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
            text="Map View",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=12)

        self.map_canvas = tk.Canvas(map_card, bg=t["bg_primary"], bd=0, highlightthickness=0)
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
            text="CV Algorithm Confidence",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        c_labels = tk.Frame(conf_card, bg=t["bg_secondary"])
        c_labels.pack(fill="x", padx=16)
        tk.Label(
            c_labels,
            text="Current",
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
        self.conf_dot = tk.Label(
            self.conf_status_frame,
            text="●",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 9),
        )
        self.conf_dot.pack(side="left")
        self.conf_status_txt = tk.Label(
            self.conf_status_frame,
            text="No active sensor feed",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 9),
        )
        self.conf_status_txt.pack(side="left", padx=6)

        sys_card = tk.Frame(
            right_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        sys_card.pack(fill="x", pady=12, ipady=12)
        tk.Label(
            sys_card,
            text="System Status",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        self.bat_lbl = self._create_telemetry_row(
            sys_card, "Battery", "N/A", t["text_status"]
        )
        self.time_lbl = self._create_telemetry_row(
            sys_card, "Flight Time", "00:00:00", t["text_status"]
        )
        self.wind_lbl = self._create_telemetry_row(
            sys_card, "Wind Speed", "N/A", t["text_status"]
        )

        mode_card = tk.Frame(
            right_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        mode_card.pack(fill="x", pady=(12, 0), ipady=14)
        tk.Label(
            mode_card,
            text="Flight Mode",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 8))

        m_banner = tk.Frame(mode_card, bg=t["bg_secondary"])
        m_banner.pack(fill="x", padx=16)
        self.mode_dot = tk.Label(
            m_banner,
            text="●",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 10),
        )
        self.mode_dot.pack(side="left")
        self.mode_txt = tk.Label(
            m_banner,
            text="LINK DISCONNECTED",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 10, "bold"),
        )
        self.mode_txt.pack(side="left", padx=8)

        self.hud_canvas.bind("<Configure>", lambda e: self.ui_update_hud_canvas({}))
        self.map_canvas.bind("<Configure>", lambda e: self.ui_update_map_canvas({}))

        self.controller.register_active_operator_view(self)
        conn_btn = tk.Button(
            title_strip, text="Load Dataset Feed", bg=t["bg_tertiary"], fg=t["text_primary"],
            bd=1, relief="solid", font=("Arial", 9), cursor="hand2",
            command=self._on_connect_clicked
        )
        conn_btn.pack(side="right", padx=16)

    def _on_connect_clicked(self) -> None:
        from tkinter import filedialog
        target_dir = filedialog.askdirectory(title="Select Flight Mission Dataset Root Folder")
        if target_dir:
            self.controller.start_operator_simulation(target_dir)

    def _create_telemetry_row(self, parent, label, val, color) -> tk.Label:
        row = tk.Frame(parent, bg=self.t["bg_secondary"])
        row.pack(fill="x", padx=16, pady=6)
        tk.Label(
            row,
            text=label,
            fg=self.t["text_status"],
            bg=self.t["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")
        v_lbl = tk.Label(
            row,
            text=val,
            fg=color,
            bg=self.t["bg_secondary"],
            font=("Arial", 10, "bold"),
        )
        v_lbl.pack(side="right")
        return v_lbl

    def ui_update_telemetry(self, t_data: dict) -> None:
        self.bat_lbl.configure(
            text=t_data.get("battery", "N/A"), fg=self.t["accent_green"]
        )
        self.time_lbl.configure(
            text=t_data.get("time", "00:00:00"), fg=self.t["text_primary"]
        )
        self.wind_lbl.configure(
            text=t_data.get("wind", "N/A"), fg=self.t["text_primary"]
        )

        conf = float(t_data.get("confidence", 0.0))
        self.conf_percent_lbl.configure(text=f"{int(conf)}%")
        self.conf_progress["value"] = conf

        if conf >= 80:
            self.conf_dot.configure(fg=self.t["accent_green"])
            self.conf_status_txt.configure(
                text="Excellent tracking", fg=self.t["accent_green"]
            )
        elif conf >= 60:
            self.conf_dot.configure(fg=self.t["accent_yellow"])
            self.conf_status_txt.configure(
                text="Acceptable tracking", fg=self.t["accent_yellow"]
            )
        elif conf > 0:
            self.conf_dot.configure(fg=self.t["accent_red"])
            self.conf_status_txt.configure(
                text="Poor tracking", fg=self.t["accent_red"]
            )

        self.ui_update_hud_canvas(t_data)

    def ui_update_status(self, is_active: bool, mode_text: str) -> None:
        if is_active:
            self.mode_subtitle.configure(text=f"Navigation Mode - {mode_text}")
            self.stream_dot.configure(fg=self.t["accent_red"])
            self.stream_status.configure(text="LIVE STREAM ACTIVE")
            self.mode_dot.configure(fg=self.t["accent_blue"])
            self.mode_txt.configure(text=mode_text.upper(), fg=self.t["accent_blue"])
        else:
            self.mode_subtitle.configure(
                text="Navigation Mode - Awaiting Hardware Link Connection"
            )
            self.stream_dot.configure(fg=self.t["text_muted"])
            self.stream_status.configure(text="STREAM IDLE")
            self.mode_dot.configure(fg=self.t["text_muted"])
            self.mode_txt.configure(text="LINK DISCONNECTED", fg=self.t["text_muted"])

    def ui_update_hud_canvas(self, t_data: dict) -> None:
        self.hud_canvas.delete("all")
        w, h = self.hud_canvas.winfo_width(), self.hud_canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if t_data and t_data.get("frame_tk"):
            self._current_frame = t_data["frame_tk"]
            self.hud_canvas.create_image(0, 0, anchor="nw", image=self._current_frame)
        elif not t_data:
            self.hud_canvas.create_text(w/2, h/2, text="NO SIGNAL", fill=self.t["text_muted"], font=("Courier", 10, "bold"))
            return

        for idx in range(1, 8):
            dx = w * (idx / 8)
            self.hud_canvas.create_line(dx, 0, dx, h, fill=self.t["border_color"], width=1, dash=(2, 4))
        for idx in range(1, 6):
            dy = h * (idx / 6)
            self.hud_canvas.create_line(0, dy, w, dy, fill=self.t["border_color"], width=1, dash=(2, 4))

        self.hud_canvas.create_line(0, h/2, w, h/2, fill=self.t["accent_blue"], width=1)
        cx, cy = w / 2, h / 2
        self.hud_canvas.create_oval(cx-24, cy-24, cx+24, cy+24, outline=self.t["accent_green"], width=2)
        self.hud_canvas.create_line(cx-36, cy, cx+36, cy, fill=self.t["accent_green"], width=1)
        self.hud_canvas.create_line(cx, cy-36, cx, cy+36, fill=self.t["accent_green"], width=1)

        self.hud_canvas.create_text(30, 30, text=f"ALT: {t_data.get('alt', 0.0):.1f}m", fill=self.t["accent_green"], font=("Courier", 11, "bold"), anchor="w")
        self.hud_canvas.create_text(30, 50, text=f"SPD: {t_data.get('spd', 0.0):.1f}m/s", fill=self.t["accent_green"], font=("Courier", 11, "bold"), anchor="w")
        self.hud_canvas.create_text(30, 70, text=f"HDG: {t_data.get('hdg', 0.0):.0f}°", fill=self.t["accent_green"], font=("Courier", 11, "bold"), anchor="w")

    def ui_update_map_canvas(self, map_data: dict) -> None:
        self.map_canvas.delete("all")
        w, h = self.map_canvas.winfo_width(), self.map_canvas.winfo_height()
        if w < 10 or h < 10:
            return

        for x in range(0, w, 50):
            self.map_canvas.create_line(
                x, 0, x, h, fill=self.t["border_color"], width=1, dash=(1, 5)
            )
        for y in range(0, h, 50):
            self.map_canvas.create_line(
                0, y, w, y, fill=self.t["border_color"], width=1, dash=(1, 5)
            )

        if not map_data:
            self.map_canvas.create_text(
                w / 2,
                h / 2,
                text="AWAITING COORDINATE STREAM",
                fill=self.t["text_muted"],
                font=("Courier", 10),
            )
            return

        cx, cy = w / 2, h / 2
        self.map_canvas.create_oval(
            cx - 8,
            cy - 8,
            cx + 8,
            cy + 8,
            fill=self.t["accent_red"],
            outline=self.t["text_primary"],
            width=1,
        )
        self.map_canvas.create_text(
            20,
            20,
            text=f"LAT: {map_data.get('lat', 0.0):.4f}°\nLON: {map_data.get('lon', 0.0):.4f}°",
            fill=self.t["text_status"],
            font=("Courier", 10),
            anchor="w",
        )