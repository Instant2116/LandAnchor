"""
Operator view for LandAnchor.
Provides the user interface for dataset processing, telemetry, and map tracking.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Dict, Any, Optional, List, Tuple

FONT_TITLE = ("Arial", 22, "bold")
FONT_HEADING = ("Arial", 11, "bold")
FONT_NORMAL = ("Arial", 10)
FONT_SMALL = ("Arial", 9)

FONT_MONO_NORMAL = ("Courier", 9)
FONT_MONO_HEADING = ("Courier", 10, "bold")
FONT_MONO_LARGE = ("Courier", 14, "bold")


class OperatorView(tk.Frame):
    def __init__(self, parent: tk.Widget, controller: Any) -> None:
        self.controller = controller
        self.t: Dict[str, str] = controller.app_config["theme"]

        super().__init__(parent, bg=self.t["bg_primary"])

        self.flight_path: List[Tuple[float, float]] = []
        self._current_frame: Optional[tk.PhotoImage] = None

        self._build_header()
        self._build_layout()

        self._bind_events()
        self.controller.register_active_operator_view(self)

    def _build_header(self) -> None:
        """Constructs the static title bar context."""
        title_strip = tk.Frame(self, bg=self.t["bg_primary"])
        title_strip.pack(fill="x", padx=24, pady=20)

        tk.Label(
            title_strip,
            text="Dataset Processing Dashboard",
            fg=self.t["text_primary"],
            bg=self.t["bg_primary"],
            font=FONT_TITLE,
            anchor="w",
        ).pack(fill="x")

        self.mode_subtitle = tk.Label(
            title_strip,
            text="Awaiting Dataset Feed",
            fg=self.t["text_status"],
            bg=self.t["bg_primary"],
            font=FONT_NORMAL,
            anchor="w",
        )
        self.mode_subtitle.pack(fill="x")

        # Create Load Button
        conn_btn = tk.Button(
            title_strip,
            text="Load Dataset",
            bg=self.t["bg_tertiary"],
            fg=self.t["text_primary"],
            bd=1,
            relief="solid",
            font=FONT_SMALL,
            cursor="hand2",
            command=self._on_connect_clicked,
        )
        # Pack to the right FIRST (it will be the rightmost button)
        conn_btn.pack(side="right", padx=5)

        # Create Reset Button
        reset_btn = tk.Button(
            title_strip,
            text="Clear Path & Stats",
            bg=self.t["bg_error"],
            fg=self.t["text_primary"],
            bd=0,
            padx=10,
            font=FONT_SMALL,
            cursor="hand2",
            command=self._on_reset_clicked,
        )
        # Pack to the right SECOND (it will be to the left of the Load button)
        reset_btn.pack(side="right", padx=5)

    def _on_reset_clicked(self) -> None:
        """Triggers a session reset through the controller."""
        # Optional: Add a confirmation dialog if desired
        # if tk.messagebox.askyesno("Reset Session", "Are you sure you want to clear all path data and statistics?"):
        self.controller.operator_manager.reset_session()

    def _build_layout(self) -> None:
        main_layout = tk.Frame(self, bg=self.t["bg_primary"])
        main_layout.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        main_layout.grid_columnconfigure(0, weight=2)
        main_layout.grid_columnconfigure(1, weight=1)
        main_layout.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(main_layout, bg=self.t["bg_primary"])
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        right_col = tk.Frame(main_layout, bg=self.t["bg_primary"])
        right_col.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self._build_video_card(left_col)
        self._build_map_card(left_col)
        self._build_confidence_card(right_col)

    def _build_video_card(self, parent: tk.Frame) -> None:
        video_card = tk.Frame(
            parent,
            bg=self.t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.t["border_color"],
        )
        video_card.pack(fill="both", expand=True, pady=(0, 12))

        v_title_bar = tk.Frame(video_card, bg=self.t["bg_secondary"])
        v_title_bar.pack(fill="x", padx=16, pady=12)
        tk.Label(
            v_title_bar,
            text="Dataset Frame Feed",
            fg=self.t["text_primary"],
            bg=self.t["bg_secondary"],
            font=FONT_HEADING,
        ).pack(side="left")

        self.hud_canvas = tk.Canvas(
            video_card, bg=self.t["bg_primary"], bd=0, highlightthickness=0
        )
        self.hud_canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def _build_map_card(self, parent: tk.Frame) -> None:
        map_card = tk.Frame(
            parent,
            bg=self.t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.t["border_color"],
        )
        map_card.pack(fill="both", expand=True, pady=(12, 0))

        tk.Label(
            map_card,
            text="Coordinate Tracking Map",
            fg=self.t["text_primary"],
            bg=self.t["bg_secondary"],
            font=FONT_HEADING,
        ).pack(anchor="w", padx=16, pady=12)

        self.map_canvas = tk.Canvas(
            map_card, bg=self.t["bg_primary"], bd=0, highlightthickness=0
        )
        self.map_canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def _build_confidence_card(self, parent: tk.Frame) -> None:
        conf_card = tk.Frame(
            parent,
            bg=self.t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.t["border_color"],
        )
        conf_card.pack(fill="x", pady=(0, 12), ipady=10)

        tk.Label(
            conf_card,
            text="Localization Confidence",
            fg=self.t["text_primary"],
            bg=self.t["bg_secondary"],
            font=FONT_HEADING,
        ).pack(anchor="w", padx=16, pady=(12, 16))

        c_labels = tk.Frame(conf_card, bg=self.t["bg_secondary"])
        c_labels.pack(fill="x", padx=16)

        tk.Label(
            c_labels,
            text="Inlier Ratio",
            fg=self.t["text_status"],
            bg=self.t["bg_secondary"],
            font=FONT_NORMAL,
        ).pack(side="left")

        self.conf_percent_lbl = tk.Label(
            c_labels,
            text="0%",
            fg=self.t["text_muted"],
            bg=self.t["bg_secondary"],
            font=FONT_HEADING,
        )
        self.conf_percent_lbl.pack(side="right")

        self.conf_progress = ttk.Progressbar(
            conf_card, orient="horizontal", mode="determinate"
        )
        self.conf_progress.pack(fill="x", padx=16, pady=10)
        self.conf_progress["value"] = 0.0

        self.conf_status_frame = tk.Frame(conf_card, bg=self.t["bg_secondary"])
        self.conf_status_frame.pack(fill="x", padx=16)
        self.conf_status_txt = tk.Label(
            self.conf_status_frame,
            text="Awaiting data",
            fg=self.t["text_muted"],
            bg=self.t["bg_secondary"],
            font=FONT_SMALL,
        )
        self.conf_status_txt.pack(side="left")

        # --- NEW METRICS BLOCK ---
        metrics_frame = tk.Frame(conf_card, bg=self.t["bg_secondary"])
        metrics_frame.pack(fill="x", padx=16, pady=(16, 0))

        self.avg_conf_lbl = self._create_stat_row(metrics_frame, "Avg Match Confidence:", "0.0%")
        self.lost_frames_lbl = self._create_stat_row(metrics_frame, "Tracking Lost (Frames):", "0")
        self.skipped_frames_lbl = self._create_stat_row(metrics_frame, "Read Errors (Skipped):", "0")

    def _create_stat_row(self, parent: tk.Widget, label_text: str, default_val: str) -> tk.Label:
        row = tk.Frame(parent, bg=self.t["bg_secondary"])
        row.pack(fill="x", pady=2)
        tk.Label(
            row, text=label_text, fg=self.t["text_status"], bg=self.t["bg_secondary"], font=FONT_SMALL
        ).pack(side="left")

        val_lbl = tk.Label(
            row, text=default_val, fg=self.t["text_primary"], bg=self.t["bg_secondary"], font=FONT_MONO_HEADING
        )
        val_lbl.pack(side="right")
        return val_lbl

    def _bind_events(self) -> None:
        self.hud_canvas.bind("<Configure>", lambda e: self.ui_update_hud_canvas({}))
        self.map_canvas.bind("<Configure>", lambda e: self.ui_update_map_canvas({}))

    def _on_connect_clicked(self) -> None:
        target_dir = filedialog.askdirectory(
            title="Select Flight Mission Dataset Root Folder"
        )
        if target_dir:
            self.controller.start_operator_simulation(target_dir)

    def ui_update_telemetry(self, t_data: Dict[str, Any]) -> None:
        # 1. Update Core Confidence
        conf = float(t_data.get("confidence", 0.0))
        self.conf_percent_lbl.configure(text=f"{int(conf)}%")
        self.conf_progress["value"] = conf

        if conf >= 80:
            status_text = "High Confidence Match"
            status_color = self.t["accent_green"]
        elif conf >= 50:
            status_text = "Moderate Confidence Match"
            status_color = self.t["accent_yellow"]
        elif conf > 0:
            status_text = "Low Confidence Match"
            status_color = self.t["accent_red"]
        else:
            status_text = "No Match Found"
            status_color = self.t["text_muted"]

        self.conf_status_txt.configure(text=status_text, fg=status_color)

        # 2. Update New Tracking Metrics
        avg_conf = t_data.get("avg_conf", 0.0)
        lost = t_data.get("lost_frames", 0)
        skipped = t_data.get("skipped_frames", 0)

        self.avg_conf_lbl.configure(text=f"{avg_conf:.1f}%", fg=self.t["accent_green"])

        # Color code failures dynamically
        lost_color = self.t["accent_red"] if lost > 0 else self.t["text_primary"]
        self.lost_frames_lbl.configure(text=str(lost), fg=lost_color)

        skipped_color = self.t["accent_yellow"] if skipped > 0 else self.t["text_primary"]
        self.skipped_frames_lbl.configure(text=str(skipped), fg=skipped_color)

        # 3. Update Visual Canvas
        self.ui_update_hud_canvas(t_data)

    def ui_update_status(self, is_active: bool, mode_text: str) -> None:
        if is_active:
            self.mode_subtitle.configure(text=f"Status: {mode_text}", fg=self.t["accent_blue"])
        else:
            self.mode_subtitle.configure(text="Status: Idle", fg=self.t["text_status"])

    def ui_update_hud_canvas(self, t_data: Dict[str, Any]) -> None:
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
                font=FONT_MONO_HEADING,
            )
            return

        if "alt" in t_data or "hdg" in t_data:
            alt_val = t_data.get('alt', 0.0)
            hdg_val = t_data.get('hdg', 0.0)
            self.hud_canvas.create_text(
                20,
                20,
                text=f"ALT: {alt_val:.1f}m\nYAW: {hdg_val:.1f}°",
                fill=self.t["accent_green"],
                font=FONT_MONO_HEADING,
                anchor="nw",
            )

    def ui_update_map_canvas(self, map_data: Optional[Dict[str, Any]]) -> None:
        self.map_canvas.delete("all")
        w, h = self.map_canvas.winfo_width(), self.map_canvas.winfo_height()

        if w < 10 or h < 10:
            return

        self._draw_map_grid(w, h)

        if map_data and "lat" in map_data:
            lat, lon = float(map_data["lat"]), float(map_data["lon"])
            self.flight_path.append((lat, lon))

            if len(self.flight_path) > 10000:
                self.flight_path.pop(0)

        if not self.flight_path:
            self.map_canvas.create_text(
                w / 2, h / 2, text="AWAITING INITIAL FIX", fill=self.t["text_muted"], font=FONT_MONO_NORMAL
            )
            return

        cx, cy = self._draw_flight_path(w, h)
        self._draw_drone_blip(cx, cy)
        self._draw_overlay_box(w, h, map_data)

    def _draw_map_grid(self, w: int, h: int) -> None:
        for x in range(0, w, 50):
            self.map_canvas.create_line(x, 0, x, h, fill=self.t["border_color"], width=1, dash=(1, 5))
        for y in range(0, h, 50):
            self.map_canvas.create_line(0, y, w, y, fill=self.t["border_color"], width=1, dash=(1, 5))

    def _draw_flight_path(self, w: int, h: int) -> Tuple[float, float]:
        all_lats = [p[0] for p in self.flight_path]
        all_lons = [p[1] for p in self.flight_path]

        if len(all_lats) <= 1:
            return w / 2, h / 2

        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)

        lat_range = max(max_lat - min_lat, 0.00001)
        lon_range = max(max_lon - min_lon, 0.00001)

        padding = 40

        def scale_coords(plat: float, plon: float) -> Tuple[float, float]:
            x = padding + ((plon - min_lon) / lon_range) * (w - 2 * padding)
            y = h - (padding + ((plat - min_lat) / lat_range) * (h - 2 * padding))
            return x, y

        points = [scale_coords(p[0], p[1]) for p in self.flight_path]

        for i in range(len(points) - 1):
            self.map_canvas.create_line(
                points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                fill=self.t["accent_green"], width=2
            )

        for ax, ay in points:
            self.map_canvas.create_rectangle(
                ax - 3, ay - 3, ax + 3, ay + 3,
                fill=self.t["bg_primary"], outline=self.t["accent_green"], width=1
            )

        return points[-1]

    def _draw_drone_blip(self, cx: float, cy: float) -> None:
        self.map_canvas.create_oval(
            cx - 6, cy - 6, cx + 6, cy + 6,
            fill=self.t["accent_blue"], outline=self.t["text_primary"], width=2
        )

    def _draw_overlay_box(self, w: int, h: int, map_data: Optional[Dict[str, Any]]) -> None:
        last_lat, last_lon = self.flight_path[-1]

        self.map_canvas.create_rectangle(10, 10, 220, 50, fill=self.t["bg_primary"], outline=self.t["bg_tertiary"])

        if map_data is None:
            self.map_canvas.create_text(
                18, 30, text=f"LAST LAT: {last_lat:.6f}°\nLAST LON: {last_lon:.6f}°",
                fill=self.t["text_muted"], font=FONT_MONO_NORMAL, anchor="w"
            )
            self.map_canvas.create_text(
                w / 2, h / 2 - 20, text="TRACKING LOST",
                fill=self.t["accent_red"], font=FONT_MONO_LARGE
            )
            self.map_canvas.create_text(
                w / 2, h / 2 + 10, text="Attempting to re-acquire visual lock...",
                fill=self.t["accent_red"], font=FONT_MONO_HEADING
            )
        else:
            self.map_canvas.create_text(
                18, 30, text=f"EST LAT: {last_lat:.6f}°\nEST LON: {last_lon:.6f}°",
                fill=self.t["accent_green"], font=FONT_MONO_NORMAL, anchor="w"
            )