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
        self.hud_image_item = None
        self.hud_text_item = None

        self.map_grid_drawn = False
        self.map_last_w = 0
        self.map_last_h = 0

        self.path_line_item = None
        self.drone_blip_item = None
        self.map_overlay_bg = None
        self.map_overlay_text = None
        self.map_status_text = None

        # Memory Object Pool for highly efficient waypoint rendering
        self.waypoint_pool: List[int] = []

        self._build_header()
        self._build_layout()

        self.controller.register_active_operator_view(self)

    def _build_header(self) -> None:
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
        conn_btn.pack(side="right", padx=5)

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
        reset_btn.pack(side="right", padx=5)

    def _on_reset_clicked(self) -> None:
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

    def _on_connect_clicked(self) -> None:
        target_dir = filedialog.askdirectory(
            title="Select Flight Mission Dataset Root Folder"
        )
        if target_dir:
            self.controller.start_operator_simulation(target_dir)

    def ui_update_telemetry(self, t_data: Dict[str, Any]) -> None:
        if not self.winfo_exists():
            self.controller.log_ui_event(
                "Telemetry frame dropped: Target view no longer exists in memory.",
                "DEBUG")
            return

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

        avg_conf = t_data.get("avg_conf", 0.0)
        lost = t_data.get("lost_frames", 0)
        skipped = t_data.get("skipped_frames", 0)

        self.avg_conf_lbl.configure(text=f"{avg_conf:.1f}%", fg=self.t["accent_green"])

        lost_color = self.t["accent_red"] if lost > 0 else self.t["text_primary"]
        self.lost_frames_lbl.configure(text=str(lost), fg=lost_color)

        skipped_color = self.t["accent_yellow"] if skipped > 0 else self.t["text_primary"]
        self.skipped_frames_lbl.configure(text=str(skipped), fg=skipped_color)

        self.ui_update_hud_canvas(t_data)

    def ui_update_status(self, is_active: bool, mode_text: str) -> None:
        if is_active:
            self.mode_subtitle.configure(text=f"Status: {mode_text}", fg=self.t["accent_blue"])
        else:
            self.mode_subtitle.configure(text="Status: Idle", fg=self.t["text_status"])

    def ui_update_hud_canvas(self, t_data: Dict[str, Any]) -> None:
        w, h = self.hud_canvas.winfo_width(), self.hud_canvas.winfo_height()

        if w < 10 or h < 10:
            return

        if t_data and t_data.get("frame_tk"):
            self._current_frame = t_data["frame_tk"]

            if self.hud_image_item is None:
                self.hud_image_item = self.hud_canvas.create_image(0, 0, anchor="nw", image=self._current_frame)
            else:
                self.hud_canvas.itemconfig(self.hud_image_item, image=self._current_frame, state="normal")

            if self.hud_text_item is not None:
                self.hud_canvas.itemconfig(self.hud_text_item, state="hidden")

        elif not t_data:
            if self.hud_image_item is not None:
                self.hud_canvas.itemconfig(self.hud_image_item, state="hidden")

            if self.hud_text_item is None:
                self.hud_text_item = self.hud_canvas.create_text(
                    w / 2, h / 2, text="AWAITING FRAMES", fill=self.t["text_muted"], font=FONT_MONO_HEADING
                )
            else:
                self.hud_canvas.itemconfig(self.hud_text_item, state="normal", text="AWAITING FRAMES")
            return

        if "alt" in t_data or "hdg" in t_data:
            alt_val = t_data.get('alt', 0.0)
            hdg_val = t_data.get('hdg', 0.0)
            overlay_str = f"ALT: {alt_val:.1f}m\nYAW: {hdg_val:.1f}°"

            if self.hud_text_item is None:
                self.hud_text_item = self.hud_canvas.create_text(
                    20, 20, text=overlay_str, fill=self.t["accent_green"], font=FONT_MONO_HEADING, anchor="nw"
                )
            else:
                self.hud_canvas.coords(self.hud_text_item, 20, 20)
                self.hud_canvas.itemconfig(
                    self.hud_text_item, state="normal", text=overlay_str, fill=self.t["accent_green"]
                )

    def ui_update_map_canvas(self, map_data: Optional[Dict[str, Any]]) -> None:
        w, h = self.map_canvas.winfo_width(), self.map_canvas.winfo_height()

        if w < 10 or h < 10:
            return

        if not self.map_grid_drawn or w != self.map_last_w or h != self.map_last_h:
            self.map_canvas.delete("grid_line")
            self._draw_map_grid(w, h)
            self.map_grid_drawn = True
            self.map_last_w = w
            self.map_last_h = h

        if not self.flight_path:
            # Hide the old continuous line and blip
            if self.path_line_item is not None:
                self.map_canvas.itemconfig(self.path_line_item, state="hidden")
            if self.drone_blip_item is not None:
                self.map_canvas.itemconfig(self.drone_blip_item, state="hidden")

            # Instantly hide all items currently in the waypoint pool
            for wp in self.waypoint_pool:
                self.map_canvas.itemconfig(wp, state="hidden")

            if self.map_status_text is None:
                self.map_status_text = self.map_canvas.create_text(
                    w / 2, h / 2, text="AWAITING INITIAL FIX", fill=self.t["text_muted"], font=FONT_MONO_NORMAL
                )
            else:
                self.map_canvas.coords(self.map_status_text, w / 2, h / 2)
                self.map_canvas.itemconfig(self.map_status_text, state="normal", text="AWAITING INITIAL FIX", fill=self.t["text_muted"])
            return

        if self.path_line_item is not None:
            self.map_canvas.itemconfig(self.path_line_item, state="normal")
        if self.drone_blip_item is not None:
            self.map_canvas.itemconfig(self.drone_blip_item, state="normal")

        if self.map_status_text is not None:
            self.map_canvas.itemconfig(self.map_status_text, state="hidden")

        cx, cy = self._draw_flight_path(w, h)
        self._draw_drone_blip(cx, cy)
        self._draw_overlay_box(w, h, map_data)

    def _draw_map_grid(self, w: int, h: int) -> None:
        for x in range(0, w, 50):
            self.map_canvas.create_line(x, 0, x, h, fill=self.t["border_color"], width=1, dash=(1, 5), tags="grid_line")
        for y in range(0, h, 50):
            self.map_canvas.create_line(0, y, w, y, fill=self.t["border_color"], width=1, dash=(1, 5), tags="grid_line")

    def _draw_flight_path(self, w: int, h: int) -> Tuple[float, float]:
        all_lats = [p[0] for p in self.flight_path]
        all_lons = [p[1] for p in self.flight_path]

        if len(all_lats) <= 1:
            if self.path_line_item is not None:
                self.map_canvas.itemconfig(self.path_line_item, state="hidden")
            for wp in self.waypoint_pool:
                self.map_canvas.itemconfig(wp, state="hidden")
            return w / 2, h / 2

        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)

        lat_range = max(max_lat - min_lat, 0.00001)
        lon_range = max(max_lon - min_lon, 0.00001)

        padding = 40
        flat_coords = []
        points = []

        for plat, plon in self.flight_path:
            x = padding + ((plon - min_lon) / lon_range) * (w - 2 * padding)
            y = h - (padding + ((plat - min_lat) / lat_range) * (h - 2 * padding))
            flat_coords.extend([x, y])
            points.append((x, y))

        if self.path_line_item is None:
            self.path_line_item = self.map_canvas.create_line(*flat_coords, fill=self.t["accent_green"], width=2)
        else:
            self.map_canvas.coords(self.path_line_item, *flat_coords)

        # ------------------------------------------------------------------------
        # Object Pool execution: Keeps 100% of waypoints without killing your CPU
        # ------------------------------------------------------------------------
        needed = len(points)
        current_pool_size = len(self.waypoint_pool)

        # 1. Expand the pool if the dataset is growing
        if needed > current_pool_size:
            for _ in range(needed - current_pool_size):
                new_item = self.map_canvas.create_rectangle(
                    0, 0, 0, 0, fill=self.t["bg_primary"], outline=self.t["accent_green"], width=1, tags="waypoint"
                )
                self.waypoint_pool.append(new_item)

        # 2. Mathematically update coordinates for active points
        for i in range(needed):
            ax, ay = points[i]
            self.map_canvas.coords(self.waypoint_pool[i], ax - 3, ay - 3, ax + 3, ay + 3)
            self.map_canvas.itemconfig(self.waypoint_pool[i], state="normal")

        # 3. Cleanly hide any overflow boxes (happens immediately after a reset)
        for i in range(needed, current_pool_size):
            self.map_canvas.itemconfig(self.waypoint_pool[i], state="hidden")

        return flat_coords[-2], flat_coords[-1]

    def _draw_drone_blip(self, cx: float, cy: float) -> None:
        if self.drone_blip_item is None:
            self.drone_blip_item = self.map_canvas.create_oval(
                cx - 6, cy - 6, cx + 6, cy + 6,
                fill=self.t["accent_blue"], outline=self.t["text_primary"], width=2
            )
        else:
            self.map_canvas.coords(self.drone_blip_item, cx - 6, cy - 6, cx + 6, cy + 6)
            self.map_canvas.tag_raise(self.drone_blip_item)

    def _draw_overlay_box(self, w: int, h: int, map_data: Optional[Dict[str, Any]]) -> None:
        last_lat, last_lon = self.flight_path[-1]

        if self.map_overlay_bg is None:
            self.map_overlay_bg = self.map_canvas.create_rectangle(
                10, 10, 220, 50, fill=self.t["bg_primary"], outline=self.t["bg_tertiary"]
            )
            self.map_overlay_text = self.map_canvas.create_text(
                18, 30, text="", font=FONT_MONO_NORMAL, anchor="w"
            )

        self.map_canvas.tag_raise(self.map_overlay_bg)
        self.map_canvas.tag_raise(self.map_overlay_text)

        if map_data is None:
            self.map_canvas.itemconfig(
                self.map_overlay_text,
                text=f"LAST LAT: {last_lat:.6f}°\nLAST LON: {last_lon:.6f}°",
                fill=self.t["text_muted"]
            )
        else:
            self.map_canvas.itemconfig(
                self.map_overlay_text,
                text=f"EST LAT: {last_lat:.6f}°\nEST LON: {last_lon:.6f}°",
                fill=self.t["accent_green"]
            )