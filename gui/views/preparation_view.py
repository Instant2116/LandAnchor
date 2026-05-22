import tkinter as tk
from tkinter import ttk, filedialog


class PreparationView(tk.Frame):
    def __init__(self, parent, controller):
        t = controller.config["theme"]
        super().__init__(parent, bg=t["bg_primary"])
        self.controller = controller
        self.t = t

        title_strip = tk.Frame(self, bg=t["bg_primary"])
        title_strip.pack(fill="x", padx=24, pady=20)
        tk.Label(
            title_strip,
            text="Preparation Screen",
            fg=t["text_primary"],
            bg=t["bg_primary"],
            font=("Arial", 22, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_strip,
            text="Recon Mode - Real-time ONNX XFeat Dataset Ingestion Pipeline",
            fg=t["text_status"],
            bg=t["bg_primary"],
            font=("Arial", 10),
            anchor="w",
        ).pack(fill="x")

        main_layout = tk.Frame(self, bg=t["bg_primary"])
        main_layout.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        main_layout.grid_columnconfigure(0, weight=2, uniform="preparation_layout")
        main_layout.grid_columnconfigure(1, weight=1, uniform="preparation_layout")
        main_layout.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(main_layout, bg=t["bg_primary"])
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right_col = tk.Frame(main_layout, bg=t["bg_primary"])
        right_col.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        source_bar = tk.Frame(left_col, bg=t["bg_primary"])
        source_bar.pack(fill="x", pady=(0, 16))
        tk.Label(
            source_bar,
            text="Processing Target Mode:",
            fg=t["text_secondary"],
            bg=t["bg_primary"],
            font=("Arial", 10),
        ).pack(side="left")

        self.source_combo = ttk.Combobox(
            source_bar,
            values=[
                "Dataset Directory Ingestion",
                "Live Camera Sensor Feed",
                "Pre-recorded Mission File",
            ],
            state="readonly",
            width=28,
        )
        self.source_combo.pack(side="left", padx=10)
        self.source_combo.current(0)

        self.proc_btn = tk.Button(
            source_bar,
            text="Initialize Ingestion Pipeline",
            bg=t["bg_tertiary"],
            fg=t["text_primary"],
            activebackground=t["border_color"],
            activeforeground=t["text_primary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
            font=("Arial", 10),
            cursor="hand2",
            command=self._on_ingestion_button_click,
        )
        self.proc_btn.pack(side="right")

        proc_card = tk.Frame(
            left_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        proc_card.pack(fill="both", expand=True)

        p_title_bar = tk.Frame(proc_card, bg=t["bg_secondary"])
        p_title_bar.pack(fill="x", padx=16, pady=12)
        tk.Label(
            p_title_bar,
            text="Data Frame Processing Context",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        self.feed_status_txt = tk.Label(
            p_title_bar,
            text="PIPELINE STANDBY",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 9, "bold"),
        )
        self.feed_status_txt.pack(side="right")

        self.video_canvas = tk.Canvas(
            proc_card, bg=t["bg_primary"], bd=0, highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.stat_overlay = tk.Frame(
            proc_card,
            bg=t["bg_primary"],
            bd=1,
            relief="solid",
            highlightbackground=t["bg_tertiary"],
        )
        self.stat_overlay.pack(fill="x", padx=16, pady=(0, 16), ipady=8)
        self.stat_overlay.grid_columnconfigure((0, 1, 2), weight=1)

        self.ov_feat = self._add_stat_box(
            self.stat_overlay, "Extracted Keypoints", "0", t["text_muted"], 0
        )
        self.ov_land = self._add_stat_box(
            self.stat_overlay, "Tracked Landmarks", "0", t["text_muted"], 1
        )
        self.ov_frame = self._add_stat_box(
            self.stat_overlay, "Processed Items", "0/0", t["text_muted"], 2
        )

        db_form_card = tk.Frame(
            right_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        db_form_card.pack(fill="x", pady=(0, 12), ipady=12)
        tk.Label(
            db_form_card,
            text="Active Database Ingestion Progress",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        p_labels = tk.Frame(db_form_card, bg=t["bg_secondary"])
        p_labels.pack(fill="x", padx=16)
        tk.Label(
            p_labels,
            text="Ingestion Queue Scale",
            fg=t["text_status"],
            bg=t["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")

        self.progress_percent_lbl = tk.Label(
            p_labels,
            text="0%",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 10, "bold"),
        )
        self.progress_percent_lbl.pack(side="right")

        self.db_progress = ttk.Progressbar(
            db_form_card, orient="horizontal", mode="determinate"
        )
        self.db_progress.pack(fill="x", padx=16, pady=10)

        self.side_desc = self._create_db_form_row(
            db_form_card, "Stored Descriptors Matrix Array", "0", t["accent_green"]
        )
        self.side_land = self._create_db_form_row(
            db_form_card, "Committed Relational Landmarks", "0", t["accent_blue"]
        )
        self.side_keyf = self._create_db_form_row(
            db_form_card, "Isolated Mapping Keyframes", "0", t["text_primary"]
        )

        current_db_card = tk.Frame(
            right_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        current_db_card.pack(fill="x", pady=12, ipady=12)
        tk.Label(
            current_db_card,
            text="Relational Database State Metrics",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        db_meta_box = tk.Frame(current_db_card, bg=t["bg_secondary"])
        db_meta_box.pack(fill="x", padx=16, pady=(0, 12))

        self.db_name_title = tk.Label(
            db_meta_box,
            text=self.controller.get_database_filename_node(),
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        self.db_name_title.pack(fill="x")
        tk.Label(
            db_meta_box,
            text="Hardware Isolated Real-time Matrix Channel",
            fg=t["text_muted"],
            bg=t["bg_secondary"],
            font=("Arial", 9),
            anchor="w",
        ).pack(fill="x")

        self.landmarks_lbl = self._create_telemetry_row(
            current_db_card, "Landmarks Table Rows Count", "0", t["text_primary"]
        )
        self.descriptors_lbl = self._create_telemetry_row(
            current_db_card, "Global Descriptors Allocation", "0", t["text_primary"]
        )
        self.features_lbl = self._create_telemetry_row(
            current_db_card, "Local Features Point Clusters", "0", t["text_primary"]
        )
        self.db_size_lbl = self._create_telemetry_row(
            current_db_card,
            "Physical Container Disk Footprint",
            "0.0 KB",
            t["text_primary"],
        )

        params_card = tk.Frame(
            right_col,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        params_card.pack(fill="x", pady=(12, 0), ipady=12)
        tk.Label(
            params_card,
            text="Linked Computing Core Configuration",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 12))

        cv_params = self.controller.get_cv_params()
        self._create_telemetry_row(
            params_card,
            "Active Model Target:",
            cv_params.get("featureDetector", "Unknown"),
            t["accent_blue"],
        )
        self._create_telemetry_row(
            params_card,
            "Matching Scalar Ratio Limit:",
            f"{cv_params.get('matchRatio', 0.0):.2f}",
            t["text_primary"],
        )
        self._create_telemetry_row(
            params_card,
            "XFeat Internal Tracking Array Bounds:",
            str(cv_params.get("xfeatMaxFeatures", 0)),
            t["text_primary"],
        )
        self.flight_path = []
        self.anchor_points = []
        self.video_canvas.bind("<Configure>", lambda e: self.update_canvas_frame_view())

        self.refresh_ui_metrics_display()
        self.flight_path = []
        self.anchor_points = []
        self.video_canvas.bind("<Configure>", lambda e: self.update_canvas_frame_view())

        # Register this view with the controller so progress state survives tab switching
        if hasattr(self.controller, "register_preparation_view"):
            self.controller.register_preparation_view(self)

        self.refresh_ui_metrics_display()

    def refresh_ui_metrics_display(self) -> None:
        metrics = self.controller.get_active_database_metrics_report()
        self.landmarks_lbl.configure(text=metrics.get("landmarks_count", "0"))
        self.descriptors_lbl.configure(
            text=metrics.get("global_descriptors_count", "0")
        )
        self.features_lbl.configure(text=metrics.get("local_features_count", "0"))
        self.db_size_lbl.configure(text=metrics.get("disk_size_string", "0.0 KB"))

    def _on_ingestion_button_click(self) -> None:
        if self.source_combo.get() == "Dataset Directory Ingestion":
            target_dir = filedialog.askdirectory(
                title="Select Flight Mission Dataset Root Folder"
            )
            if target_dir:
                self.controller.start_dataset_processing_pipeline(
                    target_dir, view_callback=self
                )

    def ui_signal_process_start(self) -> None:
        self.proc_btn.configure(state="disabled")
        self.source_combo.configure(state="disabled")
        self.feed_status_txt.configure(
            text="PROCESSING LOCAL INFERENCE RUNTIME", fg=self.t["accent_green"]
        )

    def ui_signal_process_update(self, data: dict) -> None:
        self.progress_percent_lbl.configure(
            text=f"{int(data.get('progress_ratio', 0))}%"
        )
        self.db_progress["value"] = data.get("progress_ratio", 0)
        self.ov_feat.configure(
            text=str(data.get("total_features", 0)), fg=self.t["accent_green"]
        )
        self.ov_land.configure(text=str(data.get("total_landmarks", 0)), fg=self.t["accent_blue"])
        self.ov_frame.configure(
            text=f"{data.get('current_index', 0)}/{data.get('total_items', 0)}",
            fg=self.t["text_primary"],
        )
        self.side_desc.configure(text=str(data.get("total_features", 0)))
        self.side_land.configure(text=str(data.get("total_landmarks", 0)))
        self.side_keyf.configure(text=str(data.get("total_keyframes", 0)))
        self.update_canvas_frame_view(running=True)

        # DYNAMIC UI UPDATE: Poll the database metrics during processing
        # We use modulo 5 to avoid spamming the SQLite database with queries on every single frame
        if data.get("current_index", 0) % 5 == 0:
            self.refresh_ui_metrics_display()

    def ui_signal_process_complete(self) -> None:
        self.proc_btn.configure(state="normal")
        self.source_combo.configure(state="readonly")
        self.feed_status_txt.configure(
            text="INGESTION PROCESS COMPLETE", fg=self.t["accent_blue"]
        )
        self.refresh_ui_metrics_display()
        self.update_canvas_frame_view(running=False)

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

    def _add_stat_box(self, parent, label, init_val, color, col) -> tk.Label:
        box = tk.Frame(parent, bg=self.t["bg_primary"])
        box.grid(row=0, column=col, sticky="nsew")
        tk.Label(
            box, text=label, fg=self.t["text_muted"], bg=self.t["bg_primary"], font=("Arial", 9)
        ).pack()
        v_lbl = tk.Label(
            box, text=init_val, fg=color, bg=self.t["bg_primary"], font=("Arial", 11, "bold")
        )
        v_lbl.pack()
        return v_lbl

    def _create_db_form_row(self, parent, label, val, color) -> tk.Label:
        row = tk.Frame(parent, bg=self.t["bg_secondary"])
        row.pack(fill="x", padx=16, pady=4)
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
            font=("Courier", 10, "bold"),
        )
        v_lbl.pack(side="right")
        return v_lbl

    def update_canvas_frame_view(self, running=False) -> None:
        self.video_canvas.delete("all")
        w, h = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
        if w < 10 or h < 10:
            return

        if not running:
            self.video_canvas.create_text(
                w / 2,
                h / 2,
                text="Hardware pipeline offline. Awaiting dataset injection input channel.",
                fill=self.t["text_muted"],
                font=("Courier", 11),
            )
        else:
            bx, by = int(w * 0.35), int(h * 0.35)
            self.video_canvas.create_rectangle(
                bx, by, bx + 130, by + 95, outline=self.t["accent_green"], width=2
            )
            self.video_canvas.create_text(
                bx + 8,
                by + 15,
                text="ONNX Inference Core Active",
                fill=self.t["accent_green"],
                font=("Courier", 8, "bold"),
                anchor="w",
            )