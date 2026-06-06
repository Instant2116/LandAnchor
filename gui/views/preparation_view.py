import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any, Dict, Optional
from PIL import ImageTk, Image


class PreparationView(tk.Frame):
    """
    Presentation layer for the dataset ingestion and database preparation phase.

    This view manages the interface for triggering the ONNX XFeat extraction pipeline,
    rendering real-time extraction metrics, and monitoring the storage footprint
    of resulting descriptors and landmarks in the local SQLite database.
    """

    def __init__(self, parent: tk.Widget, controller: Any) -> None:
        """
        Initializes the preparation screen layout and hooks into the controller.

        Args:
            parent: The parent Tkinter widget containing this frame.
            controller: The application controller handling data ingestion logic.
        """
        self.controller = controller
        self.theme: Dict[str, str] = controller.config["theme"]

        super().__init__(parent, bg=self.theme["bg_primary"])

        # Track trajectory elements for debug visualizer (if logic layer implements it)
        self.flight_path: list = []
        self.anchor_points: list = []

        # State cache for rendering live frames
        self._current_frame: Optional[ImageTk.PhotoImage] = None
        self._last_data: Optional[Dict[str, Any]] = None

        self._build_header()
        self._build_main_layout()

        # Register this view with the controller so progress state survives tab switching
        if hasattr(self.controller, "register_preparation_view"):
            self.controller.register_preparation_view(self)

        self.refresh_ui_metrics_display()

    def _build_header(self) -> None:
        """Constructs the static title bar context."""
        title_strip = tk.Frame(self, bg=self.theme["bg_primary"])
        title_strip.pack(fill="x", padx=24, pady=20)

        tk.Label(
            title_strip,
            text="Preparation Screen",
            fg=self.theme["text_primary"],
            bg=self.theme["bg_primary"],
            font=("Arial", 22, "bold"),
            anchor="w",
        ).pack(fill="x")

        tk.Label(
            title_strip,
            text="Provide data for mapping",
            fg=self.theme["text_status"],
            bg=self.theme["bg_primary"],
            font=("Arial", 10),
            anchor="w",
        ).pack(fill="x")

    def _build_main_layout(self) -> None:
        """Initializes the symmetrical layout columns and delegates pane construction."""
        main_layout = tk.Frame(self, bg=self.theme["bg_primary"])
        main_layout.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        main_layout.grid_columnconfigure(0, weight=2, uniform="preparation_layout")
        main_layout.grid_columnconfigure(1, weight=1, uniform="preparation_layout")
        main_layout.grid_rowconfigure(0, weight=1)

        left_col = tk.Frame(main_layout, bg=self.theme["bg_primary"])
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        right_col = tk.Frame(main_layout, bg=self.theme["bg_primary"])
        right_col.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self._build_control_and_monitor_pane(left_col)
        self._build_database_telemetry_pane(right_col)

    def _build_control_and_monitor_pane(self, parent: tk.Widget) -> None:
        """Constructs the data source selector and the dynamic visual feed canvas."""
        # Source Selection Bar
        source_bar = tk.Frame(parent, bg=self.theme["bg_primary"])
        source_bar.pack(fill="x", pady=(0, 16))

        tk.Label(
            source_bar,
            text="Processing Target Mode:",
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_primary"],
            font=("Arial", 10),
        ).pack(side="left")

        self.source_combo = ttk.Combobox(
            source_bar,
            values=[
                "Dataset",
                "Option 2",
                "Option 3",
            ],
            state="readonly",
            width=28,
        )
        self.source_combo.pack(side="left", padx=10)
        self.source_combo.current(0)

        self.proc_btn = tk.Button(
            source_bar,
            text="Load data",
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["border_color"],
            activeforeground=self.theme["text_primary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["border_color"],
            font=("Arial", 10),
            cursor="hand2",
            command=self._on_ingestion_button_click,
        )
        self.proc_btn.pack(side="right")

        # Processing Monitoring Card
        proc_card = tk.Frame(
            parent,
            bg=self.theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["border_color"],
        )
        proc_card.pack(fill="both", expand=True)

        p_title_bar = tk.Frame(proc_card, bg=self.theme["bg_secondary"])
        p_title_bar.pack(fill="x", padx=16, pady=12)

        tk.Label(
            p_title_bar,
            text="Data processing context",
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        self.feed_status_txt = tk.Label(
            p_title_bar,
            text="PIPELINE STANDBY",
            fg=self.theme["text_muted"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 9, "bold"),
        )
        self.feed_status_txt.pack(side="right")

        self.video_canvas = tk.Canvas(
            proc_card, bg=self.theme["bg_primary"], bd=0, highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.video_canvas.bind(
            "<Configure>",
            lambda e: self.update_canvas_frame_view(
                running=(self.feed_status_txt.cget("text") != "PIPELINE STANDBY"),
                data=self._last_data,
            ),
        )

        # Live Metric Overlay Bottom Bar
        self.stat_overlay = tk.Frame(
            proc_card,
            bg=self.theme["bg_primary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["bg_tertiary"],
        )
        self.stat_overlay.pack(fill="x", padx=16, pady=(0, 16), ipady=8)
        self.stat_overlay.grid_columnconfigure((0, 1, 2), weight=1)

        self.ov_feat = self._add_stat_box(
            self.stat_overlay, "Extracted Keypoints", "0", self.theme["text_muted"], 0
        )
        self.ov_land = self._add_stat_box(
            self.stat_overlay, "Tracked Landmarks", "0", self.theme["text_muted"], 1
        )
        self.ov_frame = self._add_stat_box(
            self.stat_overlay, "Processed Items", "0/0", self.theme["text_muted"], 2
        )

    def _build_database_telemetry_pane(self, parent: tk.Widget) -> None:
        """Constructs tracking cards for SQLite database footprint and pipeline configuration."""
        # Active Ingestion Progress Card
        db_form_card = tk.Frame(
            parent,
            bg=self.theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["border_color"],
        )
        db_form_card.pack(fill="x", pady=(0, 12), ipady=12)

        tk.Label(
            db_form_card,
            text="Data processing progress",
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        p_labels = tk.Frame(db_form_card, bg=self.theme["bg_secondary"])
        p_labels.pack(fill="x", padx=16)

        tk.Label(
            p_labels,
            text="Ingestion Queue Scale",
            fg=self.theme["text_status"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")

        self.progress_percent_lbl = tk.Label(
            p_labels,
            text="0%",
            fg=self.theme["text_muted"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10, "bold"),
        )
        self.progress_percent_lbl.pack(side="right")

        self.db_progress = ttk.Progressbar(
            db_form_card, orient="horizontal", mode="determinate"
        )
        self.db_progress.pack(fill="x", padx=16, pady=10)

        self.side_desc = self._create_db_form_row(
            db_form_card,
            "Stored descriptors",
            "0",
            self.theme["accent_green"],
        )
        self.side_land = self._create_db_form_row(
            db_form_card,
            "Committed Landmarks",
            "0",
            self.theme["accent_blue"],
        )
        self.side_keyf = self._create_db_form_row(
            db_form_card, "Mapping frames", "0", self.theme["text_primary"]
        )

        # Static Relational Database Metrics Card
        current_db_card = tk.Frame(
            parent,
            bg=self.theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["border_color"],
        )
        current_db_card.pack(fill="x", pady=12, ipady=12)

        tk.Label(
            current_db_card,
            text="Relational database metrics",
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 16))

        db_meta_box = tk.Frame(current_db_card, bg=self.theme["bg_secondary"])
        db_meta_box.pack(fill="x", padx=16, pady=(0, 12))

        self.db_name_title = tk.Label(
            db_meta_box,
            text=self.controller.get_database_filename_node(),
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        self.db_name_title.pack(fill="x")

        tk.Label(
            db_meta_box,
            text="Active database information",
            fg=self.theme["text_muted"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 9),
            anchor="w",
        ).pack(fill="x")

        self.landmarks_lbl = self._create_telemetry_row(
            current_db_card,
            "Landmarks count",
            "0",
            self.theme["text_primary"],
        )
        self.descriptors_lbl = self._create_telemetry_row(
            current_db_card,
            "Global descriptors",
            "0",
            self.theme["text_primary"],
        )
        self.features_lbl = self._create_telemetry_row(
            current_db_card,
            "Local features clusters",
            "0",
            self.theme["text_primary"],
        )
        self.db_size_lbl = self._create_telemetry_row(
            current_db_card,
            "Size",
            "0.0 KB",
            self.theme["text_primary"],
        )

        # Core Configuration Trace Card
        params_card = tk.Frame(
            parent,
            bg=self.theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["border_color"],
        )
        params_card.pack(fill="x", pady=(12, 0), ipady=12)

        tk.Label(
            params_card,
            text="Computing core configuration",
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 12))

        cv_params: Dict[str, Any] = self.controller.get_cv_params()
        self._create_telemetry_row(
            params_card,
            "Active Model:",
            cv_params.get("featureDetector", "Unknown"),
            self.theme["accent_blue"],
        )
        self._create_telemetry_row(
            params_card,
            "Matching limit:",
            f"{cv_params.get('matchRatio', 0.0):.2f}",
            self.theme["text_primary"],
        )
        self._create_telemetry_row(
            params_card,
            "Neural network max features:",
            str(cv_params.get("xfeatMaxFeatures", 0)),
            self.theme["text_primary"],
        )

    def refresh_ui_metrics_display(self) -> None:
        """Polls the logic controller for the latest static size metrics of the SQLite file."""
        metrics: Dict[str, str] = self.controller.get_active_database_metrics_report()
        self.landmarks_lbl.configure(text=metrics.get("landmarks_count", "0"))
        self.descriptors_lbl.configure(
            text=metrics.get("global_descriptors_count", "0")
        )
        self.features_lbl.configure(text=metrics.get("local_features_count", "0"))
        self.db_size_lbl.configure(text=metrics.get("disk_size_string", "0.0 KB"))

    def _on_ingestion_button_click(self) -> None:
        """Prompts the user for a dataset payload and fires the ingest route on the controller."""
        if self.source_combo.get() == "Dataset":
            target_dir = filedialog.askdirectory(
                title="Select Flight Mission Dataset Root Folder"
            )
            if target_dir:
                self.controller.start_dataset_processing_pipeline(
                    target_dir, view_callback=self
                )

    def ui_signal_process_start(self) -> None:
        """Locks operational controls signaling that ingestion calculation has begun."""
        self.proc_btn.configure(state="disabled")
        self.source_combo.configure(state="disabled")
        self.feed_status_txt.configure(
            text="PROCESSING LOCAL INFERENCE RUNTIME", fg=self.theme["accent_green"]
        )

    def ui_signal_process_update(self, data: Dict[str, Any]) -> None:
        """
        Reflects tracking statistics pushed from the ingestion generator loop.

        Args:
            data: Standardized payload dictionary containing frame metrics.
        """
        self._last_data = data

        self.progress_percent_lbl.configure(
            text=f"{int(data.get('progress_ratio', 0))}%"
        )
        self.db_progress["value"] = data.get("progress_ratio", 0)

        self.ov_feat.configure(
            text=str(data.get("total_features", 0)), fg=self.theme["accent_green"]
        )
        self.ov_land.configure(
            text=str(data.get("total_landmarks", 0)), fg=self.theme["accent_blue"]
        )
        self.ov_frame.configure(
            text=f"{data.get('current_index', 0)}/{data.get('total_items', 0)}",
            fg=self.theme["text_primary"],
        )

        self.side_desc.configure(text=str(data.get("total_features", 0)))
        self.side_land.configure(text=str(data.get("total_landmarks", 0)))
        self.side_keyf.configure(text=str(data.get("total_keyframes", 0)))

        self.update_canvas_frame_view(running=True, data=data)

        # Modulo check prevents I/O choking while processing heavy directories
        if data.get("current_index", 0) % 5 == 0:
            self.refresh_ui_metrics_display()


    def ui_signal_process_complete(self) -> None:
        """Restores hardware locks after the extraction dataset is exhausted."""
        self.proc_btn.configure(state="normal")
        self.source_combo.configure(state="readonly")
        self.feed_status_txt.configure(
            text="INGESTION PROCESS COMPLETE", fg=self.theme["accent_blue"]
        )
        self.refresh_ui_metrics_display()

        self._last_data = None
        self.update_canvas_frame_view(running=False)

    def _create_telemetry_row(
        self, parent: tk.Widget, label: str, val: str, color: str
    ) -> tk.Label:
        """Generates a standard dual-label row for structural tracking."""
        row = tk.Frame(parent, bg=self.theme["bg_secondary"])
        row.pack(fill="x", padx=16, pady=6)

        tk.Label(
            row,
            text=label,
            fg=self.theme["text_status"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")

        v_lbl = tk.Label(
            row,
            text=val,
            fg=color,
            bg=self.theme["bg_secondary"],
            font=("Arial", 10, "bold"),
        )
        v_lbl.pack(side="right")
        return v_lbl

    def _add_stat_box(
        self, parent: tk.Widget, label: str, init_val: str, color: str, col: int
    ) -> tk.Label:
        """Builds an isolated grid cell for metric display above the feed."""
        box = tk.Frame(parent, bg=self.theme["bg_primary"])
        box.grid(row=0, column=col, sticky="nsew")

        tk.Label(
            box,
            text=label,
            fg=self.theme["text_muted"],
            bg=self.theme["bg_primary"],
            font=("Arial", 9),
        ).pack()

        v_lbl = tk.Label(
            box,
            text=init_val,
            fg=color,
            bg=self.theme["bg_primary"],
            font=("Arial", 11, "bold"),
        )
        v_lbl.pack()
        return v_lbl

    def _create_db_form_row(
        self, parent: tk.Widget, label: str, val: str, color: str
    ) -> tk.Label:
        """Generates monospace metric rows specialized for tracking database rows."""
        row = tk.Frame(parent, bg=self.theme["bg_secondary"])
        row.pack(fill="x", padx=16, pady=4)

        tk.Label(
            row,
            text=label,
            fg=self.theme["text_status"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")

        v_lbl = tk.Label(
            row,
            text=val,
            fg=color,
            bg=self.theme["bg_secondary"],
            font=("Courier", 10, "bold"),
        )
        v_lbl.pack(side="right")
        return v_lbl

    def update_canvas_frame_view(
        self, running: bool = False, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Clears and redraws the telemetry video canvas overlay.
        Dynamically extracts, scales, and renders actual image frames.

        Args:
            running: If True, draws the active targeting HUD. Otherwise, idle state text.
            data: The live processor payload containing the raw PIL image.
        """
        self.video_canvas.delete("all")
        w, h = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()

        if w < 10 or h < 10:
            return

        if not running:
            self.video_canvas.create_text(
                w / 2,
                h / 2,
                text="Awaiting dataset input.",
                fill=self.theme["text_muted"],
                font=("Courier", 11),
            )
            return

        # 1. Image Extraction and Rendering
        if data:
            if "frame_pil" in data and data["frame_pil"] is not None:
                pil_img = data["frame_pil"]
                img_w, img_h = pil_img.size

                # Auto-scale the image to maximize window space while maintaining exact aspect ratio
                scale = min(w / img_w, h / img_h)
                new_w, new_h = int(img_w * scale), int(img_h * scale)

                if new_w > 0 and new_h > 0:
                    # LANCZOS resampling ensures the image stays sharp when resized rapidly
                    resample_filter = (
                        Image.Resampling.LANCZOS
                        if hasattr(Image, "Resampling")
                        else Image.ANTIALIAS
                    )
                    resized_img = pil_img.resize((new_w, new_h), resample_filter)

                    self._current_frame = ImageTk.PhotoImage(resized_img)
                    self.video_canvas.create_image(
                        w / 2, h / 2, anchor="center", image=self._current_frame
                    )

            elif "frame_tk" in data and data["frame_tk"] is not None:
                self._current_frame = data["frame_tk"]
                self.video_canvas.create_image(
                    w / 2, h / 2, anchor="center", image=self._current_frame
                )

        # 2. Structural Overlay
        self.video_canvas.create_rectangle(
            10,
            10,
            240,
            35,
            fill=self.theme["bg_primary"],
            outline=self.theme["bg_tertiary"],
        )
        self.video_canvas.create_text(
            15,
            22,
            text="ONNX Inference Core Active",
            fill=self.theme["accent_green"],
            font=("Courier", 9, "bold"),
            anchor="w",
        )
