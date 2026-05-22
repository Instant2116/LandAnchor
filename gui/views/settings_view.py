import tkinter as tk
from tkinter import ttk, filedialog


class SettingsView(tk.Frame):
    def __init__(self, parent, controller):
        t = controller.config["theme"]
        super().__init__(parent, bg=t["bg_primary"])
        self.controller = controller
        self.t = t

        title_strip = tk.Frame(self, bg=t["bg_primary"])
        title_strip.pack(fill="x", padx=24, pady=20)
        tk.Label(
            title_strip,
            text="Settings Panel",
            fg=t["text_primary"],
            bg=t["bg_primary"],
            font=("Arial", 22, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_strip,
            text="Administration - Real-time System Configuration",
            fg=t["text_status"],
            bg=t["bg_primary"],
            font=("Arial", 10),
            anchor="w",
        ).pack(fill="x")

        self.tabs_bar = tk.Frame(
            self,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
        )
        self.tabs_bar.pack(fill="x", padx=24, pady=(0, 16))

        self.tab_buttons = {}
        tab_schemas = [
            ("cv-params", "CV Parameters"),
            ("logs", "System Logs"),
            ("keys", "Key Info"),
        ]

        for tab_id, display_label in tab_schemas:
            btn = tk.Button(
                self.tabs_bar,
                text=display_label,
                fg=t["text_secondary"],
                bg=t["bg_secondary"],
                font=("Arial", 10, "bold"),
                bd=0,
                cursor="hand2",
                command=lambda t_id=tab_id: self.switch_internal_tab(t_id),
            )
            btn.pack(side="left", padx=16, pady=6)
            self.tab_buttons[tab_id] = btn

        self.tab_workspace = tk.Frame(self, bg=t["bg_primary"])
        self.tab_workspace.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        self.active_tab_frame = None
        self.switch_internal_tab("cv-params")

    def switch_internal_tab(self, tab_id: str) -> None:
        for key, btn in self.tab_buttons.items():
            if key == tab_id:
                btn.configure(bg=self.t["bg_accent"], fg=self.t["text_primary"])
            else:
                btn.configure(bg=self.t["bg_secondary"], fg=self.t["text_secondary"])

        if self.active_tab_frame is not None:
            self.active_tab_frame.destroy()

        if tab_id == "cv-params":
            self.active_tab_frame = CVParamsTab(
                self.tab_workspace, self.controller, self.t
            )
            self.controller.log_ui_event(
                "Settings sub-view routed to: CV Parameters Tuning Grid", "INFO"
            )
        elif tab_id == "logs":
            self.active_tab_frame = LogsTab(self.tab_workspace, self.controller, self.t)
            self.controller.log_ui_event(
                "Settings sub-view routed to: System Log Trace Console", "INFO"
            )
        elif tab_id == "keys":
            self.active_tab_frame = KeyInfoTab(
                self.tab_workspace, self.controller, self.t
            )
            self.controller.log_ui_event(
                "Settings sub-view routed to: Cryptographic Key Verification Information",
                "INFO",
            )

        self.active_tab_frame.pack(fill="both", expand=True)


class CVParamsTab(tk.Frame):
    def __init__(self, parent, controller, theme):
        super().__init__(parent, bg=theme["bg_primary"])
        self.controller = controller
        self.t = theme
        self.monitor_labels = {}
        self.current_params = self.controller.get_cv_params()

        self.grid_columnconfigure(0, weight=1, uniform="equipment_split_layout")
        self.grid_columnconfigure(1, weight=1, uniform="equipment_split_layout")
        self.grid_rowconfigure(0, weight=1)

        left_pane = tk.Frame(self, bg=theme["bg_primary"])
        left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right_pane = tk.Frame(self, bg=theme["bg_primary"])
        right_pane.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        threshold_card = tk.Frame(
            left_pane,
            bg=theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=theme["border_color"],
            highlightthickness=1,
        )
        threshold_card.pack(fill="both", expand=True, ipady=8)
        tk.Label(
            threshold_card,
            text="Pipeline Tuning Parameters",
            fg=theme["text_primary"],
            bg=theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=20, pady=(12, 12))

        self.feat_slider, self.feat_entry = self._create_slider_control(
            threshold_card, "XFeat Max Features", 100, 2000, "xfeatMaxFeatures",
            "Limits top keypoints array length. Bounds: 100 - 2000"
        )
        self.conf_slider, self.conf_entry = self._create_slider_control(
            threshold_card, "XFeat Confidence Limit", 1, 100, "xfeatConfidenceThreshold",
            "Minimum keypoint extraction score. Bounds: 0.001 - 0.100", multiplier=1000
        )
        self.gem_slider, self.gem_entry = self._create_slider_control(
            threshold_card, "GeM Pooling Power", 1, 10, "gemPoolingPower",
            "Generalized Mean Pooling parameter. Bounds: 1 - 10"
        )
        self.match_slider, self.match_entry = self._create_slider_control(
            threshold_card, "MNN Matching Ratio", 10, 100, "matchRatio",
            "Nearest Neighbor validation constraint. Bounds: 0.10 - 1.00", multiplier=100
        )
        self.ransac_slider, self.ransac_entry = self._create_slider_control(
            threshold_card, "RANSAC Outlier Radius", 5, 100, "ransacThreshold",
            "Max homography pixel deviation. Bounds: 0.5 - 10.0", multiplier=10
        )
        self.inlier_slider, self.inlier_entry = self._create_slider_control(
            threshold_card, "Min Matrix Inliers", 5, 50, "minInliers",
            "Minimum nodes for structural integrity. Bounds: 5 - 50"
        )
        self.topk_slider, self.topk_entry = self._create_slider_control(
            threshold_card, "Top-K Candidates", 1, 50, "topKCandidates",
            "Number of global search candidates. Bounds: 1 - 50"
        )
        self.gdist_slider, self.gdist_entry = self._create_slider_control(
            threshold_card, "Global Distance Threshold", 10, 100, "globalDistanceThreshold",
            "Max cosine distance for global match. Bounds: 0.10 - 1.00", multiplier=100
        )

        btn_box = tk.Frame(left_pane, bg=theme["bg_primary"])
        btn_box.pack(fill="x", pady=12)
        tk.Button(
            btn_box, text="Save State to Disk", bg=theme["bg_accent"], fg=theme["text_primary"],
            bd=0, font=("Arial", 11, "bold"), cursor="hand2", command=self.apply_config_changes,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=10)
        tk.Button(
            btn_box, text="Reset to Baseline", bg=theme["bg_tertiary"], fg=theme["text_primary"],
            bd=1, relief="solid", font=("Arial", 11), cursor="hand2", command=self.reset_to_defaults,
        ).pack(side="right", fill="x", expand=True, padx=(6, 0), ipady=10)

        adv_card = tk.Frame(
            right_pane, bg=theme["bg_secondary"], bd=1, relief="solid", highlightbackground=theme["border_color"],
            highlightthickness=1
        )
        adv_card.pack(fill="x", pady=(0, 12), ipady=12)
        tk.Label(
            adv_card, text="Advanced Architecture Layers", fg=theme["text_primary"], bg=theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=20, pady=(12, 12))

        self.det_combo = self._create_dropdown_control(
            adv_card, "Feature Detector Engine Module", ["XFeat (Local ONNX Engine)"], "featureDetector"
        )
        self.match_combo = self._create_dropdown_control(
            adv_card, "Descriptor Matcher Logic Subsystem", ["MNN Matcher (Vectorized Core)"], "descriptorMatcher"
        )
        self.filt_combo = self._create_dropdown_control(
            adv_card, "Geometric Outlier Filtering Core", ["RANSAC (OpenCV Matrix)"], "outlierFilter"
        )

        vis_row = tk.Frame(adv_card, bg=theme["bg_secondary"])
        vis_row.pack(fill="x", padx=20, pady=6)
        tk.Label(
            vis_row, text="Active Context Debug Visualization", fg=theme["text_secondary"], bg=theme["bg_secondary"],
            font=("Arial", 10),
        ).pack(side="left")

        self.vis_var = tk.BooleanVar(value=bool(self.current_params["debugVisualization"]))
        chk = tk.Checkbutton(
            vis_row, variable=self.vis_var, bg=theme["bg_secondary"], fg=theme["accent_green"],
            activebackground=theme["bg_secondary"], activeforeground=theme["accent_green"],
            selectcolor=theme["bg_success"],
            bd=0, highlightthickness=0, command=self.apply_checkbox_change,
        )
        chk.pack(side="right")

        self.monitor_card = tk.Frame(
            right_pane, bg=theme["bg_secondary"], bd=1, relief="solid", highlightbackground=theme["border_color"],
            highlightthickness=1
        )
        self.monitor_card.pack(fill="both", expand=True)
        tk.Label(
            self.monitor_card, text="Live Runtime Variables Monitor", fg=theme["text_primary"],
            bg=theme["bg_secondary"], font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=20, pady=(12, 8))

        self.dump_frame = tk.Frame(self.monitor_card, bg=theme["bg_secondary"])
        self.dump_frame.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        self._init_static_monitor_rows()
        self.refresh_config_monitor_view()

    def _create_slider_control(self, parent, label, min_v, max_v, param_key, hint, multiplier=1) -> tuple:
        row = tk.Frame(parent, bg=self.t["bg_secondary"])
        row.pack(fill="x", padx=20, pady=3)

        lbl_bar = tk.Frame(row, bg=self.t["bg_secondary"])
        lbl_bar.pack(fill="x")
        tk.Label(
            lbl_bar, text=label, fg=self.t["text_secondary"], bg=self.t["bg_secondary"], font=("Arial", 10),
        ).pack(side="left")

        current_v = self.current_params.get(param_key, 0) * multiplier
        entry = tk.Entry(
            lbl_bar, bg=self.t["bg_tertiary"], fg=self.t["accent_green"], insertbackground=self.t["accent_green"],
            font=("Arial", 10, "bold"), width=8, justify="right", bd=1, relief="solid",
        )
        entry.pack(side="right")

        # Dynamic formatting based on the multiplier required for the variable
        if multiplier == 1000:
            initial_text = f"{self.current_params.get(param_key, 0):.3f}"
        elif multiplier > 1:
            initial_text = f"{self.current_params.get(param_key, 0):.2f}"
        else:
            initial_text = str(self.current_params.get(param_key, 0))
        entry.insert(0, initial_text)

        slider = tk.Scale(
            row, from_=min_v, to=max_v, orient="horizontal", showvalue=False,
            bg=self.t["bg_tertiary"], fg=self.t["text_primary"], troughcolor=self.t["bg_primary"], bd=0,
            highlightthickness=0,
        )
        slider.set(int(current_v))
        slider.pack(fill="x", pady=(2, 2))

        is_internal_update = False

        def _on_slider_motion(event) -> None:
            nonlocal is_internal_update
            if is_internal_update:
                return
            is_internal_update = True
            val = slider.get()
            actual_num = float(val) / multiplier if multiplier > 1 else int(val)
            entry.delete(0, "end")
            if multiplier == 1000:
                entry.insert(0, f"{actual_num:.3f}")
            elif multiplier > 1:
                entry.insert(0, f"{actual_num:.2f}")
            else:
                entry.insert(0, str(actual_num))
            self.current_params[param_key] = actual_num
            self.refresh_config_monitor_view()
            is_internal_update = False

        def _on_slider_release(event) -> None:
            self.controller.update_cv_param(param_key, self.current_params[param_key])

        slider.bind("<B1-Motion>", _on_slider_motion)
        slider.bind("<ButtonRelease-1>", _on_slider_release)

        def _on_entry_focus_out(event) -> None:
            nonlocal is_internal_update
            current_saved_val = self.current_params[param_key]
            raw_input = entry.get().strip()
            try:
                parsed_value = float(raw_input)
                min_b = min_v / multiplier if multiplier > 1 else min_v
                max_b = max_v / multiplier if multiplier > 1 else max_v
                if not (min_b <= parsed_value <= max_b):
                    raise ValueError()
            except ValueError:
                is_internal_update = True
                entry.delete(0, "end")
                if multiplier == 1000:
                    entry.insert(0, f"{current_saved_val:.3f}")
                elif multiplier > 1:
                    entry.insert(0, f"{current_saved_val:.2f}")
                else:
                    entry.insert(0, str(current_saved_val))
                slider.set(int(current_saved_val * multiplier))
                self.controller.log_ui_event(f"Manual input validation error for {param_key}", "ERROR")
                is_internal_update = False

        entry.bind("<FocusOut>", _on_entry_focus_out)
        entry.bind("<Return>", lambda e: parent.focus_set())
        tk.Label(
            row, text=hint, fg=self.t["text_muted"], bg=self.t["bg_secondary"], font=("Arial", 8), anchor="w",
        ).pack(fill="x")
        return slider, entry

    def _create_dropdown_control(self, parent, label, options, param_key) -> ttk.Combobox:
        row = tk.Frame(parent, bg=self.t["bg_secondary"])
        row.pack(fill="x", padx=20, pady=6)
        tk.Label(
            row, text=label, fg=self.t["text_secondary"], bg=self.t["bg_secondary"], font=("Arial", 10), anchor="w",
        ).pack(fill="x", pady=(0, 2))
        combo = ttk.Combobox(row, values=options, state="readonly")
        combo.pack(fill="x")
        combo.set(self.current_params.get(param_key, options[0]))
        combo.bind("<<ComboboxSelected>>", lambda e, c=combo, k=param_key: self._on_dropdown_select(c, k))
        return combo

    def _on_dropdown_select(self, combo: ttk.Combobox, param_key: str) -> None:
        val = combo.get()
        self.current_params[param_key] = val
        self.controller.update_cv_param(param_key, val)
        self.refresh_config_monitor_view()

    def apply_checkbox_change(self) -> None:
        state = bool(self.vis_var.get())
        self.current_params["debugVisualization"] = state
        self.controller.update_cv_param("debugVisualization", state)
        self.refresh_config_monitor_view()

    def _init_static_monitor_rows(self) -> None:
        blueprint = [
            ("xfeat_max_features:", "xfeatMaxFeatures", ""),
            ("xfeat_conf_thresh :", "xfeatConfidenceThreshold", ".3f"),
            ("gem_pooling_power :", "gemPoolingPower", ""),
            ("match_ratio_limit :", "matchRatio", ".2f"),
            ("ransac_threshold  :", "ransacThreshold", ".1f"),
            ("min_inliers_count :", "minInliers", ""),
            ("top_k_candidates  :", "topKCandidates", ""),
            ("global_dist_limit :", "globalDistanceThreshold", ".2f"),
        ]
        for label_text, param_key, rule in blueprint:
            row_frame = tk.Frame(self.dump_frame, bg=self.t["bg_secondary"])
            row_frame.pack(fill="x", pady=2)
            tk.Label(
                row_frame, text=label_text, fg=self.t["text_status"], bg=self.t["bg_secondary"], font=("Courier", 10),
            ).pack(side="left")
            val_lbl = tk.Label(
                row_frame, text="", fg=self.t["accent_green"], bg=self.t["bg_secondary"], font=("Courier", 10, "bold"),
            )
            val_lbl.pack(side="right")
            self.monitor_labels[param_key] = (val_lbl, rule)

    def refresh_config_monitor_view(self) -> None:
        for key, (lbl, rule) in self.monitor_labels.items():
            val = self.current_params.get(key, 0)
            text = f"{val:{rule}}" if rule and isinstance(val, (int, float)) else str(val)
            lbl.configure(text=text)

    def apply_config_changes(self) -> None:
        self.controller.save_configuration_profile()

    def reset_to_defaults(self) -> None:
        self.current_params = self.controller.reset_cv_params_to_defaults()

        self.feat_slider.set(self.current_params["xfeatMaxFeatures"])
        self.feat_entry.delete(0, "end")
        self.feat_entry.insert(0, str(self.current_params["xfeatMaxFeatures"]))

        self.conf_slider.set(int(self.current_params["xfeatConfidenceThreshold"] * 1000))
        self.conf_entry.delete(0, "end")
        self.conf_entry.insert(0, f"{self.current_params['xfeatConfidenceThreshold']:.3f}")

        self.gem_slider.set(self.current_params["gemPoolingPower"])
        self.gem_entry.delete(0, "end")
        self.gem_entry.insert(0, str(self.current_params["gemPoolingPower"]))

        self.match_slider.set(int(self.current_params["matchRatio"] * 100))
        self.match_entry.delete(0, "end")
        self.match_entry.insert(0, f"{self.current_params['matchRatio']:.2f}")

        self.ransac_slider.set(int(self.current_params["ransacThreshold"] * 10))
        self.ransac_entry.delete(0, "end")
        self.ransac_entry.insert(0, f"{self.current_params['ransacThreshold']:.2f}")

        self.inlier_slider.set(self.current_params["minInliers"])
        self.inlier_entry.delete(0, "end")
        self.inlier_entry.insert(0, str(self.current_params["minInliers"]))

        self.topk_slider.set(self.current_params["topKCandidates"])
        self.topk_entry.delete(0, "end")
        self.topk_entry.insert(0, str(self.current_params["topKCandidates"]))

        self.gdist_slider.set(int(self.current_params["globalDistanceThreshold"] * 100))
        self.gdist_entry.delete(0, "end")
        self.gdist_entry.insert(0, f"{self.current_params['globalDistanceThreshold']:.2f}")

        self.vis_var.set(self.current_params["debugVisualization"])
        self.refresh_config_monitor_view()

class LogsTab(tk.Frame):
    def __init__(self, parent, controller, theme):
        super().__init__(parent, bg=theme["bg_primary"])
        self.controller = controller
        self.t = theme

        viewer_card = tk.Frame(
            self,
            bg=theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=theme["border_color"],
        )
        viewer_card.pack(fill="both", expand=True)

        title_bar = tk.Frame(viewer_card, bg=theme["bg_secondary"])
        title_bar.pack(fill="x", padx=20, pady=16)
        tk.Label(
            title_bar,
            text="System Log Trace Console",
            fg=theme["text_primary"],
            bg=theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        btn_box = tk.Frame(title_bar, bg=theme["bg_secondary"])
        btn_box.pack(side="right")
        tk.Button(
            btn_box,
            text="Export",
            bg=theme["bg_tertiary"],
            fg=theme["text_primary"],
            bd=1,
            relief="solid",
            font=("Arial", 9),
            cursor="hand2",
            command=self.export_logs_to_file,
        ).pack(side="left", padx=4)
        tk.Button(
            btn_box,
            text="Clear",
            bg=theme["bg_tertiary"],
            fg=theme["text_primary"],
            bd=1,
            relief="solid",
            font=("Arial", 9),
            cursor="hand2",
            command=self.clear_logs_canvas,
        ).pack(side="left", padx=4)

        self.text_area = tk.Text(
            viewer_card,
            bg=theme["bg_primary"],
            fg=theme["text_secondary"],
            insertbackground=theme["text_primary"],
            bd=0,
            font=("Courier", 10),
            padx=16,
            pady=16,
        )
        self.text_area.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.text_area.tag_configure("TIMESTAMP", foreground=theme["text_muted"])
        self.text_area.tag_configure("INFO", foreground=theme["accent_blue"])
        self.text_area.tag_configure("WARN", foreground=theme["accent_yellow"])
        self.text_area.tag_configure("ERROR", foreground=theme["accent_red"])
        self.text_area.tag_configure("TEXT", foreground=theme["text_secondary"])

        self.populate_logs_view()

    def populate_logs_view(self) -> None:
        self.text_area.configure(state="normal")
        self.text_area.delete("1.0", "end")
        for entry in self.controller.get_system_logs():
            self.text_area.insert("end", f"[{entry['time']}] ", "TIMESTAMP")
            self.text_area.insert("end", f"{entry['level']:<6} ", entry["level"])
            self.text_area.insert("end", f"{entry['message']}\n", "TEXT")
        self.text_area.configure(state="disabled")

    def export_logs_to_file(self) -> None:
        target_path = filedialog.asksaveasfilename(
            initialfile="landanchor_export.log",
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("All Files", "*.*")],
        )
        if target_path:
            self.controller.export_system_logs(target_path)

    def clear_logs_canvas(self) -> None:
        self.controller.clear_system_logs()
        self.populate_logs_view()


class KeyInfoTab(tk.Frame):
    def __init__(self, parent, controller, theme):
        super().__init__(parent, bg=theme["bg_primary"])
        self.controller = controller
        self.t = theme

        self.left_box = tk.Frame(
            self,
            bg=theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=theme["border_color"],
        )
        self.left_box.pack(
            side="left", fill="both", expand=True, padx=(0, 12), ipadx=16, ipady=16
        )

        l_title_bar = tk.Frame(self.left_box, bg=theme["bg_secondary"])
        l_title_bar.pack(fill="x", padx=20, pady=16)
        tk.Label(
            l_title_bar,
            text="Security Key File Registration Status",
            fg=theme["text_primary"],
            bg=theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        self.key_list_frame = tk.Frame(self.left_box, bg=theme["bg_secondary"])
        self.key_list_frame.pack(fill="both", expand=True)

        self.right_box = tk.Frame(
            self,
            bg=theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=theme["border_color"],
        )
        self.right_box.pack(
            side="right", fill="both", expand=True, padx=(12, 0), ipadx=16, ipady=16
        )
        tk.Label(
            self.right_box,
            text="Identity Key Metadata Information",
            fg=theme["text_primary"],
            bg=theme["bg_secondary"],
            font=("Arial", 11, "bold"),
        ).pack(anchor="w", padx=20, pady=(16, 16))

        self.meta_frame = tk.Frame(self.right_box, bg=theme["bg_secondary"])
        self.meta_frame.pack(fill="both", expand=True)

        self.render_static_key_information()

    def render_static_key_information(self) -> None:
        meta = self.controller.get_active_key_metadata_report()

        k_row = tk.Frame(
            self.key_list_frame,
            bg=self.t["bg_primary"],
            bd=1,
            relief="solid",
            highlightbackground=self.t["bg_tertiary"],
        )
        k_row.pack(fill="x", padx=20, pady=6, ipady=8)
        tk.Label(
            k_row,
            text=meta.get("file_name", "Unknown"),
            fg=self.t["text_primary"],
            bg=self.t["bg_primary"],
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12)
        tk.Label(
            k_row,
            text=meta.get("file_metrics", "No metrics"),
            fg=self.t["text_muted"],
            bg=self.t["bg_primary"],
            font=("Arial", 9),
            anchor="w",
        ).pack(fill="x", padx=12)

        info_schema = [
            ("Hardware Lock Identifier (HWID)", meta.get("hwid", "Unknown")),
            ("Key Subsystem Type", "RSA 2048-bit Private Key File"),
            ("Cryptographic Fingerprint", meta.get("fingerprint", "Unknown")),
            ("Active Authorized Session", meta.get("session_owner", "Unknown")),
            ("Token Target File Path", meta.get("absolute_path", "Unknown")),
        ]

        for title, value in info_schema:
            meta_block = tk.Frame(self.meta_frame, bg=self.t["bg_secondary"])
            meta_block.pack(fill="x", padx=20, pady=6)
            tk.Label(
                meta_block,
                text=title,
                fg=self.t["text_muted"],
                bg=self.t["bg_secondary"],
                font=("Arial", 9),
                anchor="w",
            ).pack(fill="x")

            c = (
                self.t["accent_green"]
                if "Fingerprint" in title or "HWID" in title
                else self.t["text_primary"]
            )
            f = (
                ("Courier", 9, "bold")
                if "Fingerprint" in title or "HWID" in title
                else ("Arial", 10)
            )

            lbl = tk.Label(
                meta_block,
                text=value,
                fg=c,
                bg=self.t["bg_secondary"],
                font=f,
                anchor="w",
                justify="left",
            )
            if "Fingerprint" in title or "HWID" in title:
                lbl.configure(wraplength=340)
            lbl.pack(fill="x")