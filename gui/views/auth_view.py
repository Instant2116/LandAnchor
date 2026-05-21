import tkinter as tk


class AuthorizationView(tk.Frame):
    def __init__(self, parent, controller):
        t = controller.config["theme"]
        super().__init__(parent, bg=t["bg_primary"])
        self.controller = controller
        self.t = t

        self.card = tk.Frame(
            self,
            bg=t["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=t["border_color"],
            highlightthickness=1,
        )
        self.card.place(relx=0.5, rely=0.5, anchor="center", width=440, height=560)

        tk.Label(
            self.card,
            text="Authorization Required",
            fg=t["text_primary"],
            bg=t["bg_secondary"],
            font=("Arial", 20, "bold"),
        ).pack(pady=(40, 4))
        tk.Label(
            self.card,
            text="Hardware Token Identity Validation Engine",
            fg=t["text_status"],
            bg=t["bg_secondary"],
            font=("Arial", 10),
        ).pack(pady=(0, 30))

        tk.Label(
            self.card,
            text="Media Status",
            fg=t["text_secondary"],
            bg=t["bg_secondary"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=40)

        self.media_container = tk.Frame(
            self.card,
            bg=t["bg_success"],
            bd=1,
            relief="solid",
            highlightbackground=t["accent_green"],
        )
        self.media_container.pack(fill="x", padx=40, pady=(6, 20), ipady=12)
        tk.Label(
            self.media_container,
            text="USB Drive Detected",
            fg=t["accent_green"],
            bg=t["bg_success"],
            font=("Arial", 11, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16)
        tk.Label(
            self.media_container,
            text="SECURE-KEY-A4F2",
            fg=t["accent_success"],
            bg=t["bg_success"],
            font=("Arial", 9),
            anchor="w",
        ).pack(fill="x", padx=16)

        tk.Label(
            self.card,
            text="Hardware Token Target Path",
            fg=t["text_secondary"],
            bg=t["bg_secondary"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=40)

        self.key_path_entry = tk.Entry(
            self.card,
            bg=t["bg_tertiary"],
            fg=t["text_primary"],
            insertbackground=t["text_primary"],
            bd=1,
            relief="solid",
            font=("Arial", 11),
        )
        self.key_path_entry.insert(
            0, self.controller.config["system"]["default_key_name"]
        )
        self.key_path_entry.pack(fill="x", padx=40, pady=(6, 4), ipady=10)

        tk.Button(
            self.card,
            text="Generate Fresh Demo Key File to Project Root",
            bg=t["bg_tertiary"],
            fg=t["accent_blue"],
            font=("Arial", 9),
            bd=0,
            cursor="hand2",
            command=self._on_generate_demo_key_clicked,
        ).pack(anchor="w", padx=40, pady=(0, 20))

        tk.Label(
            self.card,
            text="Key Validation Status",
            fg=t["text_secondary"],
            bg=t["bg_secondary"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=40)

        self.status_banner = tk.Frame(self.card, bg=t["bg_tertiary"], bd=0)
        self.status_banner.pack(fill="x", padx=40, pady=(6, 30), ipady=14)

        self.status_text = tk.Label(
            self.status_banner,
            text="Awaiting validation",
            fg=t["text_status"],
            bg=t["bg_tertiary"],
            font=("Arial", 11),
            anchor="w",
        )
        self.status_text.pack(fill="x", padx=16)

        self.action_btn = tk.Button(
            self.card,
            text="Validate Key Signature",
            bg=t["bg_accent"],
            fg=t["text_primary"],
            activebackground=t["bg_accent"],
            activeforeground=t["text_primary"],
            bd=0,
            font=("Arial", 12, "bold"),
            cursor="hand2",
            command=self._on_validate_clicked,
        )
        self.action_btn.pack(fill="x", padx=40, ipady=12)

    def _on_generate_demo_key_clicked(self) -> None:
        self.controller.generate_demo_hardware_key(
            self.key_path_entry.get().strip(), self
        )

    def _on_validate_clicked(self) -> None:
        self.key_path_entry.configure(state="disabled")
        self.action_btn.configure(state="disabled", text="Evaluating Signature...")
        self.status_banner.configure(bg=self.t["bg_accent"])
        self.status_text.configure(
            text="Validating cryptographic checksum...",
            fg=self.t["accent_blue"],
            bg=self.t["bg_accent"],
        )
        self.controller.validate_hardware_key(
            self.key_path_entry.get().strip(), view_callback=self
        )

    def ui_signal_demo_key_generated(self) -> None:
        self.status_banner.configure(bg=self.t["bg_success"])
        self.status_text.configure(
            text="Demo file generated and injected.",
            fg=self.t["accent_green"],
            bg=self.t["bg_success"],
        )

    def ui_signal_auth_success(self, username: str) -> None:
        self.status_banner.configure(bg=self.t["bg_success"])
        self.status_text.configure(
            text=f"Access Granted: {username}",
            fg=self.t["accent_green"],
            bg=self.t["bg_success"],
        )
        self.action_btn.configure(
            bg=self.t["accent_success"], text="Access Approved", state="disabled"
        )

    def ui_signal_auth_failure(self, error_message: str) -> None:
        self.key_path_entry.configure(state="normal")
        self.action_btn.configure(state="normal", text="Validate Key Signature")
        self.status_banner.configure(bg=self.t["bg_error"])
        self.status_text.configure(
            text=error_message, fg=self.t["accent_red"], bg=self.t["bg_error"]
        )