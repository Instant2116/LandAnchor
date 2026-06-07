import tkinter as tk
from typing import Any


class AuthView(tk.Frame):
    """
    View for the authorization screen.

    It handles the UI for entering the .pem key file path and shows the validation status.
    """

    def __init__(self, parent: tk.Widget, controller: Any) -> None:
        """
        Initializes the view and UI components.

        Args:
            parent: The parent Tkinter widget.
            controller: The application controller.
        """
        self.controller = controller
        self.theme = controller.app_config["theme"]
        super().__init__(parent, bg=self.theme["bg_primary"])

        self._build_ui()

    def _build_ui(self) -> None:
        """Builds the UI elements."""
        self.card = tk.Frame(
            self,
            bg=self.theme["bg_secondary"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["border_color"],
            highlightthickness=1,
        )
        self.card.place(relx=0.5, rely=0.5, anchor="center", width=440, height=560)

        # Header section
        tk.Label(
            self.card,
            text="Authorization required",
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 20, "bold"),
        ).pack(pady=(40, 4))

        tk.Label(
            self.card,
            text="Cryptographic key required",
            fg=self.theme["text_status"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10),
        ).pack(pady=(0, 30))

        # Media status section
        tk.Label(
            self.card,
            text="File status",
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=40)

        self.status_box = tk.Frame(
            self.card,
            bg=self.theme["bg_success"],
            bd=1,
            relief="solid",
            highlightbackground=self.theme["accent_green"],
        )
        self.status_box.pack(fill="x", padx=40, pady=(6, 20), ipady=12)

        tk.Label(
            self.status_box,
            text="PEM file detected",
            fg=self.theme["accent_green"],
            bg=self.theme["bg_success"],
            font=("Arial", 11, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16)

        tk.Label(
            self.status_box,
            text="SHA-256 signature ready",
            fg=self.theme["accent_success"],
            bg=self.theme["bg_success"],
            font=("Arial", 9),
            anchor="w",
        ).pack(fill="x", padx=16)

        # Key path input section
        tk.Label(
            self.card,
            text="PEM file target path",
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=40)

        self.path_input = tk.Entry(
            self.card,
            bg=self.theme["bg_tertiary"],
            fg=self.theme["text_primary"],
            insertbackground=self.theme["text_primary"],
            bd=1,
            relief="solid",
            font=("Arial", 11),
        )
        self.path_input.insert(0, self.controller.app_config["system"]["default_key_name"])
        self.path_input.pack(fill="x", padx=40, pady=(6, 4), ipady=10)

        tk.Button(
            self.card,
            text="Generate demo .pem file",
            bg=self.theme["bg_tertiary"],
            fg=self.theme["accent_blue"],
            font=("Arial", 9),
            bd=0,
            cursor="hand2",
            command=self._on_generate_click,
        ).pack(anchor="w", padx=40, pady=(0, 20))

        # Status output section
        tk.Label(
            self.card,
            text="Key validation status",
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_secondary"],
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=40)

        self.msg_box = tk.Frame(self.card, bg=self.theme["bg_tertiary"], bd=0)
        self.msg_box.pack(fill="x", padx=40, pady=(6, 30), ipady=14)

        self.msg_label = tk.Label(
            self.msg_box,
            text="Awaiting validation",
            fg=self.theme["text_status"],
            bg=self.theme["bg_tertiary"],
            font=("Arial", 11),
            anchor="w",
        )
        self.msg_label.pack(fill="x", padx=16)

        # Action section
        self.validate_btn = tk.Button(
            self.card,
            text="Validate key signature",
            bg=self.theme["bg_accent"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_accent"],
            activeforeground=self.theme["text_primary"],
            bd=0,
            font=("Arial", 12, "bold"),
            cursor="hand2",
            command=self._on_validate_click,
        )
        self.validate_btn.pack(fill="x", padx=40, ipady=12)

    def _on_generate_click(self) -> None:
        """Requests a demo .pem file from the controller."""
        target_path = self.path_input.get().strip()
        self.controller.generate_demo_hardware_key(target_path, self)

    def _on_validate_click(self) -> None:
        """Locks the UI and sends the validation request."""
        self.path_input.configure(state="disabled")
        self.validate_btn.configure(state="disabled", text="Evaluating signature...")
        self.msg_box.configure(bg=self.theme["bg_accent"])
        self.msg_label.configure(
            text="Validating SHA-256 checksum...",
            fg=self.theme["accent_blue"],
            bg=self.theme["bg_accent"],
        )

        target_path = self.path_input.get().strip()
        self.controller.validate_hardware_key(target_path, view_callback=self)

    def show_demo_generated(self) -> None:
        """Updates UI to show demo file was generated."""
        self.msg_box.configure(bg=self.theme["bg_success"])
        self.msg_label.configure(
            text="Demo .pem file generated.",
            fg=self.theme["accent_green"],
            bg=self.theme["bg_success"],
        )

    def show_auth_success(self, username: str) -> None:
        """Updates UI for successful authorization."""
        self.msg_box.configure(bg=self.theme["bg_success"])
        self.msg_label.configure(
            text=f"Access granted: {username}",
            fg=self.theme["accent_green"],
            bg=self.theme["bg_success"],
        )
        self.validate_btn.configure(
            bg=self.theme["accent_success"], text="Access approved", state="disabled"
        )

    def show_auth_failure(self, error_message: str) -> None:
        """Updates UI for failed authorization and enables retry."""
        self.path_input.configure(state="normal")
        self.validate_btn.configure(state="normal", text="Validate key signature")
        self.msg_box.configure(bg=self.theme["bg_error"])
        self.msg_label.configure(
            text=error_message, fg=self.theme["accent_red"], bg=self.theme["bg_error"]
        )
