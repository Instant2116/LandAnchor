import tkinter as tk
from tkinter import messagebox
from typing import Any

from db.db_manager import DBManager
from logic.auth import CryptographicAuthManager
from logic.logger import SystemLogger
from logic.settings_manager import SettingsManager
from logic.operator_manager import OperatorManager

from gui.views.auth_view import AuthView
from gui.views.operator_view import OperatorView
from gui.views.preparation_view import PreparationView
from gui.views.settings_view import SettingsView
from logic.data_processor import DataProcessor


class LandAnchorApp(tk.Tk):
    def __init__(self, db_path: str):
        super().__init__()
        self.title("Drone Navigation System - LandAnchor")
        self.geometry("1440x1000")
        self.minsize(1280, 720)

        # Load first
        self.settings_manager = SettingsManager("system_preferences.json")
        self.app_config = self.settings_manager.data

        self.configure(bg=self.app_config["theme"]["bg_primary"])

        self.db_manager = DBManager(db_path)
        self.auth_manager = CryptographicAuthManager(self.db_manager)

        self.logger = SystemLogger()
        write_debug = self.app_config.get("cv_defaults", {}).get(
            "writeDebugLogs", False
        )
        self.logger.set_debug_file_logging(write_debug)

        self.operator_manager = OperatorManager(self.db_manager, self.settings_manager)

        # Initialize processor and register agnostic thread hooks
        self.data_processor = DataProcessor(self.db_manager, self.settings_manager)
        self.data_processor.register_hooks(
            progress_callback=self._handle_processor_update,
            completion_callback=self._handle_processor_complete,
        )

        self.current_user = None
        self.is_hardware_key_valid = False
        self.active_view_name = None
        self.current_frame = None
        self.prep_view = None

        self.grid_columnconfigure(0, weight=0, minsize=260)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = tk.Frame(self, bg=self.app_config["theme"]["bg_secondary"], bd=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.workspace = tk.Frame(self, bg=self.app_config["theme"]["bg_primary"], bd=0)
        self.workspace.grid(row=0, column=1, sticky="nsew")

        self.nav_buttons = {}
        self._build_sidebar_layout()
        self.show_view("Authorization", AuthView)

    # --- OPERATOR & PREPARATION DELEGATION ---
    def register_active_operator_view(self, view_instance: object) -> None:
        self.operator_manager.register_view(view_instance)

    def register_preparation_view(self, view_instance: object) -> None:
        """Stores a reference to the active view to push thread-safe UI updates."""
        self.prep_view = view_instance

    def start_operator_simulation(self, dataset_dir: str) -> None:
        self.operator_manager.start_dataset_simulation(dataset_dir)

    def start_dataset_processing_pipeline(
        self, target_dir: str, view_callback: object
    ) -> None:
        """Triggers the background extraction pipeline and locks the UI."""
        self.register_preparation_view(view_callback)

        if self.prep_view and self.prep_view.winfo_exists():
            self.prep_view.ui_signal_process_start()

        self.data_processor.start_dataset_processing_pipeline(target_dir)

    def _handle_processor_update(self, payload: dict) -> None:
        """Thread-safe bridge. Receives the payload from the background thread and routes to UI."""
        if getattr(self, "prep_view", None) and self.prep_view.winfo_exists():
            self.prep_view.after(0, self.prep_view.ui_signal_process_update, payload)

    def _handle_processor_complete(self) -> None:
        """Thread-safe bridge to unlock the UI once the pipeline terminates."""
        if getattr(self, "prep_view", None) and self.prep_view.winfo_exists():
            self.prep_view.after(0, self.prep_view.ui_signal_process_complete)

    # --- SETTINGS DELEGATION ---
    def get_cv_params(self) -> dict:
        return self.settings_manager.get_cv_params()

    def update_cv_param(self, key: str, value: Any) -> None:
        self.settings_manager.update_cv_param(key, value)
        if key == "writeDebugLogs":
            self.logger.set_debug_file_logging(value)

    def save_configuration_profile(self) -> None:
        self.settings_manager.save()

    def reset_cv_params_to_defaults(self) -> dict:
        return self.settings_manager.reset_to_defaults()

    # --- LOGGER DELEGATION ---
    def log_ui_event(self, message: str, level: str) -> None:
        getattr(self.logger, level.lower())(message)

    def get_system_logs(self) -> list:
        return self.logger.get_entries()

    def export_system_logs(self, target_path: str) -> None:
        self.logger.export_logs(target_path)

    def clear_system_logs(self) -> None:
        self.logger.clear_gui_buffer()

    def destroy(self) -> None:
        if hasattr(self, "logger"):
            self.logger.shutdown()
        super().destroy()

    # --- NAVIGATION & VIEW MANAGEMENT ---
    def _build_sidebar_layout(self) -> None:
        header_frame = tk.Frame(
            self.sidebar, bg=self.app_config["theme"]["bg_secondary"]
        )
        header_frame.pack(fill="x", padx=24, pady=(30, 40))

        tk.Label(
            header_frame,
            text="Autonomus Navigation System",
            fg=self.app_config["theme"]["accent_blue"],
            bg=self.app_config["theme"]["bg_secondary"],
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            header_frame,
            text=self.app_config["system"]["version"],
            fg=self.app_config["theme"]["text_muted"],
            bg=self.app_config["theme"]["bg_secondary"],
            font=("Arial", 9),
            anchor="w",
        ).pack(fill="x")

        menu_schema = [
            ("Authorization", "Authorization"),
            ("Operator Dashboard", "Operator Dashboard"),
            ("Preparation", "Preparation"),
            ("Settings", "Settings"),
        ]

        for internal_key, display_label in menu_schema:
            btn = tk.Button(
                self.sidebar,
                text=f"  {display_label}",
                fg=self.app_config["theme"]["text_secondary"],
                bg=self.app_config["theme"]["bg_secondary"],
                activebackground=self.app_config["theme"]["bg_tertiary"],
                activeforeground="#ffffff",
                font=("Arial", 11, "normal"),
                anchor="w",
                bd=0,
                cursor="hand2",
                command=lambda k=internal_key: self._handle_navigation(k),
            )
            btn.pack(fill="x", padx=16, pady=4, ipady=10)
            self.nav_buttons[internal_key] = btn

    def _handle_navigation(self, target_view: str) -> None:
        if target_view == "Authorization":
            self.show_view("Authorization", AuthView)
            return
        if not self.is_hardware_key_valid:
            messagebox.showwarning(
                "Security Restriction",
                "Access Denied: Missing verified hardware key token.",
            )
            return

        if target_view == "Operator Dashboard":
            self.show_view("Operator Dashboard", OperatorView)
        elif target_view == "Preparation":
            self.show_view("Preparation", PreparationView)
        elif target_view == "Settings":
            self.show_view("Settings", SettingsView)

    def show_view(self, view_name: str, view_class) -> None:
        self.active_view_name = view_name
        for key, button in self.nav_buttons.items():
            if key == view_name:
                button.configure(
                    bg=self.app_config["theme"]["bg_accent"],
                    fg="#ffffff",
                    font=("Arial", 11, "bold"),
                )
            else:
                button.configure(
                    bg=self.app_config["theme"]["bg_secondary"],
                    fg=self.app_config["theme"]["text_secondary"],
                    font=("Arial", 11, "normal"),
                )

        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = view_class(parent=self.workspace, controller=self)
        self.current_frame.pack(fill="both", expand=True)

    # --- AUTH & DB HELPERS ---
    def generate_demo_hardware_key(self, path: str, view: AuthView) -> None:
        self.auth_manager.generate_demo_key_file(path)
        registration_success = self.auth_manager.register_new_token_offline(
            file_path=path, proposed_username="Demo_User", target_role="Technician"
        )
        if registration_success:
            view.show_demo_generated()
        else:
            view.show_auth_failure("Database registration for demo key failed.")

    def validate_hardware_key(self, path: str, view_callback: AuthView) -> None:
        user = self.auth_manager.authenticate_by_token(path)
        if user:
            self.current_user = user
            self.is_hardware_key_valid = True
            view_callback.show_auth_success(user["username"])
        else:
            view_callback.show_auth_failure("Invalid or expired hardware token.")

    def get_database_filename_node(self) -> str:
        return self.db_manager.get_database_filename_node()

    def get_active_key_metadata_report(self) -> dict:
        return self.auth_manager.get_key_metadata(
            "hardware_key.pem", self.current_user or {}
        )

    def get_active_database_metrics_report(self) -> dict:
        return self.db_manager.get_active_database_metrics_report()
