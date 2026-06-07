import logging
import logging.handlers
import queue
import threading
import collections
import shutil
import os
from datetime import datetime


class SystemLogger:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, log_file_name="landanchor_system.log", max_ram_entries=1000):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemLogger, cls).__new__(cls)
                cls._instance._initialize(log_file_name, max_ram_entries)
            return cls._instance

    def _initialize(self, log_file_name: str, max_ram_entries: int) -> None:
        # Isolate logs in a dedicated directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, log_file_name)

        self.log_buffer = collections.deque(maxlen=max_ram_entries)

        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(logging.DEBUG)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-6s %(message)s", datefmt="%H:%M:%S"
        )

        # Rotating file handler limits disk usage, saved as instance attribute
        self.file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
        self.file_handler.setLevel(logging.INFO)  # Block DEBUG from disk by default
        self.file_handler.setFormatter(formatter)

        # Asynchronous queue prevents disk I/O from blocking execution threads
        log_queue = queue.Queue(-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        self.logger.addHandler(queue_handler)

        # Background thread processes the queue and writes to disk via file_handler
        self.queue_listener = logging.handlers.QueueListener(
            log_queue, self.file_handler, respect_handler_level=True
        )
        self.queue_listener.start()

        self._write_session_header()

    def _write_session_header(self) -> None:
        """Writes the session start marker directly to the file."""
        session_start_msg = f"\n--- Log Session Initiated At {datetime.now()} ---\n"
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(session_start_msg)
        except IOError:
            pass

    def _record_to_ram(self, level: str, message: str) -> None:
        """Maintains the limited RAM buffer strictly for GUI access."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {"time": timestamp, "level": level, "message": message}
        with self._lock:
            self.log_buffer.append(log_entry)

    def set_debug_file_logging(self, enabled: bool) -> None:
        """Toggles writing DEBUG level logs to the physical disk."""
        with self._lock:
            if enabled:
                self.file_handler.setLevel(logging.DEBUG)
                self.info("Disk logging level expanded to include DEBUG traces.")
            else:
                self.file_handler.setLevel(logging.INFO)
                self.info("Disk logging restricted to INFO and above.")

    def info(self, message: str) -> None:
        self.logger.info(message)
        self._record_to_ram("INFO", message)

    def warn(self, message: str) -> None:
        self.logger.warning(message)
        self._record_to_ram("WARN", message)

    def error(self, message: str) -> None:
        self.logger.error(message)
        self._record_to_ram("ERROR", message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)
        self._record_to_ram("DEBUG", message)

    def get_entries(self) -> list:
        """Returns a snapshot of the current GUI buffer."""
        with self._lock:
            return list(self.log_buffer)

    def clear_gui_buffer(self) -> None:
        """Wipes only the RAM buffer for the GUI, leaving the disk log intact."""
        with self._lock:
            self.log_buffer.clear()

    def export_logs(self, destination_path: str) -> None:
        """Copies the full disk log to a user-specified location."""
        try:
            shutil.copy2(self.log_file, destination_path)
            self.info(f"Logs successfully exported to {destination_path}")
        except IOError as e:
            self.error(f"Failed to export logs: {e}")

    def shutdown(self) -> None:
        """Gracefully stops the background logging thread before application exit."""
        self.queue_listener.stop()
