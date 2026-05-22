import datetime
import threading


class SystemLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_file_name="landanchor_system.log"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SystemLogger, cls).__new__(cls)
                cls._instance.log_file = log_file_name
                cls._instance.log_buffer = []

                # Initialize or append to the physical file log session header
                try:
                    with open(cls._instance.log_file, "a", encoding="utf-8") as f:
                        f.write(
                            f"\n--- Log Session Initiated At {datetime.datetime.now()} ---\n"
                        )
                except IOError:
                    pass
            return cls._instance

    def _write_log(self, level: str, message: str) -> None:
        """
        Internal synchronized routine to record system trace metrics to memory and disk.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = {"time": timestamp, "level": level, "message": message}

        with self._lock:
            self.log_buffer.append(log_entry)

            # Write immediately to the physical log file container
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] {level:<6} {message}\n")
            except IOError:
                pass

    def info(self, message: str) -> None:
        self._write_log("INFO", message)

    def warn(self, message: str) -> None:
        self._write_log("WARN", message)

    def error(self, message: str) -> None:
        self._write_log("ERROR", message)

    def debug(self, message: str) -> None:
        self._write_log("DEBUG", message)

    def get_entries(self) -> list:
        with self._lock:
            return list(self.log_buffer)

    def clear_logs(self) -> None:
        """
        Wipes active session buffers and resets the tracking log architecture on disk.
        """
        with self._lock:
            self.log_buffer.clear()
            try:
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write(
                        f"--- Log Buffer Cleared At {datetime.datetime.now()} ---\n"
                    )
            except IOError:
                pass
