import hashlib
import uuid
import platform
import os
from datetime import datetime
from typing import Optional, Dict, Any


class CryptographicAuthManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.hardware_id = self._generate_hardware_id()

    def _generate_hardware_id(self) -> str:
        """Generates a unique HWID for hardware binding."""
        node = str(uuid.getnode())
        processor = platform.processor()
        system = platform.system()
        raw_hw_string = f"{node}-{processor}-{system}"
        return hashlib.sha256(raw_hw_string.encode("utf-8")).hexdigest().upper()[:32]

    def calculate_pem_fingerprint(self, file_path: str) -> str:
        """Calculates the SHA-256 fingerprint of the key file."""
        if not os.path.exists(file_path):
            return "MISSING_FILE"
        with open(file_path, "rb") as f:
            return f"SHA256:{hashlib.sha256(f.read()).hexdigest().upper()}"

    def register_new_token_offline(
        self, file_path: str, proposed_username: str, target_role: str
    ) -> bool:
        """Registers a token bound to the current HWID."""
        try:
            fingerprint = self.calculate_pem_fingerprint(file_path)
            mixed_signature = f"{fingerprint}-{self.hardware_id}"
            return self.db_manager.add_user_signature(
                proposed_username, mixed_signature, target_role
            )
        except Exception:
            return False

    def authenticate_by_token(self, key_file_path: str) -> Optional[Dict[str, Any]]:
        """Main verification logic: Expiration + HWID + Signature."""
        # Time Bomb Check
        if datetime.now() > datetime(2027, 4, 3):
            return None

        fingerprint = self.calculate_pem_fingerprint(key_file_path)
        mixed_signature = f"{fingerprint}-{self.hardware_id}"

        # Database verification
        return self.db_manager.verify_key_signature(mixed_signature)

    def get_key_metadata(
        self, file_path: str, current_user: Dict[str, Any]
    ) -> Dict[str, str]:
        """Prepares data for the View presentation layer."""
        try:
            stats = os.stat(file_path)
            creation_date = datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%d")
            size_kb = stats.st_size / 1024
        except OSError as e:
            print(f"Skipping file. System/Disk error: {e}")
            creation_date = "None"
            size_kb="0"
        except (ValueError, OverflowError) as e:
            print(f"Skipping file. Invalid timestamp data: {e}")
            creation_date = "None"
            size_kb="0"
        return {
            "file_name": os.path.basename(file_path),
            "file_metrics": f"{size_kb:.1f} KB • Created {creation_date}",
            "hwid": self.hardware_id,
            "fingerprint": self.calculate_pem_fingerprint(file_path),
            "session_owner": f"{current_user.get('username', 'Unknown')} ({current_user.get('role', 'N/A')})",
            "absolute_path": os.path.abspath(file_path),
        }

    def generate_demo_key_file(self, file_path: str):
        """Creates a mock file for demonstration purposes."""
        with open(file_path, "w") as f:
            f.write("--- MOCK_KEY_DATA_CONTENT ---")