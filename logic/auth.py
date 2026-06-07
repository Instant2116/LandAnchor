import hashlib
import uuid
import platform
import os
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, Any

from logic.logger import SystemLogger


class CryptographicAuthManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = SystemLogger()
        self.hardware_id = self._generate_hardware_id()
        self.logger.debug(
            f"Hardware cryptography module initialized. HWID bound: {self.hardware_id}"
        )

    def _generate_hardware_id(self) -> str:
        """Generates a unique HWID for hardware binding."""
        node = str(uuid.getnode())
        processor = platform.processor()
        system = platform.system()
        raw_hw_string = f"{node}-{processor}-{system}"
        return hashlib.sha256(raw_hw_string.encode("utf-8")).hexdigest().upper()[:32]

    def _extract_expiration_date(self, file_path: str) -> Optional[datetime]:
        """Parses the embedded expiration date from the key file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r"Valid-Until:\s*(\d{4}-\d{2}-\d{2})", content)
                if match:
                    return datetime.strptime(match.group(1), "%Y-%m-%d")
        except UnicodeDecodeError:
            self.logger.warn(
                f"Failed to decode token as text. Ensure keys are PEM/Text format."
            )
        except Exception as e:
            self.logger.error(f"Failed to parse key expiration metadata: {e}")
        return None

    def calculate_pem_fingerprint(self, file_path: str) -> str:
        """Calculates the SHA-256 fingerprint of the key file."""
        if not os.path.exists(file_path):
            self.logger.warn(
                f"Cryptographic token file missing at target path: {file_path}"
            )
            return "MISSING_FILE"

        with open(file_path, "rb") as f:
            return f"SHA256:{hashlib.sha256(f.read()).hexdigest().upper()}"

    def register_new_token_offline(
        self, file_path: str, proposed_username: str, target_role: str
    ) -> bool:
        """Registers a token bound to the current HWID."""
        try:
            expiration_date = self._extract_expiration_date(file_path)
            if not expiration_date:
                self.logger.error(
                    "Registration rejected: Token is missing a valid 'Valid-Until: YYYY-MM-DD' tag."
                )
                return False

            fingerprint = self.calculate_pem_fingerprint(file_path)
            mixed_signature = f"{fingerprint}-{self.hardware_id}"

            success = self.db_manager.add_user_signature(
                proposed_username, mixed_signature, target_role
            )

            if success:
                self.logger.info(
                    f"Hardware token successfully registered for role: {target_role}. Expires: {expiration_date.strftime('%Y-%m-%d')}"
                )
            else:
                self.logger.error(
                    "Database transaction rejected hardware token registration."
                )

            return success
        except Exception as e:
            self.logger.error(f"Hardware token registration failed with exception: {e}")
            return False

    def authenticate_by_token(self, key_file_path: str) -> Optional[Dict[str, Any]]:
        """Main verification logic: Expiration + HWID + Signature."""
        expiration_date = self._extract_expiration_date(key_file_path)

        if not expiration_date:
            self.logger.warn(
                "Authentication rejected: Key file lacks structural expiration metadata."
            )
            return None

        if datetime.now() > expiration_date:
            self.logger.warn(
                f"Authentication rejected: Cryptographic token expired on {expiration_date.strftime('%Y-%m-%d')}."
            )
            return None

        fingerprint = self.calculate_pem_fingerprint(key_file_path)
        mixed_signature = f"{fingerprint}-{self.hardware_id}"

        # Database verification
        user_record = self.db_manager.verify_key_signature(mixed_signature)

        if user_record:
            self.logger.info(
                f"Token authenticated successfully. Session granted for: {user_record.get('username')}"
            )
        else:
            self.logger.warn(
                "Token authentication failed: Cryptographic signature mismatch or invalid HWID."
            )

        return user_record

    def get_key_metadata(
        self, file_path: str, current_user: Dict[str, Any]
    ) -> Dict[str, str]:
        """Prepares data for the View presentation layer."""
        try:
            stats = os.stat(file_path)
            creation_date = datetime.fromtimestamp(stats.st_ctime).strftime("%Y-%m-%d")
            size_kb = stats.st_size / 1024
        except OSError as e:
            self.logger.error(f"Failed to read token metadata. System/Disk error: {e}")
            creation_date = "None"
            size_kb = 0
        except (ValueError, OverflowError) as e:
            self.logger.error(f"Failed to parse token timestamp data: {e}")
            creation_date = "None"
            size_kb = 0

        expiration_date = self._extract_expiration_date(file_path)
        expiration_string = (
            expiration_date.strftime("%Y-%m-%d") if expiration_date else "INVALID"
        )

        return {
            "file_name": os.path.basename(file_path),
            "file_metrics": f"{size_kb:.1f} KB - Created {creation_date} - Expires {expiration_string}",
            "hwid": self.hardware_id,
            "fingerprint": self.calculate_pem_fingerprint(file_path),
            "session_owner": f"{current_user.get('username', 'Unknown')} ({current_user.get('role', 'N/A')})",
            "absolute_path": os.path.abspath(file_path),
        }

    def generate_demo_key_file(self, file_path: str):
        """Creates a mock file with unique content and a 1-year expiration for demonstration purposes."""
        import secrets

        # Calculate exactly 1 year from the current date
        expiration = datetime.now() + relativedelta(years=1)
        expiration_str = expiration.strftime("%Y-%m-%d")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"--- MOCK_KEY_DATA_CONTENT_{secrets.token_hex(16)} ---\n")
            f.write(f"Valid-Until: {expiration_str}\n")

        self.logger.info(
            f"Mock hardware token generated at: {file_path} with expiration {expiration_str}"
        )
