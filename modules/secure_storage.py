"""
secure_storage.py — Centralized security utilities for data-at-rest protection.

Provides:
  - Secure file deletion (multi-pass overwrite before removal)
  - Secure temporary file management (auto-wipe on exit)
  - Field-level encryption for database columns (Fernet AES-128)
  - HMAC-SHA256 deterministic hashing for encrypted field lookups
  - Encrypted audit log handler (each log line encrypted at rest)
  - File-level encryption / decryption helpers
  - Key protection utilities
"""

import os
import sys
import hmac
import json
import hashlib
import secrets
import tempfile
import logging
import stat
from contextlib import contextmanager
from typing import Optional, List

from cryptography.fernet import Fernet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Key Management ────────────────────────────────────────────────────────

def _load_raw_key() -> str:
    """
    Load the raw Fernet key string from the key file or environment variable.
    Returns the key string, or raises EnvironmentError if not found.
    """
    key = None

    # 1. Try persistent key file first
    key_file = os.path.join(os.path.dirname(__file__), "..", config.FACE_DB_KEY_FILE)
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key = f.read().strip()
        if key:
            os.environ[config.FACE_DB_KEY_ENV] = key

    # 2. Fall back to environment variable
    if not key:
        key = os.environ.get(config.FACE_DB_KEY_ENV)

    if not key:
        raise EnvironmentError(
            f"[ERROR] Encryption key not found.\n"
            f"Run enrollment or set '{config.FACE_DB_KEY_ENV}' env variable."
        )
    return key


def get_fernet() -> Fernet:
    """Retrieve the Fernet encryption instance."""
    key = _load_raw_key()
    return Fernet(key.encode() if isinstance(key, str) else key)


def protect_key_file(key_file_path: str) -> None:
    """
    Set restrictive file permissions on the key file.
    On Windows: removes inheritance and sets owner-only access.
    On Unix: sets mode 0o600 (owner read/write only).
    """
    try:
        if os.name == 'nt':
            # Windows: use icacls to restrict access
            import subprocess
            username = os.environ.get('USERNAME', '')
            if username:
                subprocess.run(
                    ['icacls', key_file_path, '/inheritance:r',
                     '/grant:r', f'{username}:(R,W)'],
                    capture_output=True, timeout=10,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                )
        else:
            os.chmod(key_file_path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception as e:
        print(f"[SECURITY] Could not restrict key file permissions: {e}")


def generate_and_save_key() -> str:
    """
    Generate a new Fernet key, save it to the key file with
    restrictive permissions, and set the environment variable.
    Returns the key string.
    """
    key = Fernet.generate_key().decode()
    os.environ[config.FACE_DB_KEY_ENV] = key

    key_file = os.path.join(os.path.dirname(__file__), "..", config.FACE_DB_KEY_FILE)
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    with open(key_file, "w") as f:
        f.write(key)

    protect_key_file(key_file)
    print(f"[SECURITY] Encryption key generated and saved with restricted permissions.")
    return key


# ─── Secure File Operations ────────────────────────────────────────────────

def secure_delete(filepath: str, passes: int = 3) -> bool:
    """
    Securely delete a file by overwriting with random data before removal.

    Uses the DoD 5220.22-M standard: multiple passes of random data
    followed by a final zero pass, then OS-level deletion.

    Args:
        filepath: Path to the file to securely delete.
        passes: Number of overwrite passes (default 3).

    Returns:
        True if file was securely deleted.
    """
    if not os.path.exists(filepath):
        return False

    try:
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            os.remove(filepath)
            return True

        with open(filepath, "r+b") as f:
            for _ in range(passes):
                f.seek(0)
                # Write in chunks to handle large files
                remaining = file_size
                while remaining > 0:
                    chunk = min(remaining, 65536)
                    f.write(os.urandom(chunk))
                    remaining -= chunk
                f.flush()
                os.fsync(f.fileno())

            # Final pass: zeros
            f.seek(0)
            remaining = file_size
            while remaining > 0:
                chunk = min(remaining, 65536)
                f.write(b'\x00' * chunk)
                remaining -= chunk
            f.flush()
            os.fsync(f.fileno())

        os.remove(filepath)
        return True
    except Exception as e:
        print(f"[SECURITY] Secure delete failed for '{filepath}': {e}")
        # Fall back to regular delete
        try:
            os.remove(filepath)
        except Exception:
            pass
        return False


def get_secure_temp_dir() -> str:
    """Get or create a secure temporary directory for transient files."""
    secure_dir = os.path.join(
        os.path.dirname(__file__), "..", config.SECURE_TEMP_DIR
    )
    secure_dir = os.path.abspath(secure_dir)
    os.makedirs(secure_dir, exist_ok=True)
    return secure_dir


@contextmanager
def secure_temp_file(suffix=".wav", prefix="sec_"):
    """
    Context manager that creates a temporary file in the secure temp directory
    and securely deletes it when the context exits.

    Usage:
        with secure_temp_file(suffix=".wav") as tmp_path:
            record_audio(tmp_path)
            process(tmp_path)
        # File is securely wiped and deleted here
    """
    tmp_dir = get_secure_temp_dir()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=tmp_dir)
    os.close(fd)
    try:
        yield tmp_path
    finally:
        secure_delete(tmp_path)


def cleanup_secure_temp() -> int:
    """
    Securely wipe all files in the secure temp directory.
    Called on application shutdown to ensure no residual data.

    Returns:
        Number of files cleaned up.
    """
    secure_dir = get_secure_temp_dir()
    count = 0
    if os.path.exists(secure_dir):
        for f in os.listdir(secure_dir):
            filepath = os.path.join(secure_dir, f)
            if os.path.isfile(filepath):
                secure_delete(filepath)
                count += 1
    return count


def cleanup_residual_wav_files() -> int:
    """
    Find and securely delete any WAV files left in the project root
    from previous runs (before secure storage was implemented).

    Returns:
        Number of files cleaned up.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    count = 0
    wav_patterns = ["user_voice_auth.wav", "user_voice_enroll.wav",
                    "temp_recording.wav"]
    for filename in wav_patterns:
        filepath = os.path.join(project_root, filename)
        if os.path.exists(filepath):
            print(f"[SECURITY] Cleaning up residual file: {filename}")
            secure_delete(filepath)
            count += 1

    # Also clean any .wav files in the project root
    for f in os.listdir(project_root):
        if f.lower().endswith('.wav') and os.path.isfile(os.path.join(project_root, f)):
            print(f"[SECURITY] Cleaning up residual WAV: {f}")
            secure_delete(os.path.join(project_root, f))
            count += 1
    return count


# ─── Field-Level Encryption ────────────────────────────────────────────────

def encrypt_field(plaintext: str) -> bytes:
    """Encrypt a string field for secure database storage."""
    fernet = get_fernet()
    return fernet.encrypt(plaintext.encode("utf-8"))


def decrypt_field(ciphertext: bytes) -> str:
    """Decrypt an encrypted database field back to its original string."""
    fernet = get_fernet()
    return fernet.decrypt(ciphertext).decode("utf-8")


def hmac_hash(value: str, purpose: str = "lookup") -> str:
    """
    Create a deterministic HMAC-SHA256 hash for encrypted field lookups.

    This allows searching for a user by user_id without storing the
    plaintext user_id in the database. The hash is deterministic (same
    input always produces the same hash) but irreversible.

    Args:
        value: The plaintext value to hash.
        purpose: A domain separator to prevent cross-field hash collisions.

    Returns:
        A 64-character hex string (HMAC-SHA256 digest).
    """
    raw_key = _load_raw_key()
    hmac_key = hashlib.sha256(f"hmac_key:{raw_key}".encode()).digest()
    message = f"{purpose}:{value}".encode("utf-8")
    return hmac.new(hmac_key, message, hashlib.sha256).hexdigest()


# ─── Encrypted Audit Log Handler ──────────────────────────────────────────

class EncryptedLogHandler(logging.Handler):
    """
    Custom logging handler that encrypts each log line with Fernet
    before writing to the log file.

    Each line in the log file is an independently encrypted token.
    This ensures that even if the log file is accessed, the contents
    (user IDs, IP addresses, failure reasons) remain confidential.
    """

    def __init__(self, filepath: str, max_bytes: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        super().__init__()
        self.filepath = filepath
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def emit(self, record):
        try:
            msg = self.format(record)
            fernet = get_fernet()
            encrypted_line = fernet.encrypt(msg.encode("utf-8")).decode("ascii")

            # Check if rotation is needed
            if os.path.exists(self.filepath):
                try:
                    if os.path.getsize(self.filepath) >= self.max_bytes:
                        self._rotate()
                except OSError:
                    pass

            with open(self.filepath, "a") as f:
                f.write(encrypted_line + "\n")
        except Exception:
            self.handleError(record)

    def _rotate(self):
        """Rotate encrypted log files."""
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.filepath}.{i}"
            dst = f"{self.filepath}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    secure_delete(dst)
                os.rename(src, dst)
        if os.path.exists(self.filepath):
            dst = f"{self.filepath}.1"
            if os.path.exists(dst):
                secure_delete(dst)
            os.rename(self.filepath, dst)


def decrypt_audit_log(filepath: str) -> List[str]:
    """
    Decrypt and read an encrypted audit log file.

    Args:
        filepath: Path to the encrypted log file.

    Returns:
        List of decrypted log line strings (JSON).
    """
    lines = []
    if not os.path.exists(filepath):
        return lines

    fernet = get_fernet()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    decrypted = fernet.decrypt(line.encode("ascii")).decode("utf-8")
                    lines.append(decrypted)
                except Exception:
                    # Could be a legacy unencrypted line
                    lines.append(line)
    return lines


# ─── File-Level Encryption ─────────────────────────────────────────────────

def encrypt_file_to(filepath: str, dest_path: str) -> Optional[str]:
    """
    Encrypt a file and write the ciphertext to dest_path.
    The original file is NOT deleted (caller manages lifecycle).

    Returns:
        The dest_path on success, None on failure.
    """
    if not os.path.exists(filepath):
        return None

    try:
        fernet = get_fernet()
        with open(filepath, "rb") as f:
            data = f.read()
        encrypted = fernet.encrypt(data)
        with open(dest_path, "wb") as f:
            f.write(encrypted)
        return dest_path
    except Exception as e:
        print(f"[SECURITY] File encryption failed: {e}")
        return None


def decrypt_file_to(enc_filepath: str, dest_path: str) -> Optional[str]:
    """
    Decrypt an encrypted file and write plaintext to dest_path.

    Returns:
        The dest_path on success, None on failure.
    """
    if not os.path.exists(enc_filepath):
        return None

    try:
        fernet = get_fernet()
        with open(enc_filepath, "rb") as f:
            encrypted = f.read()
        decrypted = fernet.decrypt(encrypted)
        with open(dest_path, "wb") as f:
            f.write(decrypted)
        return dest_path
    except Exception as e:
        print(f"[SECURITY] File decryption failed: {e}")
        return None
