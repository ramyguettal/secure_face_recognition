"""
session.py — NEW Layer 2: Session token generation + encrypted audit logging.

Generates time-limited session tokens after successful authentication.
Maintains encrypted, rotating JSON-lines audit log of all auth events.
Triggers SECURITY_ALERT on repeated failures from the same user.

SECURITY:
  - Audit log entries are individually encrypted with Fernet (AES-128)
  - Session tokens are stored with encrypted user_id
  - Failure records use HMAC-hashed user_id for privacy
  - Legacy plaintext audit logs are replaced by encrypted format
"""

import os
import sys
import json
import time
import secrets
import logging
import socket
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from sqlalchemy import create_engine, Column, String, Float, Integer, LargeBinary
from sqlalchemy.orm import declarative_base, Session as DBSession

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from modules.secure_storage import (
    EncryptedLogHandler, encrypt_field, decrypt_field,
    hmac_hash, secure_delete
)

Base = declarative_base()


# ─── Session Token Model ────────────────────────────────────────────────────

class SessionToken(Base):
    """SQLAlchemy model for active session tokens."""
    __tablename__ = "session_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token = Column(String(128), unique=True, nullable=False)
    user_id_hash = Column(String, nullable=True)      # HMAC hash for lookup
    encrypted_user_id = Column(LargeBinary, nullable=True)  # Fernet-encrypted
    user_id = Column(String, nullable=True)            # Legacy (cleared after migration)
    issued_at = Column(Float, nullable=False)
    expires_at = Column(Float, nullable=False)
    encrypted_ip = Column(LargeBinary, nullable=True)  # Fernet-encrypted IP
    ip_address = Column(String, nullable=True)         # Legacy (cleared after migration)


# ─── Failure Tracking Model ─────────────────────────────────────────────────

class FailureRecord(Base):
    """Tracks recent auth failures for security alerting."""
    __tablename__ = "failure_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_hash = Column(String, nullable=True)       # HMAC hash for querying
    user_id = Column(String, nullable=True)            # Legacy (cleared after migration)
    timestamp = Column(Float, nullable=False)
    step_reached = Column(String, nullable=True)
    encrypted_reason = Column(LargeBinary, nullable=True)  # Fernet-encrypted
    reason = Column(String, nullable=True)             # Legacy (cleared after migration)


def _get_engine():
    """Create or retrieve the SQLAlchemy engine."""
    db_path = os.path.join(os.path.dirname(__file__), "..", config.FACE_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _get_local_ip() -> str:
    """Get the local IP address for logging."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─── Audit Logger ───────────────────────────────────────────────────────────

class AuditLogger:
    """
    Encrypted structured audit logger.

    Each log entry is individually encrypted with Fernet before being
    written to disk. This ensures that user IDs, IP addresses, and
    failure reasons are never stored in plaintext.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("secure_face_auth.audit")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Don't bubble up to root logger

        # Use encrypted log file path
        enc_log_path = os.path.join(
            os.path.dirname(__file__), "..", config.ENCRYPTED_AUDIT_LOG
        )
        os.makedirs(os.path.dirname(enc_log_path), exist_ok=True)

        # Only add handler if one doesn't already exist
        if not self.logger.handlers:
            handler = EncryptedLogHandler(
                enc_log_path,
                max_bytes=config.AUDIT_LOG_MAX_BYTES,
                backup_count=config.AUDIT_LOG_BACKUP_COUNT,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self.engine = _get_engine()

        # Clean up legacy plaintext audit log if it exists
        self._cleanup_legacy_log()

    def _cleanup_legacy_log(self):
        """Securely delete any legacy plaintext audit log."""
        legacy_path = os.path.join(
            os.path.dirname(__file__), "..", config.AUDIT_LOG_FILE
        )
        if os.path.exists(legacy_path):
            print("[SECURITY] Securely deleting legacy plaintext audit log...")
            secure_delete(legacy_path)

    def log_event(
        self,
        user_id: str,
        event_type: str,
        step_reached: str,
        outcome: str,
        reason: str = "",
    ) -> None:
        """
        Log a structured authentication event as an encrypted JSON line.

        Args:
            user_id: The user involved (or "unknown").
            event_type: e.g. "AUTH_ATTEMPT", "AUTH_SUCCESS", "AUTH_FAILURE".
            step_reached: Pipeline step where event occurred.
            outcome: "success" or "failure".
            reason: Human-readable reason string.
        """
        ip = _get_local_ip()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "event_type": event_type,
            "ip_address": ip,
            "step_reached": step_reached,
            "outcome": outcome,
            "reason": reason,
        }
        # This will be encrypted by the EncryptedLogHandler
        self.logger.info(json.dumps(entry))

        # Track failures for security alerting
        if outcome == "failure":
            self._record_failure(user_id, step_reached, reason)

    def _record_failure(self, user_id: str, step: str, reason: str) -> None:
        """Record a failure with encrypted fields and check for security alerts."""
        now = time.time()

        # Create HMAC hash for lookup and encrypt the reason
        uid_hash = hmac_hash(user_id, "failure_tracking")
        try:
            enc_reason = encrypt_field(reason)
        except Exception:
            enc_reason = None

        with DBSession(self.engine) as session:
            record = FailureRecord(
                user_id=None,  # No plaintext
                user_id_hash=uid_hash,
                timestamp=now,
                step_reached=step,
                reason=None,  # No plaintext
                encrypted_reason=enc_reason,
            )
            session.add(record)
            session.commit()

            # Check for repeated failures within the alert window
            window_start = now - (config.SECURITY_ALERT_WINDOW_MINUTES * 60)
            recent_failures = (
                session.query(FailureRecord)
                .filter(
                    FailureRecord.user_id_hash == uid_hash,
                    FailureRecord.timestamp >= window_start,
                )
                .count()
            )

            # Also count legacy format failures
            if user_id != "unknown":
                legacy_count = (
                    session.query(FailureRecord)
                    .filter(
                        FailureRecord.user_id == user_id,
                        FailureRecord.timestamp >= window_start,
                    )
                    .count()
                )
                recent_failures += legacy_count

            if recent_failures >= config.SECURITY_ALERT_THRESHOLD:
                self._trigger_security_alert(user_id, recent_failures)

    def _trigger_security_alert(self, user_id: str, failure_count: int) -> None:
        """Log a security alert when repeated failures are detected."""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "event_type": "SECURITY_ALERT",
            "ip_address": _get_local_ip(),
            "step_reached": "security_monitor",
            "outcome": "alert",
            "reason": (
                f"{failure_count} failed attempts from user "
                f"within {config.SECURITY_ALERT_WINDOW_MINUTES} minutes."
            ),
        }
        # Encrypted via the EncryptedLogHandler
        self.logger.warning(json.dumps(alert))
        print(f"\n⚠️  SECURITY ALERT: {alert['reason']}")


# ─── Session Manager ────────────────────────────────────────────────────────

class SessionManager:
    """Manages time-limited session tokens after successful auth."""

    def __init__(self) -> None:
        self.engine = _get_engine()

    def create_token(self, user_id: str) -> str:
        """
        Generate a cryptographically secure session token.
        User ID and IP are stored encrypted in the session record.

        Returns:
            The hex token string.
        """
        token = secrets.token_hex(config.SESSION_TOKEN_BYTES)
        now = time.time()
        expires = now + (config.SESSION_EXPIRY_MINUTES * 60)

        # Encrypt sensitive fields
        uid_hash = hmac_hash(user_id, "session")
        enc_uid = encrypt_field(user_id)
        enc_ip = encrypt_field(_get_local_ip())

        with DBSession(self.engine) as session:
            token_record = SessionToken(
                token=token,
                user_id=None,  # No plaintext
                user_id_hash=uid_hash,
                encrypted_user_id=enc_uid,
                issued_at=now,
                expires_at=expires,
                ip_address=None,  # No plaintext
                encrypted_ip=enc_ip,
            )
            session.add(token_record)
            session.commit()

        return token

    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate a session token and return the associated user_id.

        Returns:
            user_id if token is valid and not expired, None otherwise.
        """
        now = time.time()
        with DBSession(self.engine) as session:
            record = (
                session.query(SessionToken)
                .filter(SessionToken.token == token)
                .first()
            )
            if record is None:
                return None
            if record.expires_at < now:
                # Token expired — clean it up
                session.delete(record)
                session.commit()
                return None

            # Decrypt user_id from encrypted field
            if record.encrypted_user_id:
                return decrypt_field(record.encrypted_user_id)
            return record.user_id  # Legacy fallback

    def invalidate_token(self, token: str) -> bool:
        """Manually invalidate (revoke) a session token."""
        with DBSession(self.engine) as session:
            record = (
                session.query(SessionToken)
                .filter(SessionToken.token == token)
                .first()
            )
            if record:
                session.delete(record)
                session.commit()
                return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired tokens. Returns count of removed tokens."""
        now = time.time()
        with DBSession(self.engine) as session:
            expired = (
                session.query(SessionToken)
                .filter(SessionToken.expires_at < now)
                .all()
            )
            count = len(expired)
            for t in expired:
                session.delete(t)
            session.commit()
        return count


if __name__ == "__main__":
    print("Session & Audit Module — Test")

    # Test audit logging
    audit = AuditLogger()
    audit.log_event("test_user", "AUTH_ATTEMPT", "face_recognition", "failure", "Test failure")
    audit.log_event("test_user", "AUTH_ATTEMPT", "blink_detection", "failure", "Test failure 2")
    audit.log_event("test_user", "AUTH_ATTEMPT", "voice_challenge", "failure", "Test failure 3")
    print("  Audit events logged (encrypted).")

    # Test session tokens
    sm = SessionManager()
    token = sm.create_token("test_user")
    print(f"  Created token: {token[:16]}...")
    valid_user = sm.validate_token(token)
    print(f"  Validated: user_id = {valid_user}")
    sm.invalidate_token(token)
    print(f"  Invalidated. Valid now: {sm.validate_token(token)}")
