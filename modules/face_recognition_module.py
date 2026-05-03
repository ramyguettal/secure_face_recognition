"""
face_recognition_module.py — Step 2: Face embedding match against encrypted database.

Uses the face_recognition library (dlib-based) to generate 128-d embeddings.
Embeddings are encrypted at rest with Fernet (AES-128). Includes rate limiting
with exponential backoff and lockout.

SECURITY:
  - All user data (user_id, display_name, embedding, voiceprint) is encrypted at rest.
  - User lookup uses HMAC-SHA256 hash — no plaintext user_id stored in the database.
  - Auto-migrates any legacy plaintext data on first run.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import face_recognition
from cryptography.fernet import Fernet
from typing import Optional, Tuple, Dict, List
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer, inspect
from sqlalchemy.orm import declarative_base, Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from modules.secure_storage import (
    get_fernet, encrypt_field, decrypt_field, hmac_hash
)

Base = declarative_base()


class EnrolledUser(Base):
    """
    SQLAlchemy model for the enrolled user database.

    All PII fields are encrypted:
      - user_id_hash: HMAC-SHA256 of user_id (for lookups, irreversible)
      - encrypted_user_id: Fernet-encrypted user_id
      - encrypted_display_name: Fernet-encrypted display name
      - encrypted_embedding: Fernet-encrypted 128-d face embedding
      - encrypted_voiceprint: Fernet-encrypted voice embedding (optional)
    """
    __tablename__ = "enrolled_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Legacy columns (kept for migration, will be NULLed after migration)
    user_id = Column(String, nullable=True)
    display_name = Column(String, nullable=True)
    # New secure columns
    user_id_hash = Column(String, nullable=True, index=True)
    encrypted_user_id = Column(LargeBinary, nullable=True)
    encrypted_display_name = Column(LargeBinary, nullable=True)
    # Already encrypted
    encrypted_embedding = Column(LargeBinary, nullable=False)
    encrypted_voiceprint = Column(LargeBinary, nullable=True)


class RateLimiter:
    """
    Tracks failed authentication attempts and enforces exponential backoff
    and lockout policies.
    """

    def __init__(self) -> None:
        # {identifier: {"count": int, "last_failure": float, "lockout_until": float}}
        self._attempts: Dict[str, Dict] = {}

    def check_allowed(self, identifier: str = "default") -> Tuple[bool, str]:
        """
        Check if an authentication attempt is allowed for the given identifier.

        Returns:
            (allowed: bool, reason: str)
        """
        if identifier not in self._attempts:
            return True, "OK"

        record = self._attempts[identifier]
        now = time.time()

        # Check lockout
        if record.get("lockout_until", 0) > now:
            remaining = int(record["lockout_until"] - now)
            return False, f"Account locked out. Try again in {remaining}s."

        # Check exponential backoff
        count = record.get("count", 0)
        if count >= config.MAX_FAILED_BEFORE_BACKOFF:
            backoff = min(2 ** (count - config.MAX_FAILED_BEFORE_BACKOFF + 1),
                          config.BACKOFF_CAP_SECONDS)
            elapsed = now - record.get("last_failure", 0)
            if elapsed < backoff:
                wait = int(backoff - elapsed)
                return False, f"Too many failed attempts. Wait {wait}s before retrying."

        return True, "OK"

    def record_failure(self, identifier: str = "default") -> str:
        """Record a failed attempt and return the current status message."""
        now = time.time()
        if identifier not in self._attempts:
            self._attempts[identifier] = {"count": 0, "last_failure": now}

        record = self._attempts[identifier]
        record["count"] += 1
        record["last_failure"] = now

        count = record["count"]

        if count >= config.MAX_FAILED_BEFORE_LOCKOUT:
            record["lockout_until"] = now + config.LOCKOUT_DURATION_SECONDS
            return f"LOCKOUT: {count} failed attempts. Locked for {int(config.LOCKOUT_DURATION_SECONDS)}s."

        if count >= config.MAX_FAILED_BEFORE_BACKOFF:
            backoff = min(2 ** (count - config.MAX_FAILED_BEFORE_BACKOFF + 1),
                          config.BACKOFF_CAP_SECONDS)
            return f"BACKOFF: {count} failed attempts. Next attempt in {int(backoff)}s."

        return f"Failed attempt #{count}."

    def record_success(self, identifier: str = "default") -> None:
        """Reset counters on successful authentication."""
        if identifier in self._attempts:
            del self._attempts[identifier]


# ─── Backward-compatible alias ──────────────────────────────────────────────
def _get_fernet() -> Fernet:
    """Retrieve the Fernet encryption key. Delegates to secure_storage."""
    return get_fernet()


def _get_engine():
    """Create SQLAlchemy engine for the face database."""
    db_path = os.path.join(os.path.dirname(__file__), "..", config.FACE_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


# ─── Database Migration ────────────────────────────────────────────────────

def _migrate_if_needed(engine) -> None:
    """
    Auto-migrate legacy plaintext user data to encrypted format.

    Checks if any rows have plaintext user_id set (legacy format).
    If found, encrypts user_id and display_name, creates HMAC hash,
    and NULLs out the plaintext columns.
    """
    try:
        with Session(engine) as session:
            # Find rows with legacy plaintext data
            legacy_users = (
                session.query(EnrolledUser)
                .filter(
                    EnrolledUser.user_id.isnot(None),
                    EnrolledUser.user_id != "",
                    # Only migrate rows that haven't been migrated yet
                    EnrolledUser.user_id_hash.is_(None)
                )
                .all()
            )

            if not legacy_users:
                return

            print(f"[MIGRATION] Encrypting {len(legacy_users)} legacy user record(s)...")

            for user in legacy_users:
                uid = user.user_id
                dname = user.display_name or uid

                # Create encrypted versions
                user.user_id_hash = hmac_hash(uid, "user_id")
                user.encrypted_user_id = encrypt_field(uid)
                user.encrypted_display_name = encrypt_field(dname)

                # Clear plaintext columns
                user.user_id = None
                user.display_name = None

                print(f"[MIGRATION]   Encrypted user: {uid[:1]}*** -> HMAC:{user.user_id_hash[:12]}...")

            session.commit()
            print(f"[MIGRATION] ✓ All legacy records encrypted successfully.")

    except Exception as e:
        print(f"[MIGRATION] Warning: Could not migrate legacy data: {e}")


# ─── Encryption Helpers ────────────────────────────────────────────────────

def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """Encrypt a face embedding array using Fernet."""
    fernet = get_fernet()
    embedding_bytes = embedding.tobytes()
    return fernet.encrypt(embedding_bytes)


def decrypt_embedding(encrypted: bytes) -> np.ndarray:
    """Decrypt an encrypted face embedding back to a numpy array."""
    fernet = get_fernet()
    decrypted_bytes = fernet.decrypt(encrypted)
    return np.frombuffer(decrypted_bytes, dtype=np.float64)


# ─── User Enrollment ──────────────────────────────────────────────────────

def enroll_user(user_id: str, display_name: str, face_images: List[np.ndarray],
                voiceprint: Optional[np.ndarray] = None) -> bool:
    """
    Enroll a new user by averaging embeddings from multiple face images.

    All data is encrypted before storage:
      - user_id → HMAC hash (for lookup) + Fernet encrypted (for retrieval)
      - display_name → Fernet encrypted
      - face embedding → Fernet encrypted (averaged from multiple images)
      - voiceprint → Fernet encrypted (optional)

    Args:
        user_id: Unique identifier for the user.
        display_name: Display name for the user.
        face_images: List of BGR images containing the user's face.
        voiceprint: Optional voiceprint embedding for speaker verification.

    Returns:
        True if enrollment successful.
    """
    embeddings = []
    for idx, img in enumerate(face_images):
        try:
            # Downscale large frames to prevent dlib crashes
            h, w = img.shape[:2]
            max_dim = 640
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                print(f"[ENROLL] Image {idx+1}: resized from {w}x{h} to {img.shape[1]}x{img.shape[0]}")

            rgb = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB, force C-contiguous
            print(f"[ENROLL] Image {idx+1}/{len(face_images)}: extracting encoding...")
            encs = face_recognition.face_encodings(rgb, model=config.FACE_ENCODING_MODEL)
            if encs:
                embeddings.append(encs[0])
                print(f"[ENROLL] Image {idx+1}: encoding extracted OK")
            else:
                print(f"[ENROLL] Image {idx+1}: no face found in image, skipping")
        except Exception as e:
            print(f"[ENROLL] Image {idx+1}: encoding failed: {e}, skipping")
            continue

    if len(embeddings) < 3:
        print(f"[ERROR] Only {len(embeddings)} valid face encodings captured. Need at least 3.")
        return False

    # Average the embeddings for robustness
    avg_embedding = np.mean(embeddings, axis=0)
    encrypted_emb = encrypt_embedding(avg_embedding)

    # Encrypt voiceprint if provided
    encrypted_vp = None
    if voiceprint is not None:
        fernet = get_fernet()
        if isinstance(voiceprint, np.ndarray):
            vp_bytes = voiceprint.astype(np.float32).tobytes()
        elif isinstance(voiceprint, bytes):
            vp_bytes = voiceprint
        else:
            vp_bytes = bytes(voiceprint)
        encrypted_vp = fernet.encrypt(vp_bytes)
        print(f"[ENROLL] Voiceprint stored: {len(vp_bytes)} bytes (encrypted)")

    # Encrypt user metadata
    uid_hash = hmac_hash(user_id, "user_id")
    encrypted_uid = encrypt_field(user_id)
    encrypted_dname = encrypt_field(display_name)

    engine = _get_engine()
    _migrate_if_needed(engine)

    with Session(engine) as session:
        # Look up by HMAC hash (no plaintext comparison)
        existing = session.query(EnrolledUser).filter_by(user_id_hash=uid_hash).first()

        # Also check legacy plaintext for backward compatibility
        if existing is None:
            existing = session.query(EnrolledUser).filter_by(user_id=user_id).first()

        if existing:
            existing.user_id_hash = uid_hash
            existing.encrypted_user_id = encrypted_uid
            existing.encrypted_display_name = encrypted_dname
            existing.encrypted_embedding = encrypted_emb
            existing.user_id = None  # Clear any legacy plaintext
            existing.display_name = None
            if encrypted_vp:
                existing.encrypted_voiceprint = encrypted_vp
            print(f"[INFO] Updated existing enrollment for user (HMAC:{uid_hash[:12]}...).")
        else:
            user = EnrolledUser(
                user_id=None,  # No plaintext storage
                display_name=None,
                user_id_hash=uid_hash,
                encrypted_user_id=encrypted_uid,
                encrypted_display_name=encrypted_dname,
                encrypted_embedding=encrypted_emb,
                encrypted_voiceprint=encrypted_vp,
            )
            session.add(user)
            print(f"[INFO] Enrolled new user (HMAC:{uid_hash[:12]}...).")
        session.commit()
    return True


# ─── Face Matching ─────────────────────────────────────────────────────────

def match_face(live_embedding: np.ndarray) -> Optional[Tuple[str, str, float]]:
    """
    Match a live face embedding against all enrolled users.

    Decrypts user data only in memory — never persisted as plaintext.

    Args:
        live_embedding: 128-d face embedding from the live capture.

    Returns:
        (user_id, display_name, distance) of the best match, or None if no match
        below the threshold.
    """
    engine = _get_engine()
    _migrate_if_needed(engine)
    best_match: Optional[Tuple[str, str, float]] = None
    best_distance = float("inf")

    with Session(engine) as session:
        users = session.query(EnrolledUser).all()
        if not users:
            print("[WARN] No enrolled users in database.")
            return None

        for user in users:
            try:
                stored_embedding = decrypt_embedding(user.encrypted_embedding)
                distance = np.linalg.norm(live_embedding - stored_embedding)

                if distance < best_distance:
                    best_distance = distance

                    # Decrypt user metadata in memory only
                    if user.encrypted_user_id:
                        uid = decrypt_field(user.encrypted_user_id)
                        dname = decrypt_field(user.encrypted_display_name)
                    else:
                        # Legacy fallback
                        uid = user.user_id or "unknown"
                        dname = user.display_name or uid

                    best_match = (uid, dname, distance)
            except Exception as e:
                print(f"[WARN] Could not process user record (id={user.id}): {e}")
                continue

    if best_match and best_match[2] <= config.FACE_MATCH_THRESHOLD:
        return best_match

    return None


def get_live_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extract face embedding from a live camera frame."""
    try:
        # Downscale large frames to prevent dlib crashes
        h, w = frame.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR to RGB, force C-contiguous
        encodings = face_recognition.face_encodings(rgb, model=config.FACE_ENCODING_MODEL)
        if encodings:
            return encodings[0]
    except Exception as e:
        print(f"[ERROR] get_live_embedding failed: {e}")
    return None


def check_user_exists(user_id: str) -> bool:
    """Check if a given user_id is already enrolled (using HMAC lookup)."""
    engine = _get_engine()
    _migrate_if_needed(engine)
    uid_hash = hmac_hash(user_id, "user_id")
    with Session(engine) as session:
        # Check new encrypted format
        found = session.query(EnrolledUser).filter_by(user_id_hash=uid_hash).first()
        if found:
            return True
        # Check legacy plaintext format
        found = session.query(EnrolledUser).filter_by(user_id=user_id).first()
        return found is not None


def get_user_voiceprint(user_id: str) -> Optional[np.ndarray]:
    """Retrieve and decrypt a user's stored voiceprint (256-dim float32 embedding)."""
    engine = _get_engine()
    _migrate_if_needed(engine)
    uid_hash = hmac_hash(user_id, "user_id")

    with Session(engine) as session:
        # Try new HMAC lookup first
        user = session.query(EnrolledUser).filter_by(user_id_hash=uid_hash).first()
        # Fallback to legacy
        if user is None:
            user = session.query(EnrolledUser).filter_by(user_id=user_id).first()

        if user and user.encrypted_voiceprint:
            fernet = get_fernet()
            decrypted = fernet.decrypt(user.encrypted_voiceprint)
            vp = np.frombuffer(decrypted, dtype=np.float32).copy()
            print(f"[VOICE] Retrieved voiceprint: shape={vp.shape}")
            return vp
    return None


# Shared rate limiter instance
rate_limiter = RateLimiter()


if __name__ == "__main__":
    print("Face Recognition Module — Test")
    print(f"  DB Path: {config.FACE_DB_PATH}")
    print(f"  Match Threshold: {config.FACE_MATCH_THRESHOLD}")
    try:
        _get_fernet()
        print(f"  Encryption key: OK (loaded from ${config.FACE_DB_KEY_ENV})")
    except EnvironmentError as e:
        print(f"  Encryption key: MISSING — {e}")
