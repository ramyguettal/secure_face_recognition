"""
face_recognition_module.py — Step 2: Face embedding match against encrypted database.

Uses the face_recognition library (dlib-based) to generate 128-d embeddings.
Embeddings are encrypted at rest with Fernet (AES-128). Includes rate limiting
with exponential backoff and lockout.
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
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer
from sqlalchemy.orm import declarative_base, Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

Base = declarative_base()


class EnrolledUser(Base):
    """SQLAlchemy model for the enrolled user database."""
    __tablename__ = "enrolled_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    encrypted_embedding = Column(LargeBinary, nullable=False)
    encrypted_voiceprint = Column(LargeBinary, nullable=True)  # For Step 6 speaker verification


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


def _get_fernet() -> Fernet:
    """Retrieve the Fernet encryption key from file or environment variable."""
    key = None

    # 1. Try persistent key file first
    key_file = os.path.join(os.path.dirname(__file__), "..", config.FACE_DB_KEY_FILE)
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key = f.read().strip()
        if key:
            # Also set env var so rest of session works
            os.environ[config.FACE_DB_KEY_ENV] = key

    # 2. Fall back to environment variable
    if not key:
        key = os.environ.get(config.FACE_DB_KEY_ENV)

    if not key:
        raise EnvironmentError(
            f"[ERROR] Environment variable '{config.FACE_DB_KEY_ENV}' is not set.\n"
            f"Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            f"Then set it: set {config.FACE_DB_KEY_ENV}=<your_key>"
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def _get_engine():
    """Create SQLAlchemy engine for the face database."""
    db_path = os.path.join(os.path.dirname(__file__), "..", config.FACE_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """Encrypt a face embedding array using Fernet."""
    fernet = _get_fernet()
    embedding_bytes = embedding.tobytes()
    return fernet.encrypt(embedding_bytes)


def decrypt_embedding(encrypted: bytes) -> np.ndarray:
    """Decrypt an encrypted face embedding back to a numpy array."""
    fernet = _get_fernet()
    decrypted_bytes = fernet.decrypt(encrypted)
    return np.frombuffer(decrypted_bytes, dtype=np.float64)


def enroll_user(user_id: str, display_name: str, face_images: List[np.ndarray],
                voiceprint: Optional[np.ndarray] = None) -> bool:
    """
    Enroll a new user by averaging embeddings from multiple face images.

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
    encrypted = encrypt_embedding(avg_embedding)

    encrypted_vp = None
    if voiceprint is not None:
        fernet = _get_fernet()
        if isinstance(voiceprint, np.ndarray):
            vp_bytes = voiceprint.astype(np.float32).tobytes()
        elif isinstance(voiceprint, bytes):
            vp_bytes = voiceprint
        else:
            vp_bytes = bytes(voiceprint)
        encrypted_vp = fernet.encrypt(vp_bytes)
        print(f"[ENROLL] Voiceprint stored: {len(vp_bytes)} bytes")

    engine = _get_engine()
    with Session(engine) as session:
        existing = session.query(EnrolledUser).filter_by(user_id=user_id).first()
        if existing:
            existing.encrypted_embedding = encrypted
            existing.display_name = display_name
            if encrypted_vp:
                existing.encrypted_voiceprint = encrypted_vp
            print(f"[INFO] Updated existing enrollment for '{user_id}'.")
        else:
            user = EnrolledUser(
                user_id=user_id,
                display_name=display_name,
                encrypted_embedding=encrypted,
                encrypted_voiceprint=encrypted_vp
            )
            session.add(user)
            print(f"[INFO] Enrolled new user '{user_id}'.")
        session.commit()
    return True


def match_face(live_embedding: np.ndarray) -> Optional[Tuple[str, str, float]]:
    """
    Match a live face embedding against all enrolled users.

    Args:
        live_embedding: 128-d face embedding from the live capture.

    Returns:
        (user_id, display_name, distance) of the best match, or None if no match
        below the threshold.
    """
    engine = _get_engine()
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
                    best_match = (user.user_id, user.display_name, distance)
            except Exception as e:
                print(f"[WARN] Could not decrypt embedding for user '{user.user_id}': {e}")
                continue

    if best_match and best_match[2] <= config.FACE_MATCH_THRESHOLD:
        return best_match

    return None


def get_live_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extract face embedding from a live camera frame."""
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
    return None

def check_user_exists(user_id: str) -> bool:
    """Check if a given user_id is already enrolled."""
    engine = _get_engine()
    with Session(engine) as session:
        return session.query(EnrolledUser).filter_by(user_id=user_id).first() is not None



def get_user_voiceprint(user_id: str) -> Optional[np.ndarray]:
    """Retrieve and decrypt a user's stored voiceprint (256-dim float32 embedding)."""
    engine = _get_engine()
    with Session(engine) as session:
        user = session.query(EnrolledUser).filter_by(user_id=user_id).first()
        if user and user.encrypted_voiceprint:
            fernet = _get_fernet()
            decrypted = fernet.decrypt(user.encrypted_voiceprint)
            vp = np.frombuffer(decrypted, dtype=np.float32).copy()
            print(f"[VOICE] Retrieved voiceprint for '{user_id}': shape={vp.shape}")
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
