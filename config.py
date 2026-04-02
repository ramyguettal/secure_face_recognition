"""
config.py — Central configuration for the Secure Face Auth system.
All thresholds, timeouts, challenge pools, and environment variable names live here.
"""

import os
from typing import List

# ─── Environment Variables ───────────────────────────────────────────────────
FACE_DB_KEY_ENV: str = "FACE_DB_KEY"

# ─── Face Recognition (Step 2) ──────────────────────────────────────────────
FACE_MATCH_THRESHOLD: float = 0.5          # Euclidean distance threshold (lower = stricter)
FACE_ENCODING_MODEL: str = "small"         # "small" or "large" (128-d embeddings)
ENROLLMENT_CAPTURE_COUNT: int = 5          # Number of images to capture during enrollment

# ─── Anti-Spoofing / PAD (NEW-1) ────────────────────────────────────────────
PAD_REAL_THRESHOLD: float = 0.8            # Minimum "real" probability to accept
BRIGHTNESS_UNIFORMITY_THRESHOLD: float = 0.15  # Screen glow detection threshold
LAPLACIAN_VARIANCE_THRESHOLD: float = 50.0     # Texture sharpness minimum (real faces have more texture)

# ─── Rate Limiting (Decision Gate) ──────────────────────────────────────────
MAX_FAILED_BEFORE_BACKOFF: int = 5         # Failed attempts before exponential backoff kicks in
MAX_FAILED_BEFORE_LOCKOUT: int = 10        # Failed attempts before full lockout
BACKOFF_CAP_SECONDS: float = 300.0         # Maximum backoff = 5 minutes
LOCKOUT_DURATION_SECONDS: float = 900.0    # Lockout period = 15 minutes

# ─── Voice Greeting (Step 3) ────────────────────────────────────────────────
GREETING_TEXT: str = "Hello! Please complete verification."

# ─── Head Movement Challenge (Step 4) ───────────────────────────────────────
HEAD_CHALLENGE_DIRECTIONS: List[str] = ["turn left", "turn right", "tilt up", "tilt down"]
HEAD_POSE_YAW_THRESHOLD: float = 8.0       # Degrees of yaw rotation required
HEAD_POSE_PITCH_THRESHOLD: float = 6.0     # Degrees of pitch rotation required
HEAD_CHALLENGE_TIMEOUT: float = 10.0       # Seconds to complete the challenge
HEAD_MIN_INTERMEDIATE_FRAMES: int = 2      # Minimum frames showing continuous motion

# ─── Blink Detection (Step 5) ───────────────────────────────────────────────
EAR_BLINK_THRESHOLD: float = 0.2           # Eye Aspect Ratio below this = eye closed
EAR_CONSEC_FRAMES: int = 2                 # Consecutive frames below threshold to detect blink
EAR_VARIANCE_EPSILON: float = 0.005        # Reject if EAR variance is suspiciously low
BLINK_RANDOM_WINDOW_MIN: float = 2.0       # Min seconds before blink prompt
BLINK_RANDOM_WINDOW_MAX: float = 4.0       # Max seconds after blink prompt
BLINK_CHALLENGE_TIMEOUT: float = 6.0       # Total timeout for blink challenge

# ─── Voice Challenge (Step 6) ───────────────────────────────────────────────
VOICE_CHALLENGE_PHRASES: List[str] = [
    "say the word: apple",
    "say the word: mountain",
    "say the word: river",
    "say the word: sunset",
    "say the word: guitar",
    "say the color: blue",
    "say the color: green",
    "say the color: purple",
    "say the color: orange",
    "say the color: silver",
    "count to three",
    "count to five",
    "say the number: seven",
    "say the number: twenty",
    "say the phrase: good morning",
    "say the phrase: open the door",
    "say the phrase: hello world",
    "say the phrase: start the engine",
    "say the phrase: sunny day",
    "say the phrase: winter sky",
]
VOICE_FUZZY_MATCH_RATIO: float = 0.85      # difflib.SequenceMatcher ratio threshold
VOICE_LISTEN_TIMEOUT: float = 5.0          # Seconds to wait for speech
VOICE_PHRASE_TIMEOUT: float = 10.0         # Max duration of the spoken phrase

# ─── Session Token (NEW-2) ──────────────────────────────────────────────────
SESSION_TOKEN_BYTES: int = 32              # secrets.token_hex(32) → 64-char hex string
SESSION_EXPIRY_MINUTES: int = 30           # Token validity duration

# ─── Audit Logging ──────────────────────────────────────────────────────────
AUDIT_LOG_FILE: str = "data/audit.log"
AUDIT_LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
AUDIT_LOG_BACKUP_COUNT: int = 5
SECURITY_ALERT_WINDOW_MINUTES: int = 10    # Time window for repeated failure alerts
SECURITY_ALERT_THRESHOLD: int = 3          # Failures from same user to trigger alert

# ─── Database ───────────────────────────────────────────────────────────────
FACE_DB_PATH: str = "data/face_db.db"
FACE_DB_KEY_FILE: str = "data/.face_key"  # Persistent key file (auto-created)

# ─── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0                      # Default camera index
CAMERA_FRAME_WIDTH: int = 640
CAMERA_FRAME_HEIGHT: int = 480
FACE_DETECTION_CONFIDENCE: float = 0.7     # MediaPipe face detection min confidence
