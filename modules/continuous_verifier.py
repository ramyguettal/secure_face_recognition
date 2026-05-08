"""
continuous_verifier.py — Background swap-attack prevention.

Runs a daemon thread during pipeline Steps 3-7 that periodically
re-extracts a face embedding from the live camera and compares it
against the embedding captured in Step 2 (matched user).

If the face is missing or belongs to a different person for
CONSECUTIVE_MISSES ticks in a row the verifier raises a violation flag
which the pipeline checks at each step boundary.

Design choices:
  - Uses the shared _dlib_lock so it never races with other dlib calls.
  - Non-blocking: if it cannot acquire the lock within 0.2 s it simply
    skips that tick (counts as no reading, not a miss).
  - Threshold is configurable — defaults to config.FACE_MATCH_THRESHOLD
    with a small tolerance multiplier to handle pose variation during
    challenges.
"""

import sys
import os
import time
import threading
import cv2
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ContinuousVerifier:
    """
    Periodically re-checks the live face against the enrolled embedding.

    Usage::

        verifier = ContinuousVerifier(enrolled_embedding, camera)
        verifier.start()

        # ... run pipeline steps ...

        if verifier.is_violated():
            abort(verifier.violation_reason())

        verifier.stop()

    Thread-safe: all state is protected by an internal lock.
    """

    # Number of consecutive failed ticks before a violation is declared.
    # 1 tick ≈ sample_interval seconds.
    CONSECUTIVE_MISSES_THRESHOLD = 2

    def __init__(
        self,
        enrolled_embedding: np.ndarray,
        camera,
        sample_interval: float = 1.5,
        threshold_multiplier: float = 1.15,
    ) -> None:
        """
        Args:
            enrolled_embedding: 128-d face embedding captured during Step 2.
            camera: CameraCapture instance (shared with the pipeline thread).
            sample_interval: Seconds between verification ticks.
            threshold_multiplier: Multiplied against config.FACE_MATCH_THRESHOLD
                to give a slightly looser threshold for re-verification
                (head pose during challenges causes natural variation).
        """
        self._enrolled_embedding = enrolled_embedding.copy()
        self._camera = camera
        self._sample_interval = sample_interval
        self._threshold = config.FACE_MATCH_THRESHOLD * threshold_multiplier

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._violated = False
        self._violation_reason = ""
        self._consecutive_misses = 0
        self._ticks = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background verification thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ContinuousVerifier"
        )
        self._thread.start()
        print("[CV] Continuous verifier started (interval=%.1fs, threshold=%.3f)"
              % (self._sample_interval, self._threshold))

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        print(f"[CV] Continuous verifier stopped after {self._ticks} ticks.")

    def is_violated(self) -> bool:
        """Return True if a swap/identity violation was detected."""
        with self._lock:
            return self._violated

    def violation_reason(self) -> str:
        """Return a human-readable description of the violation."""
        with self._lock:
            return self._violation_reason

    # ── Background Thread ───────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: sample the camera at regular intervals."""
        while not self._stop_event.is_set():
            tick_start = time.time()

            self._tick()

            # Sleep for the remainder of the interval
            elapsed = time.time() - tick_start
            sleep_remaining = self._sample_interval - elapsed
            if sleep_remaining > 0:
                # Use short sleeps so the stop event is checked promptly
                deadline = time.time() + sleep_remaining
                while time.time() < deadline and not self._stop_event.is_set():
                    time.sleep(0.1)

    def _tick(self) -> None:
        """One verification sample."""
        if self._camera is None or not getattr(self._camera, "is_active", False):
            return  # Camera not ready — skip silently

        frame = self._camera.read_frame()
        if frame is None:
            return  # No frame available — skip

        embedding = self._extract_embedding(frame)

        with self._lock:
            self._ticks += 1

            if embedding is None:
                # Face not visible in this frame
                self._consecutive_misses += 1
                print(
                    f"[CV] Tick {self._ticks}: face not detected "
                    f"(miss {self._consecutive_misses}/{self.CONSECUTIVE_MISSES_THRESHOLD + 1})"
                )
                # Allow one extra miss for faces that briefly exit frame
                if self._consecutive_misses > self.CONSECUTIVE_MISSES_THRESHOLD:
                    self._violated = True
                    self._violation_reason = (
                        f"Face disappeared from camera for "
                        f"{self._consecutive_misses} consecutive checks "
                        f"(possible swap or camera obstruction)."
                    )
            else:
                distance = float(np.linalg.norm(embedding - self._enrolled_embedding))
                print(
                    f"[CV] Tick {self._ticks}: dist={distance:.4f} "
                    f"threshold={self._threshold:.4f}"
                )

                if distance > self._threshold:
                    self._consecutive_misses += 1
                    print(
                        f"[CV] MISMATCH — "
                        f"consecutive misses: {self._consecutive_misses}"
                    )
                    if self._consecutive_misses >= self.CONSECUTIVE_MISSES_THRESHOLD:
                        self._violated = True
                        self._violation_reason = (
                            f"Face identity mismatch detected after "
                            f"{self._consecutive_misses} consecutive checks "
                            f"(dist={distance:.3f} > threshold={self._threshold:.3f}). "
                            f"Possible swap attack."
                        )
                else:
                    # Reset miss counter on a good reading
                    self._consecutive_misses = 0

    def _extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 128-d face embedding from a frame.

        Uses a downscaled frame and HOG detection for speed.
        Acquires _dlib_lock non-blockingly; returns None if busy.
        """
        try:
            import face_recognition
            from modules.camera import _dlib_lock

            # Downscale for speed
            h, w = frame.shape[:2]
            scale = 0.5 if max(h, w) > 320 else 1.0
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Non-blocking lock attempt — skip tick if dlib is busy
            acquired = _dlib_lock.acquire(timeout=0.2)
            if not acquired:
                return None

            try:
                locations = face_recognition.face_locations(rgb, model="hog")
                if not locations:
                    return None
                encodings = face_recognition.face_encodings(
                    rgb, known_face_locations=locations, model="small"
                )
            finally:
                _dlib_lock.release()

            if not encodings:
                return None

            return encodings[0].astype(np.float64)

        except Exception as e:
            print(f"[CV] Embedding extraction error: {e}")
            return None
