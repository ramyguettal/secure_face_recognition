"""
voice_challenge.py — Step 6: AI-driven voice challenge with STT + speaker verification.

Flow:
  1. Bot says the phrase to repeat
  2. GUI shows a "GET READY" countdown (3-2-1)
  3. GUI shows a "RECORDING" countdown with a big timer
  4. Transcribe and check the phrase
  5. Compare voiceprint against stored DB embedding (resemblyzer)

SECURITY:
  - Voice recordings are stored in the secure temp directory
  - All WAV files are securely wiped after transcription + voiceprint comparison
"""

import os
import sys
import time
import random
import threading
import numpy as np
from typing import Tuple, Optional, Callable

sys.path.insert(0, os.path.dirname(__file__))
import config
from modules.secure_storage import secure_delete


class VoiceChallenge:
    def __init__(self):
        self._resemblyzer_available = False
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder(device="cpu")
            self._resemblyzer_available = True
            print("[VOICE] Speaker verification: ENABLED (resemblyzer loaded, device=cpu)")
        except ImportError:
            print("[VOICE] Speaker verification: DISABLED (resemblyzer not installed)")

    def _compare_voiceprint(self, wav_file: str, user_id: str) -> Tuple[bool, float]:
        """Compare live voiceprint against stored one. Returns (match, similarity)."""
        if not self._resemblyzer_available:
            return True, 1.0

        try:
            from resemblyzer import preprocess_wav
            from modules.face_recognition_module import get_user_voiceprint

            stored_vp = get_user_voiceprint(user_id)
            if stored_vp is None:
                print("[VOICE] No stored voiceprint — skipping speaker verification.")
                return True, 1.0

            wav = preprocess_wav(wav_file)
            live_vp = self._encoder.embed_utterance(wav).astype(np.float32)

            similarity = float(np.dot(stored_vp, live_vp) / (
                np.linalg.norm(stored_vp) * np.linalg.norm(live_vp) + 1e-8))

            print(f"[VOICE] Speaker similarity: {similarity:.3f}")
            return similarity >= 0.55, similarity

        except Exception as e:
            print(f"[VOICE] Speaker verification error: {e}")
            return True, 1.0

    def run_challenge(self, camera, user_id: str,
                      update_ui_callback: Optional[Callable[[str], None]] = None,
                      update_frame_callback: Optional[Callable] = None
                      ) -> Tuple[bool, str]:
        """
        Full voice challenge flow with visual countdown timer on UI and camera frame.

        Args:
            camera: CameraCapture instance to read frames from.
            user_id: Enrolled user ID for voiceprint comparison.
            update_ui_callback: Callable(text) to update the GUI challenge label.
            update_frame_callback: Callable(frame) to draw the frame in GUI.
        """
        import modules.ai_voice_bot as avb
        import cv2

        phrases = [
            "the sky is blue",
            "hello world",
            "open sesame",
            "secure authorization confirmed",
            "voice identity recognized",
            "biometric access granted",
            "encryption key verified",
            "authentication successful",
            "identity confirmation active",
            "multi-layer security check",
            "system integrity verified",
            "advanced neural matching",
            "cybersecurity protocol active",
            "digital signature confirmed"
        ]
        phrase = random.choice(phrases)

        # ── 1. Bot speaks the phrase ──
        if update_ui_callback:
            update_ui_callback(f"🎤  Listen carefully...")
        avb.bot.speak_sync(f"now say: {phrase}")

        # ── 2. GET READY countdown (2-1) ──
        for countdown in [2, 1]:
            if update_ui_callback:
                update_ui_callback(
                    f"🎙️  Get ready to say: \"{phrase}\"\n"
                    f"     Recording in {countdown}...")
            # Keep video feed alive during countdown
            t_end = time.time() + 0.75
            while time.time() < t_end:
                if update_ui_callback and hasattr(update_ui_callback, '__self__'):
                    if getattr(update_ui_callback.__self__, 'pipeline_abort', False):
                        return False, "Aborted by user."
                if camera and update_frame_callback:
                    frame = camera.read_frame()
                    if frame is not None:
                        update_frame_callback(frame)
                time.sleep(0.02)

        # ── 3. RECORDING with live timer ──
        duration = 5

        # Record to secure temp directory (not project root)
        record_result = [None]
        def do_record():
            record_result[0] = avb.bot.record_audio(duration, "voice_auth_challenge.wav")

        rec_thread = threading.Thread(target=do_record, daemon=True)
        rec_thread.start()

        start = time.time()
        frame_skip = 0
        while rec_thread.is_alive():
            if update_ui_callback and hasattr(update_ui_callback, '__self__'):
                if getattr(update_ui_callback.__self__, 'pipeline_abort', False):
                    # We can't easily kill the recording thread, but we can stop waiting for it
                    return False, "Aborted by user."

            elapsed = time.time() - start
            remaining = max(0, duration - elapsed)
            
            if update_ui_callback:
                bar_total = 20
                bar_filled = int((elapsed / duration) * bar_total)
                bar_empty = bar_total - bar_filled
                bar = "█" * bar_filled + "░" * bar_empty
                update_ui_callback(
                    f"🔴  RECORDING NOW!  Say: \"{phrase}\"\n"
                    f"     {bar}  {remaining:.0f}s")
            
            if camera and update_frame_callback:
                frame = camera.read_frame()
                if frame is not None:
                    frame_skip += 1
                    if frame_skip % 2 == 0:
                        update_frame_callback(frame.copy())
            
            time.sleep(0.02)

        wav_file = record_result[0] or "voice_auth_challenge.wav"

        # ── 4. Processing ──
        if update_ui_callback:
            update_ui_callback("⏳  Analyzing your voice...")

        try:
            transcript = avb.bot.transcribe(wav_file).lower()
            print(f"[VOICE CHALLENGE] Expected: '{phrase}'")
            print(f"[VOICE CHALLENGE] Heard:    '{transcript}'")

            if not transcript:
                return False, "Could not hear or transcribe any voice."

            # ── 5. Phrase matching ──
            matched_words = 0
            expected_words = phrase.lower().split()
            heard_words = transcript.split()

            for w in expected_words:
                if w in heard_words:
                    matched_words += 1

            phrase_ratio = matched_words / len(expected_words)
            print(f"[VOICE CHALLENGE] Phrase match: {phrase_ratio:.0%}")

            if phrase_ratio < 0.4:
                return False, f"Phrase mismatch ({phrase_ratio:.0%}). Heard: '{transcript}'"

            # ── 6. Speaker verification (voiceprint vs DB) ──
            if update_ui_callback:
                update_ui_callback("🔐  Verifying voiceprint...")

            speaker_match, similarity = self._compare_voiceprint(wav_file, user_id)
            if not speaker_match:
                return False, (
                    f"Speaker mismatch (similarity: {similarity:.2f}). "
                    f"Voice does not match enrolled user.")

            # ── Result ──
            result = f"Voice verified (phrase: {phrase_ratio:.0%}"
            if self._resemblyzer_available and similarity < 1.0:
                result += f", speaker: {similarity:.2f}"
            result += ")."
            return True, result

        finally:
            # ── SECURITY: Securely wipe the voice recording ──
            if wav_file and os.path.exists(wav_file):
                secure_delete(wav_file)
                print(f"[SECURITY] Voice recording securely wiped after processing.")
