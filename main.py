"""
main.py — Entry point for the Secure Face Recognition Authentication System.

Supports two modes:
  - GUI mode (default): Launches the full graphical interface
  - CLI mode (--cli flag): Runs the text-based pipeline in terminal

Orchestrates the full 6-step pipeline with 2 added security layers:
  Step 1: Camera capture + face detection
  Step 2: Face recognition against encrypted database (with rate limiting)
  Decision Gate: Known face? (deny if not, with backoff/lockout)
  NEW-1: Anti-spoofing / PAD (texture + screen glow analysis)
  Step 3: AI voice greeting (neutral — no username spoken)
  Step 4: Head movement liveness challenge (random direction)
  Step 5: Blink detection (random timing + EAR variance check)
  Step 6: Voice passphrase challenge (random phrase from pool of 20)
  NEW-2: Session token generation + audit logging

The pipeline short-circuits immediately on any step failure.
"""

import sys
import os
import time
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import config
from modules.camera import CameraCapture
from modules.face_recognition_module import (
    get_live_embedding, match_face, rate_limiter, _get_fernet
)
from modules.anti_spoofing import AntiSpoofing
from modules.voice_greeting import VoiceGreeting
from modules.head_movement import HeadMovementChallenge
from modules.blink_detection import BlinkDetection
from modules.voice_challenge import VoiceChallenge
from modules.session import SessionManager, AuditLogger


def run_pipeline_cli(camera_source=None) -> str | None:
    """
    Execute the full authentication pipeline in CLI mode.

    Args:
        camera_source: Camera index (int) or stream URL (str). Defaults to config.

    Returns:
        Session token string on success, None on failure.
    """
    audit = AuditLogger()
    session_mgr = SessionManager()

    if camera_source is not None:
        camera = CameraCapture(camera_source=camera_source)
    else:
        camera = CameraCapture()

    anti_spoof = AntiSpoofing()
    greeter = VoiceGreeting()
    head_challenge = HeadMovementChallenge()
    blink_challenge = BlinkDetection()
    voice_challenge = VoiceChallenge()

    matched_user_id: str = "unknown"
    matched_display_name: str = ""
    current_step: str = "init"

    try:
        # ─── STEP 1: Camera capture + face detection ─────────────────
        current_step = "camera_capture"
        print("\n" + "=" * 60)
        print("  🔐 SECURE FACE AUTH — AUTHENTICATION PIPELINE")
        print("=" * 60)
        print("\n▸ Step 1: Camera capture & face detection")

        result = camera.wait_for_face()
        if result is None:
            reason = "No face detected on camera."
            print(f"  ✗ {reason}")
            audit.log_event("unknown", "AUTH_FAILURE", current_step, "failure", reason)
            return None
        frame, face_bbox = result
        print("  ✓ Face detected on camera.")

        # ─── STEP 2: Face recognition against encrypted DB ───────────
        current_step = "face_recognition"
        print("\n▸ Step 2: Face recognition — matching against database")

        # Rate limiting check
        allowed, limit_reason = rate_limiter.check_allowed()
        if not allowed:
            print(f"  ✗ Rate limited: {limit_reason}")
            audit.log_event("unknown", "RATE_LIMITED", current_step, "failure", limit_reason)
            return None

        embedding = get_live_embedding(frame)
        if embedding is None:
            reason = "Could not extract face embedding from captured frame."
            print(f"  ✗ {reason}")
            audit.log_event("unknown", "AUTH_FAILURE", current_step, "failure", reason)
            rate_limiter.record_failure()
            return None

        match = match_face(embedding)
        if match is None:
            reason = "No matching face found in database."
            print(f"  ✗ {reason}")
            status = rate_limiter.record_failure()
            print(f"  ⚠ {status}")
            audit.log_event("unknown", "AUTH_FAILURE", current_step, "failure", reason)
            return None

        matched_user_id, matched_display_name, distance = match
        print(f"  ✓ Match found: {matched_display_name} (distance: {distance:.4f})")

        # ─── DECISION GATE PASSED — proceed to liveness ──────────────

        # ─── NEW LAYER 1: Anti-spoofing / PAD ────────────────────────
        current_step = "anti_spoofing"
        print("\n▸ Layer 1: Anti-spoofing (Presentation Attack Detection)")

        is_real, pad_confidence, pad_reason = anti_spoof.check_liveness(frame, face_bbox)
        print(f"  PAD result: real={is_real}, confidence={pad_confidence:.3f}")
        if not is_real:
            reason = f"Anti-spoofing failed: {pad_reason}"
            print(f"  ✗ {reason}")
            audit.log_event(matched_user_id, "AUTH_FAILURE", current_step, "failure", reason)
            return None
        print(f"  ✓ Liveness check passed: {pad_reason}")

        # ─── STEP 3: Voice greeting (neutral) ────────────────────────
        current_step = "voice_greeting"
        print("\n▸ Step 3: Voice greeting")

        # Only show first letter of name — never speak username
        initial = matched_display_name[0] if matched_display_name else None
        greeter.speak_greeting(user_initial=initial)
        print("  ✓ Neutral greeting delivered.")

        # ─── STEP 4: Head movement challenge ─────────────────────────
        current_step = "head_movement"
        print("\n▸ Step 4: Head movement liveness challenge")

        # Re-use existing camera capture
        if camera.cap is None or not camera.cap.isOpened():
            camera.open()

        head_ok, head_reason = head_challenge.run_challenge(camera.cap)
        if not head_ok:
            reason = f"Head movement failed: {head_reason}"
            print(f"  ✗ {reason}")
            audit.log_event(matched_user_id, "AUTH_FAILURE", current_step, "failure", reason)
            return None
        print(f"  ✓ {head_reason}")

        # ─── STEP 5: Blink detection ─────────────────────────────────
        current_step = "blink_detection"
        print("\n▸ Step 5: Blink detection challenge")

        blink_ok, blink_reason = blink_challenge.run_challenge(
            camera.cap, pad_passed=is_real
        )
        if not blink_ok:
            reason = f"Blink detection failed: {blink_reason}"
            print(f"  ✗ {reason}")
            audit.log_event(matched_user_id, "AUTH_FAILURE", current_step, "failure", reason)
            return None
        print(f"  ✓ {blink_reason}")

        # ─── STEP 6: Voice passphrase challenge ──────────────────────
        current_step = "voice_challenge"
        print("\n▸ Step 6: Voice passphrase challenge")

        voice_ok, voice_reason = voice_challenge.run_challenge(user_id=matched_user_id)
        if not voice_ok:
            reason = f"Voice challenge failed: {voice_reason}"
            print(f"  ✗ {reason}")
            audit.log_event(matched_user_id, "AUTH_FAILURE", current_step, "failure", reason)
            return None
        print(f"  ✓ {voice_reason}")

        # ─── NEW LAYER 2: Session token + audit log ──────────────────
        current_step = "session_token"
        print("\n▸ Layer 2: Session token generation")

        token = session_mgr.create_token(matched_user_id)
        rate_limiter.record_success()  # Reset failure counter

        audit.log_event(
            matched_user_id, "AUTH_SUCCESS", "all_steps",
            "success", "All challenges passed."
        )

        # ─── OUTCOME: Access granted ─────────────────────────────────
        print("\n" + "=" * 60)
        print("  ✅ VERIFICATION COMPLETE — ACCESS GRANTED")
        print(f"  User: {matched_display_name}")
        print(f"  Session token: {token[:16]}...")
        print(f"  Expires in: {config.SESSION_EXPIRY_MINUTES} minutes")
        print("=" * 60 + "\n")

        greeter.speak_result(success=True)
        return token

    except EnvironmentError as e:
        print(f"\n[CONFIG ERROR] {e}")
        audit.log_event(matched_user_id, "SYSTEM_ERROR", current_step, "failure", str(e))
        return None

    except KeyboardInterrupt:
        print("\n\n[INFO] Authentication cancelled by user (Ctrl+C).")
        audit.log_event(matched_user_id, "AUTH_CANCELLED", current_step, "failure", "User interrupt")
        return None

    except Exception as e:
        print(f"\n[SYSTEM ERROR] Unexpected error at step '{current_step}': {e}")
        audit.log_event(matched_user_id, "SYSTEM_ERROR", current_step, "failure", str(e))
        return None

    finally:
        camera.release()
        anti_spoof = None
        head_challenge.close()
        blink_challenge.close()
        greeter.cleanup()
        # Securely wipe all temporary files
        try:
            from modules.secure_storage import cleanup_secure_temp, cleanup_residual_wav_files
            cleanup_secure_temp()
            cleanup_residual_wav_files()
        except Exception:
            pass


def main() -> None:
    """Entry point — supports both GUI and CLI modes."""
    parser = argparse.ArgumentParser(
        description="Secure Face Recognition Authentication System"
    )
    parser.add_argument("--cli", action="store_true",
                        help="Run in CLI mode (no GUI)")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera source: index number or stream URL")
    args = parser.parse_args()

    if args.cli:
        # CLI mode
        try:
            _get_fernet()
        except EnvironmentError as e:
            print(f"\n{e}")
            print("\n[SETUP] Run 'python enroll.py' first to set up users.")
            sys.exit(1)

        camera_source = None
        if args.camera:
            try:
                camera_source = int(args.camera)
            except ValueError:
                camera_source = args.camera  # Treat as URL

        token = run_pipeline_cli(camera_source)
        if token:
            print(f"[OK] Authentication successful. Token: {token[:16]}...")
            sys.exit(0)
        else:
            print("[DENIED] Authentication failed.")
            sys.exit(1)
    else:
        # GUI mode (default)
        from gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
