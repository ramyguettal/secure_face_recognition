"""
enroll.py — CLI tool to enroll a new user into the secure face auth system.

Captures 5 face images, averages their embeddings, optionally captures a
voiceprint sample, encrypts everything, and stores it in face_db.db.
"""

import sys
import os
import time
import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import config
from modules.camera import CameraCapture
from modules.face_recognition_module import enroll_user, _get_fernet


def capture_face_images(camera: CameraCapture, count: int = config.ENROLLMENT_CAPTURE_COUNT) -> list:
    """
    Capture multiple face images from the webcam for enrollment.

    Guides the user through capturing `count` images with slight pose variations.
    """
    images = []
    poses = [
        "Look straight at the camera",
        "Slightly turn your head LEFT",
        "Slightly turn your head RIGHT",
        "Tilt your head slightly UP",
        "Look straight again",
    ]

    print(f"\n{'=' * 55}")
    print("  📸 FACE ENROLLMENT — Capturing {0} images".format(count))
    print(f"{'=' * 55}\n")

    if not camera.open():
        print("[ERROR] Cannot open camera for enrollment.")
        return images

    for i in range(count):
        instruction = poses[i] if i < len(poses) else f"Image {i + 1}"
        print(f"  [{i + 1}/{count}] {instruction}")
        print("  Press SPACE to capture, Q to cancel.\n")

        while True:
            frame = camera.read_frame()
            if frame is None:
                continue

            # Check for face
            result = camera.detect_face(frame)
            display = frame.copy()

            if result:
                _, bbox = result
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                status = "Face detected — press SPACE"
                color = (0, 255, 0)
            else:
                status = "No face — adjust position"
                color = (0, 0, 255)

            cv2.putText(display, f"[{i + 1}/{count}] {instruction}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(display, status, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            cv2.imshow("Enrollment", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and result:
                images.append(frame.copy())
                print(f"  ✓ Image {i + 1} captured!")
                time.sleep(0.5)
                break
            elif key == ord('q'):
                print("[INFO] Enrollment cancelled by user.")
                return images

    return images


def capture_voiceprint() -> "np.ndarray | None":
    """
    Capture a voiceprint sample for speaker verification.
    Uses resemblyzer to compute a voice embedding.
    """
    try:
        import speech_recognition as sr
        from resemblyzer import VoiceEncoder, preprocess_wav
        import tempfile
        import wave
    except ImportError as e:
        print(f"[WARN] Voiceprint enrollment unavailable: {e}")
        print("[INFO] Skipping voiceprint — voice challenge will use phrase matching only.")
        return None

    print("\n  🎤 VOICEPRINT ENROLLMENT")
    print("  Please say: 'My voice is my password, verify me.'")
    print("  Recording starts in 2 seconds...\n")
    time.sleep(2)

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("  [Recording...] Speak now!")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

        # Save audio to temp WAV file for resemblyzer
        temp_wav = os.path.join(tempfile.gettempdir(), "enroll_voice.wav")
        with open(temp_wav, "wb") as f:
            f.write(audio.get_wav_data())

        # Compute voiceprint embedding
        encoder = VoiceEncoder()
        wav = preprocess_wav(temp_wav)
        embedding = encoder.embed_utterance(wav)
        print("  ✓ Voiceprint captured!")

        # Clean up
        os.remove(temp_wav)
        return embedding.astype(np.float32)

    except sr.WaitTimeoutError:
        print("  [WARN] No speech detected. Skipping voiceprint.")
    except Exception as e:
        print(f"  [WARN] Voiceprint capture failed: {e}")

    return None


def main() -> None:
    """Main enrollment flow."""
    print("\n" + "=" * 55)
    print("  🔐 SECURE FACE AUTH — USER ENROLLMENT")
    print("=" * 55)

    # Check encryption key
    try:
        _get_fernet()
    except EnvironmentError as e:
        print(f"\n{e}")
        print("\n[SETUP] To generate a key, run:")
        print('  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')
        print(f"\n[SETUP] Then set it:")
        print(f"  set {config.FACE_DB_KEY_ENV}=<your_key>")
        sys.exit(1)

    # Get user info
    user_id = input("\n  Enter user ID (e.g. 'alice'): ").strip()
    if not user_id:
        print("[ERROR] User ID cannot be empty.")
        sys.exit(1)

    display_name = input("  Enter display name (e.g. 'Alice Johnson'): ").strip()
    if not display_name:
        display_name = user_id

    # Capture face images
    camera = CameraCapture()
    face_images = capture_face_images(camera, count=config.ENROLLMENT_CAPTURE_COUNT)
    camera.release()

    if len(face_images) < 3:
        print(f"\n[ERROR] Need at least 3 face images, got {len(face_images)}. Aborting.")
        sys.exit(1)

    # Capture voiceprint (optional)
    voiceprint = None
    vp_choice = input("\n  Enroll voiceprint for speaker verification? (y/n): ").strip().lower()
    if vp_choice == 'y':
        voiceprint = capture_voiceprint()

    # Enroll
    print("\n  Enrolling user...")
    success = enroll_user(user_id, display_name, face_images, voiceprint)

    if success:
        print(f"\n  ✅ User '{display_name}' ({user_id}) enrolled successfully!")
        print(f"  Database: {config.FACE_DB_PATH}")
    else:
        print(f"\n  ❌ Enrollment failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
