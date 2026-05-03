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
from modules.face_recognition_module import (
    enroll_user, _get_fernet, check_user_exists,
    get_live_embedding, match_face, decrypt_embedding
)


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


def check_face_duplicate(face_images: list) -> "tuple | None":
    """
    Check if the captured faces already exist in the database.
    
    Returns:
        (user_id, display_name, distance) if face match found, None otherwise.
    """
    if not face_images:
        return None
    
    print("\n  🔍 Checking if this face is already enrolled...")
    
    # Extract embeddings from all captured images
    embeddings = []
    for idx, img in enumerate(face_images):
        try:
            embedding = get_live_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"    Image {idx + 1}: ✓ Face detected")
            else:
                print(f"    Image {idx + 1}: ✗ No face found")
        except Exception as e:
            print(f"    Image {idx + 1}: ✗ Error: {e}")
    
    if not embeddings:
        print("  ⚠️  Could not extract face embeddings")
        return None
    
    # Average embeddings (same as enrollment does)
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Check if this face matches any enrolled user
    match_result = match_face(avg_embedding)
    
    return match_result


def capture_voiceprint() -> "np.ndarray | None":
    """
    Capture a voiceprint sample for speaker verification.
    Uses resemblyzer to compute a voice embedding.
    """
    try:
        import speech_recognition as sr
        from resemblyzer import VoiceEncoder, preprocess_wav
        import wave
        from modules.secure_storage import get_secure_temp_dir, secure_delete
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

        # Save audio to secure temp WAV file for resemblyzer
        temp_wav = os.path.join(get_secure_temp_dir(), "enroll_voice_tmp.wav")
        with open(temp_wav, "wb") as f:
            f.write(audio.get_wav_data())

        # Compute voiceprint embedding
        encoder = VoiceEncoder()
        wav = preprocess_wav(temp_wav)
        embedding = encoder.embed_utterance(wav)
        print("  ✓ Voiceprint captured!")

        # Securely wipe the temporary voice file
        secure_delete(temp_wav)
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

    # ✅ CHECK IF USER ALREADY EXISTS (BEFORE capturing images!)
    if check_user_exists(user_id):
        print(f"\n  ⚠️  User '{user_id}' is already enrolled!")
        update_choice = input("  Re-enroll this user? (y/n): ").strip().lower()
        if update_choice != 'y':
            print("  [INFO] Enrollment cancelled.")
            sys.exit(0)
        print("  [INFO] Starting re-enrollment process...\n")

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

    # ✅ CHECK IF FACE ALREADY EXISTS IN DATABASE
    face_match = check_face_duplicate(face_images)
    if face_match:
        matched_user_id, matched_display_name, distance = face_match
        print(f"\n  ⚠️  WARNING: This face matches an existing user!")
        print(f"     User ID: {matched_user_id}")
        print(f"     Display Name: {matched_display_name}")
        print(f"     Match distance: {distance:.4f}")
        
        if matched_user_id == user_id:
            print(f"     (Same user - this is a re-enrollment)")
        else:
            print(f"     ⚠️  Different user detected!")
        
        force_enroll = input("\n  Continue anyway? (y/n): ").strip().lower()
        if force_enroll != 'y':
            print("  [INFO] Enrollment cancelled to avoid duplicates.")
            sys.exit(0)
        print("  [INFO] Proceeding with enrollment...\n")

    # ✅ CONFIRMATION BEFORE PROCEEDING
    print(f"\n  👤 User ID: {user_id}")
    print(f"  📝 Display Name: {display_name}")
    print(f"  📸 Face Images Captured: {len(face_images)}")
    confirm = input("\n  Proceed with enrollment? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  [INFO] Enrollment cancelled.")
        sys.exit(0)

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
