"""
head_movement.py — Step 4: Head movement liveness challenge.

Uses face_recognition (dlib 68-point landmarks) to track head pose and verify
the user can perform a randomly selected direction challenge.
Security: Random direction per session, continuous motion verification.
"""

import sys
import os
import time
import random
import cv2
import numpy as np
from typing import Optional, Tuple, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import face_recognition


class HeadMovementChallenge:
    """Head pose tracking and liveness challenge using dlib landmarks."""

    def __init__(self) -> None:
        self.last_direction: Optional[str] = None

    def _estimate_head_pose_simple(self, landmarks_dict: dict) -> Tuple[float, float]:
        """
        Estimate head yaw and pitch based on facial feature positions.
        Returns:
            (yaw, pitch) in approximate arbitrary degrees.
        """
        nose_tip = np.array(landmarks_dict["nose_tip"][2])
        left_eye_center = np.mean([np.array(pt) for pt in landmarks_dict["left_eye"]], axis=0)
        right_eye_center = np.mean([np.array(pt) for pt in landmarks_dict["right_eye"]], axis=0)
        chin = np.array(landmarks_dict["chin"][8])
        
        mid_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
        face_width = abs(right_eye_center[0] - left_eye_center[0]) * 2.5
        if face_width > 0.01:
            yaw = ((nose_tip[0] - mid_x) / face_width) * 90.0
        else:
            yaw = 0.0
            
        mid_y = (left_eye_center[1] + right_eye_center[1]) / 2.0
        face_height = abs(chin[1] - mid_y)
        if face_height > 0.01:
            pitch = ((nose_tip[1] - mid_y) / face_height) * 90.0
        else:
            pitch = 0.0
            
        return yaw, pitch

    def _get_random_direction(self) -> str:
        """Pick a random direction for the challenge."""
        return random.choice(config.HEAD_CHALLENGE_DIRECTIONS)

    def run_challenge(self, camera,
                      use_gui: bool = False,
                      preset_direction: str = None,
                      update_frame_callback=None) -> Tuple[bool, str]:
        """
        Run the head movement liveness challenge.

        Args:
            camera: CameraCapture instance.
            use_gui: If True, skip cv2.imshow/waitKey (GUI handles display).
            preset_direction: If provided, use this direction instead of random.

        Returns:
            (success: bool, reason: str)
        """
        if preset_direction:
            direction = preset_direction
        else:
            direction = random.choice(config.HEAD_CHALLENGE_DIRECTIONS)
        self.last_direction = direction
        
        if not preset_direction:
            # Only speak if pipeline hasn't already spoken
            import modules.ai_voice_bot as avb
            avb.bot.speak(f"Please {direction}")
        print(f"[CHALLENGE] Head movement: Please {direction}.")

        start_time = time.time()
        initial_yaw: Optional[float] = None
        initial_pitch: Optional[float] = None
        intermediate_readings: List[Tuple[float, float]] = []
        challenge_met = False

        frame_skip = 0
        while time.time() - start_time < config.HEAD_CHALLENGE_TIMEOUT:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_skip += 1
            if frame_skip % 2 != 0:
                if update_frame_callback:
                    update_frame_callback(frame)
                continue

            if update_frame_callback:
                update_frame_callback(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            
            # Acquire lock — dlib C++ is not thread-safe
            from modules.camera import _dlib_lock
            landmarks = None
            with _dlib_lock:
                locations = face_recognition.face_locations(small_rgb, model="hog")
                if locations:
                    landmarks_list = face_recognition.face_landmarks(small_rgb, face_locations=locations)
                    if landmarks_list:
                        landmarks = landmarks_list[0]

            if landmarks:
                lm = landmarks
                if all(k in lm for k in ['nose_tip', 'left_eye', 'right_eye', 'chin']):
                    yaw, pitch = self._estimate_head_pose_simple(lm)

                    if initial_yaw is None:
                        initial_yaw = yaw
                        initial_pitch = pitch
                        continue

                    delta_yaw = yaw - initial_yaw
                    delta_pitch = pitch - initial_pitch

                    intermediate_readings.append((delta_yaw, delta_pitch))
                    print(f"[HEAD] delta_yaw={delta_yaw:.1f}  delta_pitch={delta_pitch:.1f}  target={direction}")

                    # Check if the correct movement is detected
                    if direction == "turn left" and delta_yaw < -config.HEAD_POSE_YAW_THRESHOLD:
                        challenge_met = True
                    elif direction == "turn right" and delta_yaw > config.HEAD_POSE_YAW_THRESHOLD:
                        challenge_met = True
                    elif direction == "tilt up" and delta_pitch < -config.HEAD_POSE_PITCH_THRESHOLD:
                        challenge_met = True
                    elif direction == "tilt down" and delta_pitch > config.HEAD_POSE_PITCH_THRESHOLD:
                        challenge_met = True

                    if challenge_met:
                        print(f"[HEAD] Challenge MET! delta_yaw={delta_yaw:.1f} delta_pitch={delta_pitch:.1f}")
                        break

            # Only show OpenCV window in CLI mode
            if not use_gui:
                display = frame.copy()
                elapsed = time.time() - start_time
                remaining = config.HEAD_CHALLENGE_TIMEOUT - elapsed
                cv2.putText(display, f"Please {direction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Time: {remaining:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Secure Face Auth", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False, "User cancelled"
            else:
                time.sleep(0.01)  # Yield CPU in GUI mode

        if not challenge_met:
            return False, f"Head movement '{direction}' not detected."

        # Verify continuous motion
        if len(intermediate_readings) < config.HEAD_MIN_INTERMEDIATE_FRAMES:
            return False, (
                f"Insufficient intermediate motion frames "
                f"({len(intermediate_readings)} < {config.HEAD_MIN_INTERMEDIATE_FRAMES}). "
                f"Possible replay attack."
            )

        # Verify motion was progressive
        if direction in ("turn left", "turn right"):
            values = [r[0] for r in intermediate_readings]
        else:
            values = [r[1] for r in intermediate_readings]

        monotonic_count = 0
        for i in range(1, len(values)):
            if direction in ("turn right", "tilt down"):
                if values[i] >= values[i - 1] - 1.0:
                    monotonic_count += 1
            else:
                if values[i] <= values[i - 1] + 1.0:
                    monotonic_count += 1

        monotonic_ratio = monotonic_count / max(len(values) - 1, 1)
        print(f"[HEAD] Monotonic ratio: {monotonic_ratio:.2f}, readings: {len(intermediate_readings)}")
        if monotonic_ratio < 0.2:
            return False, (
                f"Motion not continuous (monotonic ratio: {monotonic_ratio:.2f}). "
                f"Possible pre-recorded replay."
            )

        return True, f"Head movement '{direction}' verified successfully."

    def close(self) -> None:
        pass


if __name__ == "__main__":
    print("Head Movement Challenge — Test")
    challenge = HeadMovementChallenge()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        success, reason = challenge.run_challenge(cap, use_gui=False)
        print(f"  Result: {'PASS' if success else 'FAIL'} — {reason}")
        cap.release()
    else:
        print("[ERROR] Cannot open camera.")
    cv2.destroyAllWindows()
