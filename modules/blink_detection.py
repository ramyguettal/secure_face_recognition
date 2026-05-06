"""
blink_detection.py — Step 5: Robust EAR blink detection with anti-replay.

Uses face_recognition (dlib 68-point landmarks) to compute Eye Aspect Ratio (EAR).
Detects a blink as a significant EAR drop from the user's personal baseline.
Security: Random blink timing, EAR variance check to reject looped video.
"""

import sys
import os
import time
import random
import cv2
import numpy as np
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import face_recognition


class BlinkDetection:
    """Robust EAR blink detection with personal baseline calibration."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _compute_ear_dlib(eye_points: List[Tuple[int, int]]) -> float:
        """
        Compute Eye Aspect Ratio from 6 dlib eye points.
        Points: 0-left, 1-top-left, 2-top-right, 3-right, 4-bottom-right, 5-bottom-left
        """
        if len(eye_points) != 6:
            return 0.3  # safe default (open)

        p = [np.array(pt, dtype=float) for pt in eye_points]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h_dist = np.linalg.norm(p[0] - p[3])

        if h_dist < 1e-6:
            return 0.0

        ear = (v1 + v2) / (2.0 * h_dist)
        return float(ear)

    def _compute_both_ears(self, landmarks_dict: dict) -> Optional[float]:
        """Compute average EAR across both eyes. Returns None if eyes not found."""
        left_pts = landmarks_dict.get('left_eye', [])
        right_pts = landmarks_dict.get('right_eye', [])
        if len(left_pts) != 6 or len(right_pts) != 6:
            return None
        left_ear = self._compute_ear_dlib(left_pts)
        right_ear = self._compute_ear_dlib(right_pts)
        return (left_ear + right_ear) / 2.0

    def _get_ear_from_frame(self, frame: np.ndarray) -> Optional[float]:
        """Extract EAR from a single frame. Returns None if no face/eyes found."""
        from modules.camera import _dlib_lock
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use full resolution for better landmark accuracy
        with _dlib_lock:
            locations = face_recognition.face_locations(rgb, model="hog")
            if not locations:
                return None
            landmarks = face_recognition.face_landmarks(rgb, face_locations=locations)
        if not landmarks:
            return None
        lm = landmarks[0]
        return self._compute_both_ears(lm)

    def run_challenge(self, camera,
                      pad_passed: bool = True,
                      use_gui: bool = False,
                      update_frame_callback=None) -> Tuple[bool, str]:
        """
        Run the blink detection liveness challenge.

        Strategy:
        1. Calibrate: Read ~1 second of frames to establish the user's open-eye EAR baseline
        2. Wait for a blink: detect a significant EAR drop (>30% below baseline)
        3. Verify: EAR must come back up after the drop (confirming a real blink)
        """
        if not pad_passed:
            return False, "Anti-spoofing (PAD) check did not pass."

        # ── Phase 1: Calibrate baseline EAR (1 second) ──
        print("[BLINK] Phase 1: Calibrating baseline EAR...")
        baseline_ears = []
        cal_start = time.time()

        frame_skip = 0
        while time.time() - cal_start < 2.5:
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

            # Downscale for speed during calibration
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            from modules.camera import _dlib_lock
            with _dlib_lock:
                locations = face_recognition.face_locations(rgb, model="hog")
                if locations:
                    landmarks = face_recognition.face_landmarks(rgb, face_locations=locations)
            if locations and landmarks:
                ear = self._compute_both_ears(landmarks[0])
                if ear is not None:
                    baseline_ears.append(ear)

            if not use_gui:
                time.sleep(0.01)
            else:
                time.sleep(0.02)

        if len(baseline_ears) < 3:
            return False, "Could not calibrate — face not visible during setup."

        baseline = np.mean(baseline_ears)
        # Dynamic threshold: 70% of user's personal baseline
        blink_threshold = baseline * 0.70
        print(f"[BLINK] Baseline EAR: {baseline:.3f}, Blink threshold: {blink_threshold:.3f}")

        # ── Phase 2: Short random delay before prompt ──
        delay = random.uniform(0.5, 1.5)
        print(f"[BLINK] Waiting {delay:.1f}s before prompt...")
        time.sleep(delay)

        # Voice prompt (already spoken by the pipeline, so just log)
        import modules.ai_voice_bot as avb
        avb.bot.speak("please blink now")
        print("[BLINK] Please BLINK now!")

        # ── Phase 3: Detect blink ──
        ear_history: List[float] = []
        blink_detected = False
        eyes_closed = False
        timeout = 8.0   # generous timeout
        start_time = time.time()

        frame_skip = 0
        while time.time() - start_time < timeout:
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

            # Use downscaled frame for speed
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            from modules.camera import _dlib_lock
            landmarks = None
            with _dlib_lock:
                locations = face_recognition.face_locations(rgb, model="hog")
                if locations:
                    landmarks_list = face_recognition.face_landmarks(rgb, face_locations=locations)
                    if landmarks_list:
                        landmarks = landmarks_list[0]

            if landmarks:
                ear = self._compute_both_ears(landmarks)
                if ear is not None:
                    ear_history.append(ear)

                    # State machine: detect close then open
                    if ear < blink_threshold:
                        eyes_closed = True
                    elif eyes_closed and ear > blink_threshold:
                        # Eyes were closed and now opened → blink!
                        blink_detected = True
                        print(f"[BLINK] Blink detected! EAR dropped to "
                              f"{min(ear_history[-5:]):.3f} then recovered to {ear:.3f}")
                        break

            if not use_gui:
                display = frame.copy()
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                cv2.putText(display, "BLINK NOW!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(display, f"Time: {remaining:.1f}s", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if ear_history:
                    cv2.putText(display, f"EAR: {ear_history[-1]:.3f}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Secure Face Auth", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False, "User cancelled."
            else:
                time.sleep(0.02)  # yield CPU in GUI mode

        if not blink_detected:
            reason = "No blink detected within the time window."
            if ear_history:
                reason += f" (baseline={baseline:.3f}, min_ear={min(ear_history):.3f}, threshold={blink_threshold:.3f})"
            return False, reason

        # ── Phase 4: Anti-replay variance check ──
        if len(ear_history) >= 5:
            variance = float(np.var(ear_history))
            if variance < 0.0001:
                return False, (
                    f"EAR variance too low ({variance:.6f}). "
                    f"Possible looped video detected."
                )

        return True, "Blink detected and verified successfully."

    def close(self) -> None:
        pass


if __name__ == "__main__":
    print("Blink Detection Challenge — Test")
    detector = BlinkDetection()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        success, reason = detector.run_challenge(cap, pad_passed=True, use_gui=False)
        print(f"  Result: {'PASS' if success else 'FAIL'} — {reason}")
        cap.release()
    else:
        print("[ERROR] Cannot open camera.")
    cv2.destroyAllWindows()
