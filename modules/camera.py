"""
camera.py — Step 1: Webcam capture and face detection.

Supports both local webcams (by index) and external IP/RTSP cameras.
Uses standard face_recognition (dlib) to detect face bounds.
"""

import sys
import os
import cv2
import numpy as np
import socket
from typing import Optional, Tuple, Union, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import face_recognition

class CameraCapture:
    """Manages webcam access (local or external IP) and face detection."""

    def __init__(self, camera_source: Union[int, str] = config.CAMERA_INDEX) -> None:
        """
        Args:
            camera_source: Either an integer index for local cameras,
                           or a string URL for external IP/RTSP cameras.
        """
        self.camera_source = camera_source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_external = isinstance(camera_source, str)

    @staticmethod
    def build_external_urls(ip: str, port: str) -> List[str]:
        """Build a list of common IP camera stream URLs to try."""
        ip = ip.strip()
        port = port.strip()
        urls = [
            f"http://{ip}:{port}/video",
            f"http://{ip}:{port}/videofeed",
            f"http://{ip}:{port}/shot.jpg",
            f"http://{ip}:{port}/mjpg/video.mjpg",
            f"http://{ip}:{port}/cgi-bin/mjpg/video.cgi",
            f"http://{ip}:{port}/stream.mjpg",
            f"rtsp://{ip}:{port}/stream",
            f"rtsp://{ip}:{port}/live",
            f"rtsp://{ip}:{port}/",
            f"rtsp://{ip}:{port}/h264_ulaw.sdp",
            f"http://{ip}:{port}/mjpegfeed",
            f"http://{ip}:{port}/",
        ]
        return urls

    def open(self) -> bool:
        """Open the camera (local or external). Returns True on success."""
        try:
            if self.is_external:
                print(f"[INFO] Connecting to external camera: {self.camera_source}")
            else:
                print(f"[INFO] Opening local camera index: {self.camera_source}")

            self.cap = cv2.VideoCapture(self.camera_source)

            if not self.cap.isOpened():
                source_desc = f"external camera at {self.camera_source}" if self.is_external else f"local camera #{self.camera_source}"
                print(f"[ERROR] Could not open {source_desc}.")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("[ERROR] Camera opened but cannot read frames.")
                self.cap.release()
                self.cap = None
                return False

            print("[INFO] Camera opened successfully.")
            return True
        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    @classmethod
    def try_connect_external(cls, ip: str, port: str) -> Optional['CameraCapture']:
        """Try to connect to an external camera using multiple URL formats."""
        print(f"[INFO] Checking if {ip}:{port} is reachable...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((ip, int(port)))
            s.close()
        except Exception as e:
            print(f"[ERROR] Host {ip}:{port} is unreachable. Fast-failing. ({e})")
            return None

        urls = cls.build_external_urls(ip, port)
        print(f"[INFO] Host is up. Trying {len(urls)} URL formats for {ip}:{port}...")

        for url in urls:
            print(f"  Trying: {url}")
            cam = cls(camera_source=url)
            if cam.open():
                print(f"  ✓ Connected: {url}")
                return cam
            if cam.cap:
                cam.cap.release()
                cam.cap = None

        print(f"[ERROR] All URL formats failed for {ip}:{port}")
        return None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return None

        # Flush stale buffered frames for external (WiFi/IP) cameras
        if self.is_external:
            self.cap.grab()  # discard buffered frame
            self.cap.grab()  # discard another

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        # Mirror the frame (Left-to-Right inversion)
        frame = cv2.flip(frame, 1)
        return frame

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect a face in the frame using face_recognition (dlib).
        Returns: Tuple of (frame, (x, y, w, h)) if face detected, None otherwise.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # We process a highly compressed version internally to be super fast
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        
        locations = face_recognition.face_locations(small_frame, model="hog")
        if locations:
            top, right, bottom, left = locations[0]
            # scale back up
            top *= 2; right *= 2; bottom *= 2; left *= 2
            h, w, _ = frame.shape
            
            x = max(0, left)
            y = max(0, top)
            bw = min(right - left, w - x)
            bh = min(bottom - top, h - y)
            
            if bw > 0 and bh > 0:
                return frame, (x, y, bw, bh)
                
        return None

    def wait_for_face(self, timeout_frames: int = 300,
                      use_gui: bool = False) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Wait until a face is detected in the camera feed."""
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return None

        print("[INFO] Waiting for face to appear on camera...")
        for _ in range(timeout_frames):
            frame = self.read_frame()
            if frame is None:
                continue

            result = self.detect_face(frame)
            if result is not None:
                print("[INFO] Face detected!")
                return result

            if not use_gui:
                display = frame.copy()
                cv2.putText(display, "Looking for face...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Secure Face Auth", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] User cancelled.")
                    return None

        print("[WARN] No face detected within timeout.")
        return None

    def release(self) -> None:
        """Release the camera and close windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = CameraCapture()
    result = cam.wait_for_face()
    if result:
        frame, bbox = result
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detected Face", frame)
        cv2.waitKey(0)
    cam.release()
