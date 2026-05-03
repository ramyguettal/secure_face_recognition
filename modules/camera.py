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
        self.is_droidcam = False
        self.droidcam_session: Optional[requests.Session] = None

    @staticmethod
    def build_external_urls(ip: str, port: str) -> List[str]:
        """Build a list of common IP camera stream URLs to try."""
        ip = ip.strip()
        port = port.strip()
        urls = [
            # DroidCam URLs (with and without auth)
            f"http://{ip}:{port}/video",
            f"http://admin:admin@{ip}:{port}/video",
            f"http://admin:droidcam@{ip}:{port}/video",
            # Standard HTTP endpoints
            f"http://{ip}:{port}/videofeed",
            f"http://{ip}:{port}/shot.jpg",
            f"http://{ip}:{port}/mjpg/video.mjpg",
            f"http://{ip}:{port}/cgi-bin/mjpg/video.cgi",
            f"http://{ip}:{port}/stream.mjpg",
            # RTSP endpoints
            f"rtsp://{ip}:{port}/stream",
            f"rtsp://{ip}:{port}/live",
            f"rtsp://{ip}:{port}/",
            f"rtsp://{ip}:{port}/h264_ulaw.sdp",
            # Fallback HTTP
            f"http://{ip}:{port}/mjpegfeed",
            f"http://{ip}:{port}/",
        ]
        return urls

    def open(self) -> bool:
        """Open the camera (local or external). Returns True on success."""
        try:
            if self.is_external:
                print(f"[INFO] Connecting to external camera: {self.camera_source}")
                # Detect if this is a DroidCam stream
                if self._try_droidcam_connection():
                    return True
            else:
                print(f"[INFO] Opening local camera index: {self.camera_source}")

            # On Windows, explicitly use DirectShow to avoid MSMF crashes with internal webcams
            if not self.is_external and os.name == 'nt':
                self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_source)

            if not self.cap.isOpened():
                source_desc = f"external camera at {self.camera_source}" if self.is_external else f"local camera #{self.camera_source}"
                print(f"[ERROR] Could not open {source_desc}.")
                return False

            # Set camera properties for local cameras
            if not self.is_external:
                # Use CAP_PROP_BUFFERSIZE to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
                # Set FPS
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                # Auto-focus and exposure
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

            # Test reading a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("[ERROR] Camera opened but cannot read frames.")
                self.cap.release()
                self.cap = None
                return False

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"[INFO] Camera opened successfully. Resolution: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            return True
        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def _try_droidcam_connection(self) -> bool:
        """Try to connect to DroidCam using HTTP MJPEG stream."""
        try:
            # Try both with and without auth
            auth_variants = [None, ("admin", "admin"), ("admin", "droidcam")]
            
            for auth in auth_variants:
                try:
                    session = requests.Session()
                    session.timeout = 5
                    
                    # Test connection
                    response = session.get(self.camera_source, auth=auth, stream=True, timeout=5)
                    if response.status_code == 200:
                        print(f"[INFO] DroidCam stream authenticated successfully")
                        self.droidcam_session = session
                        self.is_droidcam = True
                        
                        # Test reading a frame
                        frame = self._read_droidcam_frame()
                        if frame is not None:
                            print("[INFO] Successfully connected to DroidCam")
                            return True
                except Exception as e:
                    continue
            
            return False
        except Exception as e:
            print(f"[DEBUG] DroidCam connection failed: {e}")
            return False

    def _read_droidcam_frame(self) -> Optional[np.ndarray]:
        """Read a frame from DroidCam MJPEG stream."""
        try:
            if self.droidcam_session is None:
                return None
                
            response = self.droidcam_session.get(self.camera_source, stream=True, timeout=5)
            
            # Parse MJPEG boundary
            boundary = response.headers.get('content-type', '').split('boundary=')
            if len(boundary) < 2:
                # Try reading as raw JPEG
                if response.content:
                    nparr = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return frame
                return None
            
            boundary = boundary[1].encode()
            data = response.content
            
            # Find JPEG frame in MJPEG stream
            start = data.find(b'\xff\xd8')  # JPEG SOI
            end = data.find(b'\xff\xd9')    # JPEG EOI
            
            if start != -1 and end != -1 and end > start:
                jpeg_data = data[start:end + 2]
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame
            
            return None
        except Exception as e:
            print(f"[DEBUG] Error reading DroidCam frame: {e}")
            return None

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

    @classmethod
    def find_best_local_camera(cls) -> Optional['CameraCapture']:
        """Try to find and connect to the best available local camera."""
        print("[INFO] Scanning for available local cameras...")
        for camera_idx in range(10):  # Try camera indices 0-9
            try:
                print(f"  Trying camera index {camera_idx}...", end=" ")
                cam = cls(camera_source=camera_idx)
                if cam.open():
                    print(f"✓ Found!")
                    return cam
                else:
                    print("✗ Failed")
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        print("[ERROR] No working local cameras found.")
        return None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera."""
        if self.is_droidcam:
            frame = self._read_droidcam_frame()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                return frame
            return None
            
        if self.cap is None or not self.cap.isOpened():
            return None

        # Flush stale buffered frames to ensure fresh capture
        # This is important for both local and external cameras
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
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Ensure the array is C-contiguous to prevent dlib segfaults
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
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
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            
        return None

    def wait_for_face(self, timeout_frames: int = 300,
                      use_gui: bool = False) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Wait until a face is detected in the camera feed."""
        # Reopen camera if not already open
        if (not self.is_droidcam and (not self.cap or not self.cap.isOpened())) or \
           (self.is_droidcam and self.droidcam_session is None):
            if not self.open():
                print("[ERROR] Failed to open camera.")
                return None

        print("[INFO] Waiting for face to appear on camera...")
        frames_checked = 0
        faces_detected_count = 0
        
        for i in range(timeout_frames):
            frame = self.read_frame()
            if frame is None:
                print(f"[WARN] Frame {i} is None, retrying...")
                continue

            frames_checked += 1
            result = self.detect_face(frame)
            
            if result is not None:
                faces_detected_count += 1
                print(f"[INFO] Face detected! (detection #{faces_detected_count})")
                return result

            # Display live feed if GUI is enabled
            if use_gui:
                display = frame.copy()
                cv2.putText(display, f"Looking for face... ({i+1}/{timeout_frames})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Press 'q' to cancel", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                cv2.imshow("Secure Face Auth - Face Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] User cancelled.")
                    return None

        print(f"[WARN] No face detected within {timeout_frames} frames ({frames_checked} frames processed).")
        return None

    def release(self) -> None:
        """Release the camera and close windows."""
        if self.droidcam_session is not None:
            try:
                self.droidcam_session.close()
            except:
                pass
            self.droidcam_session = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== Secure Face Recognition - Camera Test ===\n")
    
    # Try local camera first
    print("1. Attempting to connect to local camera...")
    cam = CameraCapture(camera_source=config.CAMERA_INDEX)
    if not cam.open():
        print("2. Local camera failed, trying to auto-detect best camera...")
        cam = CameraCapture.find_best_local_camera()
        if not cam:
            print("\nNo cameras available!")
            exit(1)
    
    print("\n3. Waiting for face detection (press 'q' to cancel)...\n")
    result = cam.wait_for_face(timeout_frames=300, use_gui=True)
    
    if result:
        frame, bbox = result
        x, y, w, h = bbox
        print(f"\n✓ Face detected at: x={x}, y={y}, w={w}, h={h}")
        
        # Display detected face
        display = frame.copy()
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(display, "Face Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Detection Result", display)
        print("Press any key to close...")
        cv2.waitKey(0)
    else:
        print("\n✗ No face detected or user cancelled.")
    
    cam.release()
    print("\nTest completed.")
