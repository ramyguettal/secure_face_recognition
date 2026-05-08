"""
camera.py — Step 1: Webcam capture and face detection.

Supports both local webcams (by index) and external IP/RTSP cameras.
Uses standard face_recognition (dlib) to detect face bounds.
"""

import sys
import os

# ── MUST be set BEFORE importing cv2/FFmpeg ──────────────────────────────
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import numpy as np
import socket
import threading
from typing import Optional, Tuple, Union, List

try:
    import requests
except ImportError:
    requests = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
import face_recognition


# ── OS-level stderr suppression for FFmpeg C output ──────────────────────
class _SuppressStderr:
    """Context manager that redirects the OS file descriptor for stderr to
    devnull.  This catches FFmpeg's C-level fprintf(stderr, ...) which
    Python's sys.stderr redirect cannot capture."""

    def __enter__(self):
        self._old_stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *args):
        os.dup2(self._old_stderr_fd, 2)
        os.close(self._old_stderr_fd)
        os.close(self._devnull)

# Global lock to serialize all dlib C++ calls (face_locations / face_encodings).
# dlib is NOT thread-safe; concurrent calls from different threads cause segfaults.
_dlib_lock = threading.Lock()


class CameraCapture:
    """Manages webcam access (local or external IP) and face detection.
    
    Features a threaded frame grabber that continuously reads frames in the
    background, so read_frame() returns instantly and never blocks the GUI.
    """

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
        self.droidcam_session = None  # type: Optional[object]

        # Threaded frame grabber state
        self._frame_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()    # Protects _latest_frame
        self._latest_frame: Optional[np.ndarray] = None
        self._thread_running = False

        # Stderr suppression state (external cameras only)
        self._stderr_suppressed = False
        self._old_stderr_fd = None
        self._devnull_fd = None

    def _suppress_stderr(self):
        """Redirects OS-level stderr to devnull to suppress FFmpeg logs."""
        if not self._stderr_suppressed:
            try:
                self._old_stderr_fd = os.dup(2)
                self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self._devnull_fd, 2)
                self._stderr_suppressed = True
            except Exception as e:
                pass

    def _restore_stderr(self):
        """Restores OS-level stderr to its original state."""
        if self._stderr_suppressed:
            try:
                os.dup2(self._old_stderr_fd, 2)
                os.close(self._old_stderr_fd)
                os.close(self._devnull_fd)
                self._stderr_suppressed = False
            except Exception as e:
                pass

    @property
    def is_active(self) -> bool:
        """True if the camera is functional (either OpenCV or snapshot mode)."""
        if self.is_droidcam and self.droidcam_session is not None:
            return True
        if self.cap is not None and self.cap.isOpened():
            return True
        return False

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
                # Try snapshot mode first (DroidCam, IP Webcam) — avoids MJPEG issues
                if self._try_snapshot_connection():
                    return True
            else:
                print(f"[INFO] Opening local camera index: {self.camera_source}")

            if not self.is_external and os.name == 'nt':
                # Try DirectShow first (avoids MSMF crashes with some webcams)
                self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback: try default backend if DSHOW failed
                    print("[INFO] DirectShow failed, trying default backend...")
                    self.cap = cv2.VideoCapture(self.camera_source)
            else:
                # For external cameras, redirect stderr ONCE to suppress
                # FFmpeg mpjpeg warnings for the entire camera session
                self._suppress_stderr()
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

            # Test reading a frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print("[ERROR] Camera opened but cannot read frames.")
                self.cap.release()
                self.cap = None
                self._restore_stderr()
                return False

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"[INFO] Camera opened successfully. Resolution: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

            # Start the threaded frame grabber for smooth video
            self._start_frame_thread()
            return True
        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def _try_snapshot_connection(self) -> bool:
        """Try to connect using a JPEG snapshot endpoint (DroidCam, IP Webcam).

        Instead of parsing MJPEG streams (which OpenCV/FFmpeg often fails at),
        this fetches individual /shot.jpg snapshots — much more reliable.
        Uses urllib (stdlib, always available) as primary HTTP client.
        """
        import re
        from urllib.request import urlopen
        from urllib.error import URLError

        # Derive the base URL (strip any path like /video, /shot.jpg, etc.)
        source = self.camera_source
        match = re.match(r'(https?://[^/]+)', source)
        if not match:
            print(f"[CAMERA] Cannot extract base URL from: {source}")
            return False
        base_url = match.group(1)

        # Snapshot endpoints to try (most common first)
        snapshot_paths = [
            "/shot.jpg",         # DroidCam & IP Webcam
            "/photo.jpg",        # Some IP Webcam versions
            "/capture",          # Alternative
            "/snap.jpg",         # Alternative
        ]

        for path in snapshot_paths:
            url = base_url + path
            try:
                print(f"[CAMERA] Trying snapshot: {url} ...", end=" ")
                resp = urlopen(url, timeout=3)
                data = resp.read()
                print(f"status={resp.status}, bytes={len(data)}")

                if resp.status == 200 and len(data) > 1000:
                    # Verify it's a valid JPEG by decoding
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        self._snapshot_url = url
                        self.is_droidcam = True
                        self.droidcam_session = True  # flag — we use urllib
                        print(f"[CAMERA] ✓ Snapshot mode active: {url} "
                              f"({frame.shape[1]}x{frame.shape[0]})")
                        self._start_frame_thread()
                        return True
                    else:
                        print(f"[CAMERA]   ✗ {path}: got data but cv2.imdecode failed")
                else:
                    print(f"[CAMERA]   ✗ {path}: status={resp.status}, too small ({len(data)} bytes)")
            except URLError as e:
                print(f"URLError: {e.reason}")
            except Exception as e:
                print(f"Error: {e}")
                continue

        self.is_droidcam = False
        self.droidcam_session = None
        return False

    def _read_droidcam_frame(self) -> Optional[np.ndarray]:
        """Fetch a single JPEG snapshot from the camera's /shot.jpg endpoint."""
        try:
            url = getattr(self, '_snapshot_url', None)
            if not url or not self.droidcam_session:
                return None

            from urllib.request import urlopen
            resp = urlopen(url, timeout=2)
            data = resp.read()
            if resp.status != 200 or len(data) < 500:
                return None

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    @classmethod
    def try_connect_external(cls, ip: str, port: str) -> Optional['CameraCapture']:
        """Try to connect to an external camera using multiple methods.

        Priority:
          1. Snapshot mode (/shot.jpg) — fast, no MJPEG parsing, works with DroidCam
          2. OpenCV VideoCapture with various URLs — for RTSP and other cameras
        """
        print(f"[INFO] Checking if {ip}:{port} is reachable...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((ip, int(port)))
            s.close()
        except Exception as e:
            print(f"[ERROR] Host {ip}:{port} is unreachable. Fast-failing. ({e})")
            return None

        print(f"[INFO] Host is up.")

        # ── 1. Try snapshot mode first (fastest, most reliable for DroidCam) ──
        print(f"[INFO] Trying snapshot mode for {ip}:{port}...")
        cam = cls(camera_source=f"http://{ip}:{port}/shot.jpg")
        if cam._try_snapshot_connection():
            print(f"  ✓ Snapshot mode connected!")
            return cam

        # ── 2. Fall back to OpenCV VideoCapture URLs ──
        urls = cls.build_external_urls(ip, port)
        print(f"[INFO] Snapshot failed. Trying {len(urls)} OpenCV URL formats...")

        for url in urls:
            print(f"  Trying: {url}")
            cam = cls(camera_source=url)
            if cam.open():
                print(f"  ✓ Connected: {url}")
                return cam
            if cam.cap:
                cam.cap.release()
                cam.cap = None

        print(f"[ERROR] All connection methods failed for {ip}:{port}")
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

    # ─── Threaded Frame Grabber ────────────────────────────────────────────

    def _start_frame_thread(self) -> None:
        """Start the background thread that continuously grabs frames."""
        if self._thread_running:
            return
        self._thread_running = True
        self._frame_thread = threading.Thread(target=self._frame_grabber_loop, daemon=True)
        self._frame_thread.start()
        print("[CAMERA] Threaded frame grabber started.")

    def _stop_frame_thread(self) -> None:
        """Stop the background frame grabber thread."""
        self._thread_running = False
        if self._frame_thread is not None:
            self._frame_thread.join(timeout=2.0)
            self._frame_thread = None
        with self._frame_lock:
            self._latest_frame = None
        print("[CAMERA] Threaded frame grabber stopped.")

    def _frame_grabber_loop(self) -> None:
        """Background loop: continuously reads frames from the camera hardware.
        
        This runs in its own thread so that read_frame() never blocks the GUI.
        The loop targets ~30fps with adaptive sleep.
        """
        while self._thread_running:
            try:
                frame = self._read_frame_raw()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
            except Exception as e:
                print(f"[CAMERA] Frame grabber error: {e}")
            # ~30fps target: sleep just enough to yield CPU
            import time
            time.sleep(0.005)

    def _read_frame_raw(self) -> Optional[np.ndarray]:
        """Low-level frame read from hardware. Called only by the grabber thread."""
        if self.is_droidcam:
            frame = self._read_droidcam_frame()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                return frame
            return None

        if self.cap is None or not self.cap.isOpened():
            return None

        # stderr is already redirected at camera open time for external cams
        if self.is_external:
            # IP cameras over HTTP MJPEG can break if you grab() before read()
            ret, frame = self.cap.read()
        else:
            # Local cameras: flush buffer with a grab() then read() to reduce latency
            self.cap.grab()
            ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        # Force C-contiguous layout so downstream dlib calls never hit bad memory
        frame = np.ascontiguousarray(frame)

        # Mirror the frame (Left-to-Right inversion)
        frame = cv2.flip(frame, 1)
        return frame

    def read_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame from the threaded grabber.
        
        This method is NON-BLOCKING — it returns instantly with a copy of
        the most recent frame. If the grabber thread hasn't produced a frame
        yet, falls back to a direct synchronous read.
        """
        # Fast path: return cached frame from the background thread
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()

        # Fallback: thread not running yet, do a direct read
        return self._read_frame_raw()

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect a face in the frame using face_recognition (dlib).
        Returns: Tuple of (frame, (x, y, w, h)) if face detected, None otherwise.
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # We process a highly compressed version internally to be super fast
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Acquire lock — dlib C++ is not thread-safe
            with _dlib_lock:
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
        """Stop the background thread and release the camera hardware."""
        self._stop_frame_thread()

        if self.droidcam_session is not None:
            try:
                self.droidcam_session.close()
            except:
                pass
            self.droidcam_session = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._restore_stderr()
        print(f"[INFO] Camera released ({self.camera_source})")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Headless OpenCV build


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
