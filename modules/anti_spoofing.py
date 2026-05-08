"""
anti_spoofing.py — NEW Layer 1: Presentation Attack Detection (PAD).

Implements texture-based liveness detection using:
1. Laplacian variance (texture sharpness — real faces have more micro-texture)
2. Local Binary Pattern (LBP) histogram analysis for texture classification
3. Brightness histogram uniformity check (screen glow detection)

Since Silent-Face-Anti-Spoofing is not a stable PyPI package, we implement
robust texture-based PAD using OpenCV directly. If a pre-trained MiniFASNet
model is placed in models/anti_spoof_model/, it will be loaded and used.
"""

import os
import sys
import cv2
import numpy as np
from typing import Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class AntiSpoofing:
    """Texture-based liveness / presentation attack detection."""

    def __init__(self) -> None:
        self.model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "anti_spoof_model"
        )
        self.model = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load a pre-trained anti-spoofing DNN model if available."""
        onnx_path = os.path.join(self.model_path, "anti_spoof.onnx")
        if os.path.exists(onnx_path):
            try:
                self.model = cv2.dnn.readNetFromONNX(onnx_path)
                print("[INFO] Loaded pre-trained anti-spoofing model.")
            except Exception as e:
                print(f"[WARN] Could not load anti-spoof model: {e}. Using texture analysis.")
        else:
            print("[INFO] No pre-trained anti-spoof model found. Using texture-based analysis.")

    def _compute_laplacian_variance(self, face_crop: np.ndarray) -> float:
        """
        Compute the variance of the Laplacian — measures texture/sharpness.
        Real faces have higher texture variance than printed photos or screens.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _compute_lbp_histogram(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Compute a Local Binary Pattern histogram for texture analysis.
        Real faces produce different LBP distributions than flat printouts.

        Uses fully vectorized NumPy operations instead of pixel-level Python
        loops for ~100x speedup.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        center = gray[1:-1, 1:-1]

        # Vectorized LBP: compare each neighbor against the center pixel
        lbp = np.zeros_like(center, dtype=np.uint8)
        lbp |= ((gray[0:-2, 0:-2] >= center).astype(np.uint8) << 7)  # top-left
        lbp |= ((gray[0:-2, 1:-1] >= center).astype(np.uint8) << 6)  # top
        lbp |= ((gray[0:-2, 2:]   >= center).astype(np.uint8) << 5)  # top-right
        lbp |= ((gray[1:-1, 2:]   >= center).astype(np.uint8) << 4)  # right
        lbp |= ((gray[2:,   2:]   >= center).astype(np.uint8) << 3)  # bottom-right
        lbp |= ((gray[2:,   1:-1] >= center).astype(np.uint8) << 2)  # bottom
        lbp |= ((gray[2:,   0:-2] >= center).astype(np.uint8) << 1)  # bottom-left
        lbp |= ((gray[1:-1, 0:-2] >= center).astype(np.uint8) << 0)  # left

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist

    def _check_brightness_uniformity(self, face_crop: np.ndarray) -> float:
        """
        Check for screen glow artifacts by analyzing brightness uniformity.
        Screens produce characteristic luminance uniformity — real faces don't.

        Returns a uniformity score: lower = more uniform (more likely screen).
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Divide face into a 4x4 grid and compute mean brightness per cell
        h, w = gray.shape
        cell_h, cell_w = h // 4, w // 4
        means = []
        for i in range(4):
            for j in range(4):
                cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                means.append(np.mean(cell))

        # Standard deviation of cell means — low = uniform = likely screen
        return float(np.std(means))

    def _check_color_distribution(self, face_crop: np.ndarray) -> float:
        """
        Analyze color channel distribution. Screens have more saturated,
        uniform color distributions compared to natural skin.

        Returns: score where higher = more natural.
        """
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        # Check saturation variance — natural skin has more variation
        sat_var = float(np.var(hsv[:, :, 1]))
        # Check value (brightness) variance
        val_var = float(np.var(hsv[:, :, 2]))
        return (sat_var + val_var) / 2.0

    def _model_predict(self, face_crop: np.ndarray) -> float:
        """Use loaded DNN model to predict liveness probability."""
        if self.model is None:
            return -1.0

        blob = cv2.dnn.blobFromImage(
            face_crop, scalefactor=1.0 / 255.0,
            size=(80, 80), mean=(0, 0, 0), swapRB=True
        )
        self.model.setInput(blob)
        output = self.model.forward()
        # Assume output is [batch, 2] with [fake_prob, real_prob]
        if output.shape[-1] >= 2:
            return float(output[0][1])
        return float(output[0][0])

    def check_liveness(self, frame: np.ndarray,
                       face_bbox: Tuple[int, int, int, int]) -> Tuple[bool, float, str]:
        """
        Run the full anti-spoofing check on a detected face.

        Args:
            frame: Full BGR frame from camera.
            face_bbox: (x, y, w, h) bounding box of the detected face.

        Returns:
            (is_real: bool, confidence: float, reason: str)
        """
        x, y, w, h = face_bbox
        # Pad the crop slightly for better analysis
        pad = int(min(w, h) * 0.1)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return False, 0.0, "Empty face crop"

        # Resize for consistent analysis
        face_resized = cv2.resize(face_crop, (160, 160))

        # ── Check 1: Laplacian texture variance ──
        lap_var = self._compute_laplacian_variance(face_resized)
        texture_ok = lap_var >= config.LAPLACIAN_VARIANCE_THRESHOLD

        # ── Check 2: Brightness uniformity (screen glow) ──
        brightness_std = self._check_brightness_uniformity(face_resized)
        no_screen_glow = brightness_std >= config.BRIGHTNESS_UNIFORMITY_THRESHOLD * 100

        # ── Check 3: Color distribution ──
        color_score = self._check_color_distribution(face_resized)
        color_ok = color_score > 200  # Empirical threshold for natural skin variation

        # ── Check 4: DNN model (if available) ──
        model_score = self._model_predict(face_resized)
        if model_score >= 0:
            model_ok = model_score >= config.PAD_REAL_THRESHOLD
        else:
            model_ok = True  # No model available, skip this check

        # ── Combine signals ──
        # Texture must pass AND (no screen glow OR color distribution OK)
        is_real = texture_ok and (no_screen_glow or color_ok) and model_ok

        # Compute aggregate confidence
        confidence_components = []
        if texture_ok:
            confidence_components.append(min(lap_var / (config.LAPLACIAN_VARIANCE_THRESHOLD * 3), 1.0))
        else:
            confidence_components.append(lap_var / (config.LAPLACIAN_VARIANCE_THRESHOLD + 1e-7) * 0.5)

        if no_screen_glow:
            confidence_components.append(0.9)
        else:
            confidence_components.append(0.3)

        if model_score >= 0:
            confidence_components.append(model_score)

        confidence = float(np.mean(confidence_components))

        reason_parts = []
        if not texture_ok:
            reason_parts.append(f"Low texture variance ({lap_var:.1f} < {config.LAPLACIAN_VARIANCE_THRESHOLD})")
        if not no_screen_glow:
            reason_parts.append(f"Screen glow detected (brightness_std={brightness_std:.1f})")
        if not model_ok:
            reason_parts.append(f"DNN model rejected (score={model_score:.2f})")
        if not color_ok:
            reason_parts.append(f"Unnatural color distribution (score={color_score:.1f})")

        reason = "; ".join(reason_parts) if reason_parts else "All checks passed"

        return is_real, confidence, reason


if __name__ == "__main__":
    print("Anti-Spoofing Module — Test")
    pad = AntiSpoofing()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    print("Press 'c' to check liveness, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Anti-Spoofing Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Simple face detection for test using OpenCV cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                bbox = (x, y, w, h)
                is_real, conf, reason = pad.check_liveness(frame, bbox)
                print(f"  Real: {is_real} | Confidence: {conf:.3f} | Reason: {reason}")
            else:
                print("  No face detected.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
