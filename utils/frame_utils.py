"""Small helpers for frame resizing, JPEG encoding, and FPS tracking -
used to keep the live video stream smooth and to support the battery
saver / performance mode."""
import time

import cv2


def resize_frame(frame, scale=1.0):
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h))


def encode_jpeg(frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def is_low_light(frame, threshold=60):
    """Return True if `frame`'s average brightness is below `threshold`
    (0-255 grayscale scale) - a cheap heuristic proxy for "too dark for
    reliable detection", needing no model at all. YOLO/OCR accuracy
    both degrade in low light, so this is used to warn the user rather
    than to silently produce unreliable results."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


class FPSTracker:
    """Exponentially-smoothed FPS counter - call `tick()` once per
    processed frame."""

    def __init__(self, smoothing=0.9):
        self._last = time.time()
        self.fps = 0.0
        self.smoothing = smoothing

    def tick(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        return round(self.fps, 1)
