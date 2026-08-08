"""
Color Detector Module (bonus feature)
========================================
Estimates the dominant color inside a bounding box (or the whole frame)
and maps it to a human-readable color name for accessibility narration,
e.g. "blue chair on your left".
"""
import cv2
import numpy as np

_COLOR_NAMES = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (139, 69, 19),
}


def _closest_color_name(rgb):
    r, g, b = rgb
    best_name, best_dist = "unknown", float("inf")
    for name, (cr, cg, cb) in _COLOR_NAMES.items():
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def detect_dominant_color(frame, bbox=None):
    """Return the name of the dominant color within `bbox`
    (x1, y1, x2, y2) of `frame`, or of the whole frame if bbox is None."""
    if frame is None or frame.size == 0:
        return "unknown"

    if bbox:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        region = frame[y1:y2, x1:x2]
    else:
        region = frame

    if region.size == 0:
        return "unknown"

    small = cv2.resize(region, (30, 30), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    # K-means (k=3) finds the dominant cluster, which is a better
    # approximation of "the object's color" than a plain average when the
    # box also contains background/shadow pixels.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3 if len(pixels) >= 3 else 1
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant_bgr = centers[np.argmax(counts)]
    b, g, r = dominant_bgr
    return _closest_color_name((r, g, b))
