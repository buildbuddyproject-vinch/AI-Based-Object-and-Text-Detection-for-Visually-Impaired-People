"""
Currency Detector Module (bonus feature)
===========================================
Lightweight, heuristic Indian-currency-note detector based on ORB feature
matching against reference note images. This is NOT a trained deep
learning model - it is included as an extensible starting point for a
bonus accessibility feature.

To make it actually recognize notes, place clear, front-facing reference
images of each denomination in:
    dataset/currency/<value>.jpg      (e.g. dataset/currency/500.jpg)

Without reference images, `is_ready` is False and `detect()` always
returns None - the caller falls back to a friendly "not recognized"
message instead of crashing.

For production-grade accuracy, replace this module with a properly
trained classification model (e.g. a small CNN, or a YOLO classification
head) trained on a labeled currency-note dataset.
"""
import os
import re

import cv2


def _denomination_from_filename(fname):
    """Turn a reference image filename into a clean numeric denomination
    label, e.g. "100rs.png" / "100_rupees.jpg" / "100.png" -> "100".
    Falls back to the raw filename (minus extension) if it doesn't start
    with digits, so unusual naming still works, just less tidily."""
    raw_label = os.path.splitext(fname)[0]
    match = re.match(r"^\s*(\d+)", raw_label)
    return match.group(1) if match else raw_label


class CurrencyDetector:
    def __init__(self, reference_dir="dataset/currency"):
        self.reference_dir = reference_dir
        self.orb = cv2.ORB_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._references = self._load_references()

    def _load_references(self):
        refs = []
        if not os.path.isdir(self.reference_dir):
            return refs
        for fname in sorted(os.listdir(self.reference_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            label = _denomination_from_filename(fname)
            path = os.path.join(self.reference_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            kp, des = self.orb.detectAndCompute(img, None)
            if des is not None:
                refs.append({"label": label, "kp": kp, "des": des})
        return refs

    @property
    def is_ready(self):
        return len(self._references) > 0

    def detect(self, frame, min_good_matches=15):
        """Return {"denomination": str, "confidence": int} for the best
        matching reference note, or None if nothing matched well enough
        (or no reference images were provided)."""
        if not self.is_ready or frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return None

        best_label, best_score = None, 0
        for ref in self._references:
            matches = self.matcher.match(des, ref["des"])
            good = [m for m in matches if m.distance < 60]
            if len(good) > best_score:
                best_score = len(good)
                best_label = ref["label"]

        if best_label and best_score >= min_good_matches:
            return {"denomination": best_label, "confidence": best_score}
        return None
