"""
Fall Detection Module (heuristic, experimental)
==================================================
Flags a *possible* fall by watching for a rapid collapse in a detected
person's bounding-box aspect ratio - tall-and-narrow while standing,
short-and-wide once lying down - within a short time window.

This is a coarse, heuristic proxy built only from 2D bounding boxes,
NOT a validated fall-detection system. Real ones typically use pose
estimation, wearable accelerometers, or multi-camera setups; this is an
experimental accessibility signal only. Both false positives (e.g.
sitting down quickly, bending over) and false negatives (e.g. a fall
that isn't the largest detected person, or happens off-frame) are
expected - it should prompt a spoken check-in, never an automatic
emergency action.
"""
import time


class FallDetector:
    def __init__(self, window_seconds=1.5, collapse_ratio=0.5, min_upright_ratio=1.2):
        self.window_seconds = window_seconds
        self.collapse_ratio = collapse_ratio
        self.min_upright_ratio = min_upright_ratio
        self._history = []  # list of (timestamp, height, width)

    def update(self, person_bbox):
        """Feed the current frame's largest "person" bounding box (or
        None if no person is currently detected). Returns True the
        moment a possible fall is detected - fires once per event, then
        resets, rather than repeatedly for the same collapse."""
        now = time.time()
        if person_bbox is None:
            self._prune(now)
            return False

        x1, y1, x2, y2 = person_bbox
        height = max(1, y2 - y1)
        width = max(1, x2 - x1)
        self._history.append((now, height, width))
        self._prune(now)

        if len(self._history) < 2:
            return False

        _, oldest_h, oldest_w = self._history[0]
        _, latest_h, latest_w = self._history[-1]

        was_upright = (oldest_h / oldest_w) >= self.min_upright_ratio
        height_collapsed = latest_h <= oldest_h * self.collapse_ratio
        now_wide = (latest_w / latest_h) >= 1.0

        if was_upright and height_collapsed and now_wide:
            self._history.clear()  # don't keep firing for the same event
            return True
        return False

    def _prune(self, now):
        cutoff = now - self.window_seconds
        self._history = [h for h in self._history if h[0] >= cutoff]
