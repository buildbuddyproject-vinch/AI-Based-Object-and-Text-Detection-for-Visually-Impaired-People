"""
Temporal Confirmation / Object Tracking Module
=================================================
Prevents single-frame detection noise (a misclassification that only
appears for one frame - e.g. a spinning ceiling fan momentarily
matching "airplane") from ever reaching the user. A detection must
appear consistently, in roughly the same place, for several consecutive
frames before it counts as "confirmed" and becomes eligible for
announcement.

This is deliberately simple frame-to-frame IoU matching, not a full
multi-object tracker (no Kalman filter, no Hungarian assignment, no
persistent track IDs across occlusion) - that complexity isn't needed
to solve the actual problem here ("one bad frame shouldn't get
announced"), and a simpler implementation is easier to trust for an
assistive/safety-adjacent system.
"""
import time


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TrackedObject:
    def __init__(self, label, bbox, confidence, now):
        self.label = label
        self.bbox = bbox
        self.confidence = confidence
        self.first_seen = now
        self.last_seen = now
        self.consecutive_frames = 1
        self._confidences = [confidence]

    def update(self, bbox, confidence, now):
        self.bbox = bbox
        self.confidence = confidence
        self._confidences = (self._confidences + [confidence])[-10:]
        self.last_seen = now
        self.consecutive_frames += 1

    @property
    def avg_confidence(self):
        return sum(self._confidences) / len(self._confidences)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class ObjectTracker:
    """Matches each frame's raw detections against recently-seen
    tracks and only exposes a detection as "confirmed" once it has
    appeared consistently for >= min_consecutive_frames.

    Example (matches the fan/airplane scenario from the spec):
        Frame 1: fan 0.70   -> tracked, 1 consecutive frame, NOT confirmed
        Frame 2: fan 0.78   -> matched, 2 consecutive frames, NOT confirmed
        Frame 3: fan 0.84   -> matched, 3 consecutive frames, NOT confirmed
        Frame 4: fan 0.86   -> matched, 4 consecutive frames, CONFIRMED -> "Fan ahead."

        Frame 1: airplane 0.51 -> tracked, 1 frame
        Frame 2: airplane 0.42 -> matched, 2 frames
        Frame 3: fan 0.31      -> below min_confidence, ignored entirely;
                                   airplane track goes unmatched this frame
        Frame 4: (nothing)     -> airplane track still unmatched
        -> airplane track never reaches min_consecutive_frames and is
           never announced.
    """

    def __init__(self, min_confidence=0.4, min_consecutive_frames=4,
                 iou_match_threshold=0.3, stale_after_seconds=1.5):
        self.min_confidence = min_confidence
        self.min_consecutive_frames = min_consecutive_frames
        self.iou_match_threshold = iou_match_threshold
        self.stale_after_seconds = stale_after_seconds
        self._tracks = []

    def update(self, detections, now=None):
        """Feed one frame's raw detections (list of dicts with at
        least "label", "bbox", "confidence"). Returns the list of
        CONFIRMED TrackedObjects, ready for the priority engine."""
        now = now if now is not None else time.time()

        # Age out tracks nothing has matched recently.
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.stale_after_seconds]

        unmatched = list(detections)
        for track in self._tracks:
            best_match, best_iou = None, 0.0
            for det in unmatched:
                if det["label"] != track.label or det["confidence"] < self.min_confidence:
                    continue
                iou = _iou(det["bbox"], track.bbox)
                if iou > best_iou:
                    best_iou, best_match = iou, det
            if best_match is not None and best_iou >= self.iou_match_threshold:
                track.update(best_match["bbox"], best_match["confidence"], now)
                unmatched.remove(best_match)

        # Leftover detections that clear the confidence floor start new
        # tracks; anything below the floor is ignored outright - it
        # never even gets a chance to accumulate consecutive frames.
        for det in unmatched:
            if det["confidence"] >= self.min_confidence:
                self._tracks.append(TrackedObject(det["label"], det["bbox"], det["confidence"], now))

        return [t for t in self._tracks if t.consecutive_frames >= self.min_consecutive_frames]

    def reset(self):
        self._tracks = []
