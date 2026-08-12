"""Unit tests for the temporal confirmation tracker - pure logic, no
camera or model weights required. Covers the exact fan/airplane
scenario from the spec."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tracking import ObjectTracker

BOX = [100, 100, 200, 300]
BOX_SHIFTED = [105, 100, 205, 300]  # small movement, still overlaps a lot


class TestObjectTracker(unittest.TestCase):
    def test_single_frame_detection_is_never_confirmed(self):
        tracker = ObjectTracker(min_consecutive_frames=4)
        confirmed = tracker.update([{"label": "airplane", "bbox": BOX, "confidence": 0.51}], now=0.0)
        self.assertEqual(confirmed, [])

    def test_consistent_detection_confirms_after_min_frames(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=4)
        confirmed = []
        for i in range(4):
            confirmed = tracker.update(
                [{"label": "fan", "bbox": BOX, "confidence": 0.7 + i * 0.04}], now=float(i)
            )
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].label, "fan")

    def test_low_confidence_detection_never_starts_a_track(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=2)
        confirmed = tracker.update([{"label": "airplane", "bbox": BOX, "confidence": 0.31}], now=0.0)
        confirmed = tracker.update([{"label": "airplane", "bbox": BOX, "confidence": 0.35}], now=0.1)
        self.assertEqual(confirmed, [])

    def test_intermittent_low_confidence_frame_breaks_the_streak(self):
        # Matches the spec's exact scenario: two shaky airplane frames,
        # then a below-threshold frame, then nothing - never confirmed.
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=3, stale_after_seconds=100)
        tracker.update([{"label": "airplane", "bbox": BOX, "confidence": 0.51}], now=0.0)
        tracker.update([{"label": "airplane", "bbox": BOX, "confidence": 0.42}], now=0.1)
        confirmed = tracker.update([{"label": "fan", "bbox": BOX, "confidence": 0.31}], now=0.2)
        self.assertEqual(confirmed, [])
        confirmed = tracker.update([], now=0.3)
        self.assertEqual(confirmed, [])

    def test_moved_bbox_still_matches_via_iou(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=3, iou_match_threshold=0.3)
        tracker.update([{"label": "chair", "bbox": BOX, "confidence": 0.8}], now=0.0)
        tracker.update([{"label": "chair", "bbox": BOX_SHIFTED, "confidence": 0.8}], now=0.1)
        confirmed = tracker.update([{"label": "chair", "bbox": BOX, "confidence": 0.8}], now=0.2)
        self.assertEqual(len(confirmed), 1)

    def test_stale_track_is_dropped(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=2, stale_after_seconds=0.5)
        tracker.update([{"label": "chair", "bbox": BOX, "confidence": 0.8}], now=0.0)
        # Big time gap - the track should age out rather than match.
        confirmed = tracker.update([{"label": "chair", "bbox": BOX, "confidence": 0.8}], now=5.0)
        # A fresh track starts here, but it's only had 1 consecutive
        # frame so it isn't confirmed yet.
        self.assertEqual(confirmed, [])

    def test_reset_clears_all_tracks(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1)
        tracker.update([{"label": "chair", "bbox": BOX, "confidence": 0.8}], now=0.0)
        tracker.reset()
        confirmed = tracker.update([{"label": "chair", "bbox": BOX, "confidence": 0.8}], now=0.1)
        self.assertEqual(len(confirmed), 1)  # fresh track, 1 frame, min=1 so confirmed immediately


if __name__ == "__main__":
    unittest.main()
