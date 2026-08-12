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

    def test_stable_bbox_smooths_jitter_around_a_boundary(self):
        # Reproduces the exact live-testing bug: a stationary object's
        # raw per-frame bbox jitters just enough to cross a zone
        # boundary, flipping "on your left" / "ahead" / "on your right"
        # every few seconds even though nothing actually moved.
        # stable_bbox (a rolling mean) should stay put near the true
        # center instead of tracking each frame's noisy reading.
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.1)
        jittery_boxes = [
            [100, 100, 200, 300],  # center x = 150
            [95, 100, 195, 300],   # center x = 145
            [110, 100, 210, 300],  # center x = 160
            [90, 100, 190, 300],   # center x = 140
            [105, 100, 205, 300],  # center x = 155
        ]
        confirmed = []
        for i, box in enumerate(jittery_boxes):
            confirmed = tracker.update([{"label": "door", "bbox": box, "confidence": 0.8}], now=float(i))
        track = confirmed[0]
        stable_cx = (track.stable_bbox[0] + track.stable_bbox[2]) / 2
        # Mean of the 5 center-x values above (150+145+160+140+155)/5 = 150
        self.assertAlmostEqual(stable_cx, 150.0, delta=1.0)
        # And it should vary far less than the raw latest-frame reading
        # would across this same jittery sequence.
        raw_cxs = [(b[0] + b[2]) / 2 for b in jittery_boxes]
        self.assertLess(max(raw_cxs) - min(raw_cxs), 25)  # sanity: the jitter is real
        self.assertAlmostEqual(stable_cx, sum(raw_cxs) / len(raw_cxs))

    def test_stable_bbox_only_uses_last_five_frames(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.1)
        confirmed = tracker.update([{"label": "chair", "bbox": [0, 0, 100, 100], "confidence": 0.8}], now=0.0)
        # 6 more updates with a very different box - after 5, the first
        # (very different) box should have rolled out of the window.
        for i in range(1, 7):
            confirmed = tracker.update(
                [{"label": "chair", "bbox": [500, 500, 600, 600], "confidence": 0.8}], now=float(i)
            )
        track = confirmed[0]
        stable_cx = (track.stable_bbox[0] + track.stable_bbox[2]) / 2
        self.assertAlmostEqual(stable_cx, 550.0)  # fully converged, old box rolled off


if __name__ == "__main__":
    unittest.main()
