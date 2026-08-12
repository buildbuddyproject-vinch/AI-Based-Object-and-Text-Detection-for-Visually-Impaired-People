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

    def test_class_stability_survives_a_wobbling_classification(self):
        # Reproduces the live-testing bug: the same physical object
        # (fixed position) alternates between two visually similar
        # trained classes frame to frame. Position-based (IoU) matching
        # should keep this as ONE track whose stable_label reflects the
        # majority-voted class, instead of splitting into two separate
        # tracks that each independently confirm and then flip-flop.
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.3)
        labels = ["door", "refrigeratorDoor", "door", "door", "refrigeratorDoor", "door"]
        confirmed = []
        for i, label in enumerate(labels):
            confirmed = tracker.update([{"label": label, "bbox": BOX, "confidence": 0.7}], now=float(i))
        self.assertEqual(len(confirmed), 1)  # one track, not two
        self.assertEqual(confirmed[0].stable_label, "door")  # 4 of 6 frames

    def test_class_stability_ties_broken_by_recency(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.3)
        confirmed = []
        for i, label in enumerate(["door", "refrigeratorDoor"]):
            confirmed = tracker.update([{"label": label, "bbox": BOX, "confidence": 0.7}], now=float(i))
        # 1-1 tie -> most recent wins.
        self.assertEqual(confirmed[0].stable_label, "refrigeratorDoor")

    def test_single_frame_misclassification_does_not_flip_a_stable_track(self):
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.3)
        confirmed = []
        for i, label in enumerate(["chair", "chair", "chair", "table", "chair"]):
            confirmed = tracker.update([{"label": label, "bbox": BOX, "confidence": 0.7}], now=float(i))
        self.assertEqual(confirmed[0].stable_label, "chair")  # 4 of 5, the one-off "table" is outvoted

    def test_low_confidence_wobble_still_never_matches_by_iou_alone(self):
        # A below-floor detection must still never be eligible to match
        # (and thus never contribute a vote to) any track, regardless of
        # this change removing the same-label requirement.
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.3)
        tracker.update([{"label": "door", "bbox": BOX, "confidence": 0.7}], now=0.0)
        confirmed = tracker.update([{"label": "airplane", "bbox": BOX, "confidence": 0.2}], now=0.1)
        self.assertEqual(confirmed[0].stable_label, "door")  # unaffected by the low-confidence "airplane"

    def test_stable_zone_smooths_a_boundary_straddling_object(self):
        # Same jittery sequence as the stable_bbox boundary test, but
        # voting on the discrete zone decision itself. thirds of a
        # 300-wide frame: left [0,100) center [100,200) right [200,300).
        def zone_of(bbox):
            cx = (bbox[0] + bbox[2]) / 2
            if cx < 100:
                return "left"
            if cx > 200:
                return "right"
            return "center"

        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.1)
        # Center-x hovers right around the left/center boundary (100):
        # 95(left), 105(center), 103(center), 107(center), 99(left) ->
        # 3 center, 2 left - center wins even though the raw sequence
        # starts and ends on "left".
        boxes = [
            [90, 0, 100, 10], [100, 0, 110, 10], [98, 0, 108, 10],
            [102, 0, 112, 10], [94, 0, 104, 10],
        ]
        confirmed = []
        for i, box in enumerate(boxes):
            confirmed = tracker.update([{"label": "door", "bbox": box, "confidence": 0.8}], now=float(i))
        track = confirmed[0]
        self.assertEqual(track.stable_zone(zone_of), "center")  # majority of the 5 frames

    def test_stable_zone_ties_broken_by_recency(self):
        def zone_of(bbox):
            return "left" if bbox[0] < 50 else "right"

        # Overlapping boxes (so IoU matching keeps this as ONE track)
        # whose left edge straddles the zone_of boundary (50).
        tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=1, iou_match_threshold=0.3)
        confirmed = []
        for i, x in enumerate([10, 60]):
            confirmed = tracker.update(
                [{"label": "door", "bbox": [x, 0, x + 100, 50], "confidence": 0.8}], now=float(i)
            )
        self.assertEqual(len(confirmed), 1)  # sanity: stayed one track, not two
        self.assertEqual(confirmed[0].stable_zone(zone_of), "right")  # 1-1 tie -> most recent

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
