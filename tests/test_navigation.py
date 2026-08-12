"""Unit tests for the heuristic navigation module - pure logic, no
camera or model weights required, so these run anywhere instantly."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.navigation import NavigationAssistant, footpath_walkability


class TestNavigationAssistant(unittest.TestCase):
    def setUp(self):
        self.nav = NavigationAssistant(near_area_ratio=0.15)

    def test_no_detections_returns_clear_path(self):
        instruction, zones = self.nav.analyze([], 640, 480)
        self.assertIn("clear", instruction.lower())
        self.assertEqual(zones, {"left": [], "center": [], "right": []})

    def test_object_on_left(self):
        detections = [{"label": "chair", "confidence": 0.9, "bbox": [0, 0, 50, 100]}]
        instruction, zones = self.nav.analyze(detections, 640, 480)
        self.assertIn("left", instruction.lower())
        self.assertIn("chair", zones["left"])

    def test_object_on_right(self):
        detections = [{"label": "door", "confidence": 0.9, "bbox": [600, 0, 640, 480]}]
        instruction, zones = self.nav.analyze(detections, 640, 480)
        self.assertIn("right", instruction.lower())
        self.assertIn("door", zones["right"])

    def test_close_center_object_suggests_move(self):
        detections = [{"label": "person", "confidence": 0.9, "bbox": [220, 0, 420, 480]}]
        instruction, zones = self.nav.analyze(detections, 640, 480)
        self.assertIn("move slightly", instruction.lower())
        self.assertIn("person", zones["center"])

    def test_far_center_object_has_no_move_suggestion(self):
        detections = [{"label": "cup", "confidence": 0.9, "bbox": [300, 200, 320, 220]}]
        instruction, _ = self.nav.analyze(detections, 640, 480)
        self.assertNotIn("move slightly", instruction.lower())
        self.assertIn("ahead", instruction.lower())

    def test_depth_map_flags_nearby_object_as_very_close(self):
        # Depth map where the object's region is much "nearer" (higher
        # value) than the rest of the frame.
        depth_map = np.full((480, 640), 100.0, dtype=np.float32)
        depth_map[0:100, 220:420] = 1000.0
        detections = [{"label": "person", "confidence": 0.9, "bbox": [220, 0, 420, 100]}]
        instruction, _ = self.nav.analyze(detections, 640, 480, depth_map=depth_map)
        self.assertIn("very close", instruction.lower())

    def test_depth_map_never_states_a_metric_distance(self):
        # Whatever wording comes out, it must never fabricate a number
        # of metres - only relative depth is available, never a
        # calibrated measurement.
        depth_map = np.random.uniform(50, 500, (480, 640)).astype(np.float32)
        detections = [{"label": "chair", "confidence": 0.9, "bbox": [100, 100, 300, 300]}]
        instruction, _ = self.nav.analyze(detections, 640, 480, depth_map=depth_map)
        self.assertNotIn("metre", instruction.lower())
        self.assertNotIn("meter", instruction.lower())


class TestPhraseFor(unittest.TestCase):
    """Section 5: navigation-first situational phrasing for a single
    already-chosen object (temporal confirmation + priority engine pick
    WHO; this only decides HOW to phrase it)."""

    def test_center_zone_says_ahead(self):
        self.assertEqual(NavigationAssistant.phrase_for("chair", "center"), "Chair ahead.")

    def test_left_zone(self):
        self.assertEqual(NavigationAssistant.phrase_for("person", "left"), "Person on your left.")

    def test_right_zone(self):
        self.assertEqual(NavigationAssistant.phrase_for("pole", "right"), "Pole on your right.")

    def test_is_close_appends_very_close(self):
        phrase = NavigationAssistant.phrase_for("door", "center", is_close=True)
        self.assertIn("very close", phrase.lower())

    def test_critical_tier_prefixes_warning(self):
        phrase = NavigationAssistant.phrase_for("car", "center", tier="CRITICAL")
        self.assertTrue(phrase.startswith("Warning."))

    def test_non_critical_tier_never_says_warning(self):
        phrase = NavigationAssistant.phrase_for("cup", "center", tier="LOW")
        self.assertNotIn("Warning", phrase)

    def test_blocked_path_mentioned_explicitly(self):
        phrase = NavigationAssistant.phrase_for("stairs", "center", blocked=True)
        self.assertIn("Path blocked.", phrase)

    def test_never_claims_absolute_safety(self):
        # Section 15: never "100% safe" / "safe to cross" / "perfect
        # detection" - spot-check across a range of inputs.
        for zone in ("left", "center", "right"):
            for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW", None):
                phrase = NavigationAssistant.phrase_for("person", zone, tier=tier)
                lowered = phrase.lower()
                self.assertNotIn("100%", lowered)
                self.assertNotIn("safe to cross", lowered)
                self.assertNotIn("perfectly safe", lowered)


class TestFootpathWalkability(unittest.TestCase):
    def test_no_detections_nothing_walkable(self):
        result = footpath_walkability([], 900)
        self.assertEqual(result, {"left": False, "center": False, "right": False})

    def test_ignores_non_footpath_labels(self):
        dets = [{"label": "person", "bbox": [0, 0, 900, 480]}]
        result = footpath_walkability(dets, 900)
        self.assertFalse(any(result.values()))

    def test_full_width_footpath_marks_all_zones_walkable(self):
        dets = [{"label": "footpath", "bbox": [0, 200, 900, 480]}]
        result = footpath_walkability(dets, 900)
        self.assertTrue(result["left"])
        self.assertTrue(result["center"])
        self.assertTrue(result["right"])

    def test_footpath_confined_to_left_third_only(self):
        # frame_width=900 -> thirds at [0,300) [300,600) [600,900)
        dets = [{"label": "footpath", "bbox": [0, 200, 280, 480]}]
        result = footpath_walkability(dets, 900)
        self.assertTrue(result["left"])
        self.assertFalse(result["center"])
        self.assertFalse(result["right"])

    def test_tiny_sliver_below_threshold_does_not_count(self):
        # Only ~10px of overlap into the center third (< 25% of 300px) -
        # too small to count as "the path covers this zone".
        dets = [{"label": "footpath", "bbox": [0, 200, 310, 480]}]
        result = footpath_walkability(dets, 900)
        self.assertTrue(result["left"])
        self.assertFalse(result["center"])


if __name__ == "__main__":
    unittest.main()
