"""Unit tests for the template-based scene summary - pure logic, no
camera or model weights required."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.scene_summary import build_scene_summary


class TestSceneSummary(unittest.TestCase):
    def test_no_detections(self):
        summary = build_scene_summary([])
        self.assertIn("don't see", summary.lower())

    def test_single_object_with_zone(self):
        detections = [{"label": "chair", "confidence": 0.9, "bbox": [0, 0, 50, 100]}]
        zone_map = {"left": ["chair"], "center": [], "right": []}
        summary = build_scene_summary(detections, zone_map)
        self.assertIn("chair", summary.lower())
        self.assertIn("left", summary.lower())

    def test_multiple_of_same_label_are_counted(self):
        detections = [
            {"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"label": "person", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
        ]
        summary = build_scene_summary(detections)
        self.assertIn("2 persons", summary.lower())

    def test_without_zone_map_defaults_to_nearby(self):
        detections = [{"label": "cup", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
        summary = build_scene_summary(detections)
        self.assertIn("nearby", summary.lower())

    def test_max_items_caps_and_notes_remainder(self):
        detections = [
            {"label": f"item{i}", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
            for i in range(6)
        ]
        summary = build_scene_summary(detections, max_items=4)
        self.assertIn("other thing", summary.lower())


if __name__ == "__main__":
    unittest.main()
