"""Unit tests for the heuristic navigation module - pure logic, no
camera or model weights required, so these run anywhere instantly."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.navigation import NavigationAssistant


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


if __name__ == "__main__":
    unittest.main()
