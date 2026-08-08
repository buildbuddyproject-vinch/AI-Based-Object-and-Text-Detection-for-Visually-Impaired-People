"""Unit tests for the dominant color detector - pure OpenCV/numpy logic,
no camera required."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.color_detector import detect_dominant_color


class TestColorDetector(unittest.TestCase):
    def test_solid_red_frame(self):
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)  # OpenCV uses BGR
        self.assertEqual(detect_dominant_color(frame), "red")

    def test_solid_blue_frame(self):
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)  # BGR blue
        self.assertEqual(detect_dominant_color(frame), "blue")

    def test_bbox_region_is_used(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :50] = (0, 0, 255)   # left half red (BGR)
        frame[:, 50:] = (0, 255, 0)   # right half green (BGR)
        self.assertEqual(detect_dominant_color(frame, bbox=[60, 0, 100, 100]), "green")

    def test_empty_frame_returns_unknown(self):
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(detect_dominant_color(frame), "unknown")


if __name__ == "__main__":
    unittest.main()
