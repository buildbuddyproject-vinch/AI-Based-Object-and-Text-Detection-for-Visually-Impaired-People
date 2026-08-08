"""Unit tests for the heuristic fall detector - pure logic, no camera
or model weights required."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fall_detector import FallDetector


class TestFallDetector(unittest.TestCase):
    def test_no_person_never_fires(self):
        fd = FallDetector()
        for _ in range(10):
            self.assertFalse(fd.update(None))

    def test_standing_still_does_not_fire(self):
        fd = FallDetector()
        # A tall, narrow, unchanging bbox - upright and stable.
        for _ in range(5):
            self.assertFalse(fd.update([100, 50, 160, 400]))

    def test_sudden_collapse_fires(self):
        fd = FallDetector(window_seconds=5.0)
        # Standing: tall and narrow.
        fd.update([100, 50, 160, 400])
        # Collapsed: short and wide, well within the time window.
        result = fd.update([80, 350, 260, 400])
        self.assertTrue(result)

    def test_fires_only_once_per_event(self):
        fd = FallDetector(window_seconds=5.0)
        fd.update([100, 50, 160, 400])
        first = fd.update([80, 350, 260, 400])
        second = fd.update([80, 350, 260, 400])
        self.assertTrue(first)
        self.assertFalse(second)


if __name__ == "__main__":
    unittest.main()
