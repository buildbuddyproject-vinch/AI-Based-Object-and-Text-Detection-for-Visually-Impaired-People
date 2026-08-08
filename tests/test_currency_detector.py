"""Sanity tests for the heuristic currency detector - no camera or
reference images required."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.currency_detector import CurrencyDetector


class TestCurrencyDetector(unittest.TestCase):
    def test_not_ready_without_reference_images(self):
        det = CurrencyDetector(reference_dir="path/does/not/exist")
        self.assertFalse(det.is_ready)

    def test_detect_returns_none_when_not_ready(self):
        det = CurrencyDetector(reference_dir="path/does/not/exist")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(det.detect(frame))


if __name__ == "__main__":
    unittest.main()
