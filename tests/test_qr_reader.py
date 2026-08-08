"""Sanity tests for the QR reader wrapper - no camera required."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.qr_reader import QRReader


class TestQRReader(unittest.TestCase):
    def test_blank_frame_returns_no_codes(self):
        reader = QRReader()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        self.assertEqual(reader.read(frame), [])

    def test_none_frame_returns_no_codes(self):
        reader = QRReader()
        self.assertEqual(reader.read(None), [])


if __name__ == "__main__":
    unittest.main()
