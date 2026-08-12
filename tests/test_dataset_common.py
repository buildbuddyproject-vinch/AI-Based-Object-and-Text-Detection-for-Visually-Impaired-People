import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from dataset_common import is_watermark_class, voc_box_to_yolo  # noqa: E402


class TestIsWatermarkClass(unittest.TestCase):
    def test_real_class_names_pass_through(self):
        for name in ("car", "pole", "truck", "flyover", "hoarding", "footpath",
                     "traffic symbols", "auto rickshaw"):
            self.assertFalse(is_watermark_class(name), name)

    def test_known_roboflow_watermark_substrings_are_filtered(self):
        self.assertTrue(is_watermark_class("Roboflow is an end-to-end computer vision platform that helps you"))
        self.assertTrue(is_watermark_class("- collaborate with your team on computer vision projects"))

    def test_empty_or_dash_is_filtered(self):
        self.assertTrue(is_watermark_class(""))
        self.assertTrue(is_watermark_class("   "))
        self.assertTrue(is_watermark_class("-"))

    def test_roboflow_project_version_label_is_filtered(self):
        # The real bug this regression test locks in: a Roboflow export
        # project/version label leaked in as a stray class 330 times in
        # dataset/outdoor/XML Files/, and the original substring-only
        # filter (checking only for "roboflow" etc.) missed it entirely
        # because this string doesn't contain any of those substrings.
        self.assertTrue(is_watermark_class("day2 - v3 2025-02-05 12-27pm"))

    def test_case_insensitive(self):
        self.assertTrue(is_watermark_class("DAY2 - V3 2025-02-05 12-27PM"))
        self.assertTrue(is_watermark_class("ROBOFLOW"))

    def test_real_class_with_no_year_or_version_marker_is_not_filtered(self):
        # Guard against the regex being so broad it eats real classes -
        # none of the actual dataset class names contain a bare 4-digit
        # year or "v<digit>" token.
        for name in ("traffic signal", "building", "bike", "pedestrian", "bus", "caravan"):
            self.assertFalse(is_watermark_class(name), name)


class TestVocBoxToYolo(unittest.TestCase):
    def test_full_image_box(self):
        cx, cy, w, h = voc_box_to_yolo(0, 0, 100, 200, 100, 200)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(h, 1.0)

    def test_centered_small_box(self):
        cx, cy, w, h = voc_box_to_yolo(40, 80, 60, 120, 100, 200)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(w, 0.2)
        self.assertAlmostEqual(h, 0.2)

    def test_clamps_out_of_bounds_coordinates(self):
        # A malformed/edge-case XML box that goes negative or past the
        # image edge must not produce a negative width/height or a
        # normalized coordinate outside [0, 1].
        cx, cy, w, h = voc_box_to_yolo(-10, -10, 50, 250, 100, 200)
        self.assertGreaterEqual(cx - w / 2, -1e-9)
        self.assertGreaterEqual(cy - h / 2, -1e-9)
        self.assertLessEqual(cx + w / 2, 1.0 + 1e-9)
        self.assertLessEqual(cy + h / 2, 1.0 + 1e-9)

    def test_swapped_min_max_is_corrected(self):
        # xmax < xmin shouldn't happen in well-formed VOC XML, but if it
        # does, the box must still come out with a positive width.
        cx, cy, w, h = voc_box_to_yolo(60, 120, 40, 80, 100, 200)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)


if __name__ == "__main__":
    unittest.main()
