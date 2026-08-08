"""Unit tests for OCR reading-order grouping and important-keyword
detection - pure logic, no EasyOCR model load required."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ocr_reader import _group_into_lines, find_important_keyword


class TestImportantKeyword(unittest.TestCase):
    def test_finds_known_keyword_case_insensitively(self):
        self.assertEqual(find_important_keyword("Beware: WET FLOOR ahead"), "wet floor")

    def test_returns_none_for_plain_text(self):
        self.assertIsNone(find_important_keyword("Fresh bread, baked daily"))

    def test_empty_text_returns_none(self):
        self.assertIsNone(find_important_keyword(""))


class TestGroupIntoLines(unittest.TestCase):
    def test_same_row_items_grouped_into_one_line(self):
        items = [
            {"text": "Aisle", "cx": 10, "cy": 100, "h": 20},
            {"text": "5", "cx": 60, "cy": 102, "h": 20},
        ]
        lines = _group_into_lines(items)
        self.assertEqual(len(lines), 1)
        self.assertEqual([it["text"] for it in lines[0]], ["Aisle", "5"])

    def test_different_rows_produce_separate_lines_top_to_bottom(self):
        items = [
            {"text": "Second", "cx": 10, "cy": 200, "h": 20},
            {"text": "First", "cx": 10, "cy": 20, "h": 20},
        ]
        lines = _group_into_lines(items)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][0]["text"], "First")
        self.assertEqual(lines[1][0]["text"], "Second")

    def test_left_to_right_order_within_a_line(self):
        items = [
            {"text": "world", "cx": 80, "cy": 50, "h": 20},
            {"text": "hello", "cx": 10, "cy": 50, "h": 20},
        ]
        lines = _group_into_lines(items)
        self.assertEqual([it["text"] for it in lines[0]], ["hello", "world"])


if __name__ == "__main__":
    unittest.main()
