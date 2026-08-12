"""Unit tests for the priority engine - pure logic, no camera or model
weights required."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.priority_engine import classify_priority, select_most_relevant


class TestClassifyPriority(unittest.TestCase):
    def test_pothole_is_critical(self):
        self.assertEqual(classify_priority("pothole"), "CRITICAL")

    def test_person_is_high(self):
        self.assertEqual(classify_priority("person"), "HIGH")

    def test_chair_is_medium(self):
        self.assertEqual(classify_priority("chair"), "MEDIUM")

    def test_unlisted_label_is_low(self):
        self.assertEqual(classify_priority("bottle"), "LOW")

    def test_very_close_escalates_to_critical(self):
        self.assertEqual(classify_priority("bottle", is_very_close=True), "CRITICAL")

    def test_blocks_path_escalates_to_critical(self):
        self.assertEqual(classify_priority("chair", blocks_path=True), "CRITICAL")


class TestSelectMostRelevant(unittest.TestCase):
    def test_empty_list_returns_none(self):
        self.assertIsNone(select_most_relevant([]))

    def test_critical_beats_high_regardless_of_size(self):
        candidates = [
            {"label": "person", "area_ratio": 0.9},
            {"label": "pothole", "area_ratio": 0.01},
        ]
        result = select_most_relevant(candidates)
        self.assertEqual(result["label"], "pothole")

    def test_within_tier_larger_wins(self):
        candidates = [
            {"label": "person", "area_ratio": 0.1},
            {"label": "car", "area_ratio": 0.5},
        ]
        result = select_most_relevant(candidates)
        self.assertEqual(result["label"], "car")

    def test_explicit_tier_overrides_label_lookup(self):
        candidates = [
            {"label": "bottle", "tier": "CRITICAL", "area_ratio": 0.01},
            {"label": "person", "area_ratio": 0.9},
        ]
        result = select_most_relevant(candidates)
        self.assertEqual(result["label"], "bottle")


if __name__ == "__main__":
    unittest.main()
