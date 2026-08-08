"""Unit tests for the contextual detection memory - pure logic, no
camera or model weights required."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.memory import DetectionMemory


class TestDetectionMemory(unittest.TestCase):
    def setUp(self):
        self.memory = DetectionMemory(retention_seconds=600)

    def test_unseen_object_returns_none(self):
        self.assertIsNone(self.memory.find_last("my bag"))
        self.assertIn("haven't seen", self.memory.describe(None))

    def test_synonym_matches_recorded_label(self):
        self.memory.record("backpack", zone="left")
        entry = self.memory.find_last("where is my bag")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["label"], "backpack")
        self.assertEqual(entry["zone"], "left")

    def test_direct_label_name_matches(self):
        self.memory.record("chair", zone="right")
        entry = self.memory.find_last("where is the chair")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["label"], "chair")

    def test_most_recent_match_wins(self):
        self.memory.record("backpack", zone="left")
        time.sleep(0.01)
        self.memory.record("handbag", zone="right")
        entry = self.memory.find_last("my bag")
        self.assertEqual(entry["label"], "handbag")

    def test_expired_entry_is_not_returned(self):
        short_memory = DetectionMemory(retention_seconds=0.05)
        short_memory.record("backpack", zone="left")
        time.sleep(0.1)
        self.assertIsNone(short_memory.find_last("my bag"))

    def test_describe_mentions_zone_and_recency(self):
        self.memory.record("chair", zone="center")
        entry = self.memory.find_last("chair")
        description = self.memory.describe(entry)
        self.assertIn("chair", description.lower())
        self.assertIn("ahead", description.lower())


if __name__ == "__main__":
    unittest.main()
