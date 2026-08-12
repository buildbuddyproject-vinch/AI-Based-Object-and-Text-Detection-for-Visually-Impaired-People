"""Unit tests for the announcement manager - uses a fake Speaker stub so
no actual audio/subprocess is involved."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.announcement_manager import AnnouncementManager


class FakeSpeaker:
    def __init__(self):
        self.spoken = []
        self.stopped = 0

    def speak(self, text, force=False, dedup_key=None):
        self.spoken.append(text)

    def stop(self):
        self.stopped += 1


class TestAnnouncementManager(unittest.TestCase):
    def setUp(self):
        self.speaker = FakeSpeaker()
        self.manager = AnnouncementManager(self.speaker, cooldown_seconds=6.0)

    def test_first_announcement_is_spoken(self):
        said = self.manager.announce("Chair ahead.", key="chair")
        self.assertTrue(said)
        self.assertEqual(self.speaker.spoken, ["Chair ahead."])

    def test_repeat_within_cooldown_is_suppressed(self):
        self.manager.announce("Chair ahead.", key="chair")
        said_again = self.manager.announce("Chair ahead.", key="chair")
        self.assertFalse(said_again)
        self.assertEqual(len(self.speaker.spoken), 1)

    def test_different_keys_both_announce(self):
        self.manager.announce("Chair ahead.", key="chair")
        said = self.manager.announce("Table on your left.", key="table")
        self.assertTrue(said)
        self.assertEqual(len(self.speaker.spoken), 2)

    def test_critical_tier_interrupts_current_speech(self):
        self.manager.announce("I can see a chair ahead.", key="chair")
        self.manager.announce("Warning. Vehicle approaching.", key="vehicle", tier="CRITICAL")
        self.assertEqual(self.speaker.stopped, 1)
        self.assertEqual(self.speaker.spoken[-1], "Warning. Vehicle approaching.")

    def test_non_critical_does_not_interrupt(self):
        self.manager.announce("Chair ahead.", key="chair")
        self.manager.announce("Table ahead.", key="table", tier="LOW")
        self.assertEqual(self.speaker.stopped, 0)

    def test_empty_text_does_nothing(self):
        said = self.manager.announce("")
        self.assertFalse(said)
        self.assertEqual(self.speaker.spoken, [])

    def test_announce_unknown_uses_its_own_longer_cooldown(self):
        self.manager.unknown_object_cooldown = 100
        self.manager.announce_unknown()
        said_again = self.manager.announce_unknown()
        self.assertFalse(said_again)
        self.assertEqual(self.speaker.spoken, ["Object not clearly recognized."])

    def test_force_bypasses_cooldown(self):
        self.manager.announce("Chair ahead.", key="chair")
        said = self.manager.announce("Chair ahead.", key="chair", force=True)
        self.assertTrue(said)
        self.assertEqual(len(self.speaker.spoken), 2)


if __name__ == "__main__":
    unittest.main()
