"""
Announcement Manager
=======================
Sits between the priority engine and the Speaker, deciding WHETHER and
WHEN to actually speak something - preventing duplicate, overlapping,
and repetitive announcements, while still letting CRITICAL warnings
interrupt whatever is currently being said.

This is the single source of truth for "have we already said this
recently" - callers should route every spoken announcement through
here (with force=True on the underlying Speaker) rather than also
relying on Speaker's own separate dedup_key/cooldown mechanism, so
there's exactly one place deciding repeat-suppression, not two
fighting each other.
"""
import time


class AnnouncementManager:
    def __init__(self, speaker, cooldown_seconds=6.0, unknown_object_cooldown=15.0):
        self.speaker = speaker
        self.cooldown_seconds = cooldown_seconds
        self.unknown_object_cooldown = unknown_object_cooldown
        self._last_said = {}  # key -> last-spoken timestamp

    def announce(self, text, key=None, tier="LOW", force=False, cooldown=None):
        """Speak `text` unless it's a duplicate of `key` (defaults to
        `text` itself) within the cooldown window. Returns True if it
        was actually spoken, False if suppressed as a repeat.

        `tier="CRITICAL"` interrupts whatever is currently playing
        (Section 27: "Warning. Vehicle approaching." must cut off a
        normal "I can see a chair ahead." mid-sentence) - every other
        tier queues normally and waits its turn.
        """
        if not text:
            return False

        key = key or text
        effective_cooldown = self.cooldown_seconds if cooldown is None else cooldown
        now = time.time()
        if not force:
            last = self._last_said.get(key, 0)
            if now - last < effective_cooldown:
                return False
        self._last_said[key] = now

        if tier == "CRITICAL":
            self.speaker.stop()
        self.speaker.speak(text, force=True)
        return True

    def announce_unknown(self):
        """"Object not clearly recognized." - rate-limited more
        aggressively than normal announcements since low-confidence/
        unstable detections can otherwise trigger it very often."""
        return self.announce(
            "Object not clearly recognized.",
            key="__unknown__",
            cooldown=self.unknown_object_cooldown,
        )

    def reset(self):
        self._last_said = {}
