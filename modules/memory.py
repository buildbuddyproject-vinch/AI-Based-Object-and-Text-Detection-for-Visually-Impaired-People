"""
Detection Memory Module
==========================
Keeps a lightweight "last seen" record per object label so the voice
assistant can answer questions like "where is my bag?" - not real
object tracking/re-identification, just a timestamped record of the
most recent time+zone each label was detected.
"""
import threading
import time

# Everyday words mapped to the COCO class names YOLOv8 actually
# detects, so a natural "my bag" question matches a detected
# "backpack"/"handbag"/"suitcase".
SYNONYMS = {
    "bag": ["backpack", "handbag", "suitcase"],
    "backpack": ["backpack"],
    "handbag": ["handbag"],
    "purse": ["handbag"],
    "suitcase": ["suitcase"],
    "wallet": ["handbag"],
    "phone": ["cell phone"],
    "mobile": ["cell phone"],
    "cellphone": ["cell phone"],
    "laptop": ["laptop"],
    "bottle": ["bottle"],
    "cup": ["cup"],
    "book": ["book"],
    "remote": ["remote"],
    "umbrella": ["umbrella"],
    "chair": ["chair"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
}


class DetectionMemory:
    def __init__(self, retention_seconds=600):
        self.retention_seconds = retention_seconds
        self._lock = threading.Lock()
        self._last_seen = {}  # label -> {"time": float, "zone": str|None}

    def record(self, label, zone=None):
        with self._lock:
            self._last_seen[label] = {"time": time.time(), "zone": zone}

    def find_last(self, query):
        """Return {"label", "time", "zone"} for the most recently seen
        label matching the natural-language `query`, or None if nothing
        recent matches (either never seen, or seen longer ago than
        `retention_seconds`)."""
        query = (query or "").lower()
        candidate_labels = set()
        for word, labels in SYNONYMS.items():
            if word in query:
                candidate_labels.update(labels)

        with self._lock:
            best_label, best_info = None, None
            for label, info in self._last_seen.items():
                if label in candidate_labels or label in query:
                    if best_info is None or info["time"] > best_info["time"]:
                        best_label, best_info = label, info

        if best_info is None:
            return None
        if time.time() - best_info["time"] > self.retention_seconds:
            return None
        return {"label": best_label, **best_info}

    @staticmethod
    def describe(entry):
        """Turn a find_last() result into a spoken sentence."""
        if entry is None:
            return "I haven't seen that recently."

        seconds_ago = int(time.time() - entry["time"])
        if seconds_ago < 5:
            when = "just now"
        elif seconds_ago < 60:
            when = f"about {seconds_ago} seconds ago"
        else:
            when = f"about {seconds_ago // 60} minute" + ("s" if seconds_ago // 60 != 1 else "") + " ago"

        zone = entry.get("zone")
        if zone == "left":
            where = " on your left"
        elif zone == "right":
            where = " on your right"
        elif zone == "center":
            where = " ahead of you"
        else:
            where = ""

        return f"I last saw a {entry['label']}{where}, {when}."


# Single shared instance, imported by app.py.
memory = DetectionMemory()
