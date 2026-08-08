"""
Structured Event Log
=======================
Lightweight, in-memory + on-disk log of notable app events (mode
changes, SOS triggers, errors, gesture/voice commands, currency/QR
hits) that powers the /dashboard page. This is deliberately simple -
not a replacement for a real observability stack, just enough to make
the running system's behaviour inspectable for a demo or project
report.
"""
import collections
import json
import os
import threading
import time

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "events.log")


class EventLog:
    def __init__(self, max_memory=200):
        self._lock = threading.Lock()
        self._events = collections.deque(maxlen=max_memory)
        self._counters = collections.Counter()
        self._start_time = time.time()
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
        except Exception:
            pass

    def record(self, event_type, **details):
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "type": event_type, **details}
        with self._lock:
            self._events.append(entry)
            self._counters[event_type] += 1
        # Best-effort persistence - a logging failure (e.g. read-only
        # filesystem on some hosts) should never take the app down.
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def recent(self, limit=50):
        with self._lock:
            return list(self._events)[-limit:]

    def counters(self):
        with self._lock:
            return dict(self._counters)

    @property
    def uptime_seconds(self):
        return time.time() - self._start_time


# Single shared instance, imported by app.py.
event_log = EventLog()
