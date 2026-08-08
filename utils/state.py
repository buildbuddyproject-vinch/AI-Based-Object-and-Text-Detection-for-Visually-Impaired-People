"""
Shared, thread-safe application state.
=========================================
Every background thread (camera reader, detection worker, voice
listener) reads/writes through this single object so the Flask routes
always see a consistent snapshot without needing direct access to the
worker threads themselves.
"""
import threading
import time


class AppState:
    def __init__(self):
        self.lock = threading.RLock()

        # camera / detection
        self.camera_active = False
        self.mode = "object"  # object | navigation | color | qr
        self.detections = []
        self.navigation_instruction = "Path looks clear."
        self.zone_map = {"left": [], "center": [], "right": []}
        self.fps = 0.0

        # OCR
        self.last_ocr_text = ""

        # bonus features
        self.last_currency = None
        self.last_qr = []

        # voice assistant
        self.voice_active = False
        self.voice_log = []
        # Set by a voice command that should take the user to a specific
        # page (e.g. "start navigation" -> "/navigation"). Consumed
        # once by /voice/status so the frontend can act on it, then
        # cleared - see AppState.pop_redirect().
        self.pending_redirect = None

        # performance / battery optimization
        self.frame_skip = 1
        self.resize_scale = 1.0
        self.battery_saver = False

        # SOS placeholder
        self.sos_active = False
        self.last_sos_time = None

        # Real GPS coordinates reported by the browser's Geolocation API
        # (see static/js/main.js) - None until a page successfully gets a
        # location, at which point /gps prefers this over the mock
        # coordinates in config.py.
        self.user_location = None

        # Scene summary (on-demand natural-language description)
        self.last_scene_summary = ""

        # Monocular depth estimation availability/state
        self.depth_enabled = False

        # Heuristic fall detection
        self.fall_detected = False
        self.last_fall_time = None

        # Hand gesture recognition
        self.gesture_active = False
        self.last_gesture = None

    def update_detections(self, detections, fps):
        with self.lock:
            self.detections = detections
            self.fps = fps

    def update_navigation(self, instruction, zone_map):
        with self.lock:
            self.navigation_instruction = instruction
            self.zone_map = zone_map

    def log_voice(self, message, limit=50):
        with self.lock:
            self.voice_log.append({"time": time.strftime("%H:%M:%S"), "message": message})
            self.voice_log = self.voice_log[-limit:]

    def set_redirect(self, path):
        with self.lock:
            self.pending_redirect = path

    def pop_redirect(self):
        """Return the pending redirect path (or None) and clear it -
        each voice-triggered redirect should only fire once."""
        with self.lock:
            path, self.pending_redirect = self.pending_redirect, None
            return path

    def set_location(self, latitude, longitude, accuracy=None):
        with self.lock:
            self.user_location = {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": accuracy,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def snapshot(self):
        """Return a plain-dict copy of the current state, safe to
        json-serialize directly for the polling API endpoints."""
        with self.lock:
            return {
                "camera_active": self.camera_active,
                "mode": self.mode,
                "detections": list(self.detections),
                "navigation_instruction": self.navigation_instruction,
                "zone_map": dict(self.zone_map),
                "fps": self.fps,
                "last_ocr_text": self.last_ocr_text,
                "last_currency": self.last_currency,
                "last_qr": list(self.last_qr),
                "voice_active": self.voice_active,
                "voice_log": list(self.voice_log),
                "frame_skip": self.frame_skip,
                "resize_scale": self.resize_scale,
                "battery_saver": self.battery_saver,
                "sos_active": self.sos_active,
                "user_location": dict(self.user_location) if self.user_location else None,
                "last_scene_summary": self.last_scene_summary,
                "depth_enabled": self.depth_enabled,
                "fall_detected": self.fall_detected,
                "gesture_active": self.gesture_active,
                "last_gesture": self.last_gesture,
            }


# Single shared instance imported by app.py and any module that needs to
# report status.
state = AppState()
