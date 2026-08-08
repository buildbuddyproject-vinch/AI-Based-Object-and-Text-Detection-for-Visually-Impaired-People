"""
Hand Gesture Recognition Module (bonus feature)
==================================================
Uses MediaPipe's Hand Landmarker (the current Tasks API - MediaPipe
1.x removed the older `mp.solutions.hands` API entirely) to recognize a
small set of static hand poses from a single frame, so the assistant
can be controlled without speaking - useful in noisy environments, or
as a second input channel alongside voice ("multimodal control": either
input triggers the same actions).

Recognized gestures (heuristic finger-counting, not a trained
classifier - reliable in good lighting with a hand held up clearly
toward the camera, less so otherwise):

    open_palm    - all 5 fingers extended  -> stop speaking
    fist         - no fingers extended     -> start/stop detection
    one_finger   - only the index finger   -> describe the scene
    two_fingers  - index + middle ("peace")-> read text
    thumbs_up    - only the thumb          -> repeat last announcement

This deliberately recognizes only *static* poses (one per frame), not
continuous/sequential gestures or sign language - real sign-language
recognition is a much harder, research-level problem, and pretending
otherwise would over-promise what a finger-counting heuristic can do.

The landmarker model (~a few MB) is downloaded on first use, the same
lazy-download pattern as the YOLO weights and EasyOCR models.

`mediapipe` itself is imported lazily, inside GestureRecognizer.__init__,
rather than at module level: this module is imported unconditionally by
app.py just to reach GESTURE_ACTIONS (a plain dict used to map gestures
to actions), and mediapipe is deliberately excluded from
requirements-render.txt (see that file for why) - a module-level import
would crash the *entire app* at startup wherever mediapipe isn't
installed, not just disable this one feature.
"""
import os
import urllib.request

import cv2

from utils.logger import get_logger

log = get_logger()

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "weights", "hand_landmarker.task")

# Landmark indices per MediaPipe Hands' 21-point model:
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
_FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
_FINGER_PIPS = [6, 10, 14, 18]
_THUMB_TIP, _THUMB_IP = 4, 3

GESTURE_ACTIONS = {
    "open_palm": "stop speaking",
    "fist": "toggle detection",
    "one_finger": "describe scene",
    "two_fingers": "read text",
    "thumbs_up": "repeat last announcement",
}


def _ensure_model_downloaded():
    if os.path.exists(_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    log.info("Downloading hand gesture model (first use only) to %s", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def _classify(landmarks, handedness_label):
    """Return a gesture name for one hand's 21 landmarks, or None if it
    doesn't match any recognized pose."""
    # Non-thumb fingers: "extended" if the tip sits above (smaller y
    # than) its own PIP joint - works for a hand held roughly upright
    # facing the camera, which is the expected use case here.
    extended = [landmarks[tip].y < landmarks[pip].y
                for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS)]

    # Thumb: extended if the tip is farther from the palm on the x-axis
    # than its IP joint - which side counts as "farther" flips with
    # handedness since MediaPipe mirrors a selfie-view frame.
    thumb_tip_x = landmarks[_THUMB_TIP].x
    thumb_ip_x = landmarks[_THUMB_IP].x
    thumb_extended = (thumb_tip_x < thumb_ip_x) if handedness_label == "Right" else (thumb_tip_x > thumb_ip_x)

    if not thumb_extended and not any(extended):
        return "fist"
    if thumb_extended and all(extended):
        return "open_palm"
    if thumb_extended and not any(extended):
        return "thumbs_up"
    if extended[0] and not thumb_extended and not any(extended[1:]):
        return "one_finger"
    if extended[0] and extended[1] and not thumb_extended and not any(extended[2:]):
        return "two_fingers"
    return None


class GestureRecognizer:
    def __init__(self, max_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        # Imported here, not at module level - see the module docstring
        # for why (app.py imports GESTURE_ACTIONS unconditionally, and
        # mediapipe is deliberately not installed on every deployment).
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self._mp = mp
        _ensure_model_downloaded()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame_bgr, draw=True):
        """Run hand detection once on `frame_bgr`. Returns (gesture,
        frame): `gesture` is one of GESTURE_ACTIONS' keys or None; if
        `draw` is True, a simple landmark overlay is drawn on
        `frame_bgr` in-place (for the live preview)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        gesture = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handedness_label = "Right"
            if result.handedness:
                handedness_label = result.handedness[0][0].category_name
            gesture = _classify(landmarks, handedness_label)

            if draw:
                self._draw_landmarks(frame_bgr, result.hand_landmarks)

        return gesture, frame_bgr

    @staticmethod
    def _draw_landmarks(frame_bgr, hands_landmarks, color=(0, 200, 255)):
        h, w = frame_bgr.shape[:2]
        for landmarks in hands_landmarks:
            points = [(int(pt.x * w), int(pt.y * h)) for pt in landmarks]
            for x, y in points:
                cv2.circle(frame_bgr, (x, y), 4, color, -1)

    def close(self):
        self._landmarker.close()
