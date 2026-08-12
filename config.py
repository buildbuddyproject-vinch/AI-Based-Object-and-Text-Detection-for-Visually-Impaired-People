"""
Central configuration for the AI Blind Assistant.

Every tunable value used across the app (camera, YOLO, OCR, TTS,
navigation heuristics, performance/battery settings, bonus features) lives
here so the rest of the codebase never hard-codes a magic number.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Camera ------------------------------------------------------------
CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", 0))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- YOLOv8 object detection --------------------------------------------
# Run `python export_onnx.py` once, then point this at the .onnx file
# instead for meaningfully faster CPU inference (Ultralytics loads
# either format transparently - no other code changes needed). Stick
# with .pt if you ever move to a GPU host.
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "yolov8n.pt")
YOLO_CONFIDENCE = 0.45
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "cpu")  # "cpu" or "0" for first GPU

# --- EasyOCR text recognition --------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_USE_GPU = False
OCR_MIN_CONFIDENCE = 0.4

# --- pyttsx3 offline text-to-speech ---------------------------------------
TTS_RATE = 170
TTS_VOLUME = 1.0
TTS_VOICE_ID = None  # set to a specific pyttsx3 voice id string to override
SPEAK_COOLDOWN_SECONDS = 6.0

# --- Navigation heuristics -------------------------------------------------
NAV_NEAR_AREA_RATIO = 0.18  # bbox_area / frame_area above this => "close"

# --- Monocular depth estimation (MiDaS-small) -------------------------------
# Adds a real per-pixel relative-depth signal to navigation guidance
# instead of only the bbox-size heuristic above. Downloads model weights
# via torch.hub on first use (needs internet then; cached after).
# Disable if that download isn't possible, or if the extra ~0.1-0.2s/
# frame inference cost is too much for your hardware.
ENABLE_DEPTH_ESTIMATION = os.environ.get("ENABLE_DEPTH_ESTIMATION", "1") == "1"
DEPTH_FRAME_INTERVAL = 3  # only re-run depth every Nth navigation-mode frame

# --- Hand gesture recognition (MediaPipe) -----------------------------------
# Downloads a small (~a few MB) landmarker model on first use.
ENABLE_GESTURES = os.environ.get("ENABLE_GESTURES", "1") == "1"
GESTURE_FRAME_INTERVAL = 2       # process every Nth frame while enabled
GESTURE_COOLDOWN_SECONDS = 2.5   # don't re-trigger the same gesture faster than this

# --- Low-light detection -----------------------------------------------------
LOW_LIGHT_THRESHOLD = 60          # mean grayscale brightness (0-255) below this = "dark"
LOW_LIGHT_CHECK_INTERVAL = 2.0    # seconds between brightness checks
# Speaks once as soon as the scene goes dark, then stays quiet about it
# (Section 10: "if nothing changes, do not repeat") until either the
# light comes back (resetting the warning) or this much time has passed
# in one continuous dark session, whichever is sooner - a single gentle
# reminder rather than nagging every 30s for as long as it stays dark.
LOW_LIGHT_REMINDER_INTERVAL = 300.0

# --- Fall detection (heuristic, experimental - see modules/fall_detector.py)
ENABLE_FALL_DETECTION = os.environ.get("ENABLE_FALL_DETECTION", "1") == "1"

# --- Contextual memory ("where is my X") -------------------------------------
MEMORY_RETENTION_SECONDS = 600  # how long a "last seen" answer stays valid

# --- Performance / battery optimization ------------------------------------
DEFAULT_FRAME_SKIP = 1        # process every Nth grabbed frame
DEFAULT_RESIZE_SCALE = 1.0    # downscale factor applied before inference
DETECTION_LOOP_SLEEP = 0.01   # small yield between detection loop cycles

# --- Bonus features ----------------------------------------------------------
CURRENCY_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "currency")
SOS_EMERGENCY_CONTACT = os.environ.get("SOS_CONTACT_NAME", "Emergency Contact")
SOS_EMERGENCY_PHONE = os.environ.get("SOS_CONTACT_PHONE", "+91-XXXXXXXXXX")
GPS_MOCK_LAT = 12.9716   # placeholder coordinates (Bengaluru) until real
GPS_MOCK_LNG = 77.5946   # GPS hardware (e.g. NEO-6M) is wired up

# --- Flask ------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
