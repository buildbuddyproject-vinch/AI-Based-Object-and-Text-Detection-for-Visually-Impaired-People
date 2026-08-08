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
