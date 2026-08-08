"""
AI-Based Object and Text Detection for Visually Impaired People
===================================================================
Flask application entry point. Wires together the threaded camera
stream, YOLOv8 object detector, EasyOCR reader, navigation assistant,
bonus feature modules (color / currency / QR) and the offline speaker,
and exposes everything to the web frontend.

Run with:
    python app.py
then open http://127.0.0.1:5000 in a browser.
"""
import os
import time
import threading

from flask import Flask, Response, jsonify, render_template, request

import config
from utils.state import state
from utils.camera_stream import VideoCamera
from utils.frame_utils import resize_frame, encode_jpeg, FPSTracker
from utils.logger import get_logger

from modules.object_detector import ObjectDetector
from modules.ocr_reader import OCRReader
from modules.speaker import Speaker
from modules.navigation import NavigationAssistant
from modules.voice_commands import VoiceCommandListener
from modules.color_detector import detect_dominant_color
from modules.currency_detector import CurrencyDetector
from modules.qr_reader import QRReader

log = get_logger()

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY

# ---------------------------------------------------------------------------
# Lazily-created singletons (heavy models load on first use, not at import
# time, so the dev server starts instantly).
# ---------------------------------------------------------------------------
camera = None
detector = None
ocr_reader = None
currency_detector = None
qr_reader = None

speaker = Speaker(
    rate=config.TTS_RATE,
    volume=config.TTS_VOLUME,
    voice_id=config.TTS_VOICE_ID,
    cooldown=config.SPEAK_COOLDOWN_SECONDS,
)
navigator = NavigationAssistant(near_area_ratio=config.NAV_NEAR_AREA_RATIO)
voice_listener = VoiceCommandListener(
    on_log=lambda msg: state.log_voice(msg),
    is_speaking_check=lambda: speaker.is_speaking,
)

_detection_thread = None
_previous_labels = set()
_last_nav_instruction = None
_pending_nav_instruction = None
_pending_nav_since = 0.0
NAV_DEBOUNCE_SECONDS = 0.6  # ignore instruction flicker shorter than this

_frame_lock = threading.Lock()
_annotated_frame = None


def get_detector():
    global detector
    if detector is None:
        if not os.path.exists(config.YOLO_WEIGHTS_PATH):
            log.warning(
                "YOLO weights not found at %s - attempting to auto-download "
                "yolov8n.pt (see README.md for manual instructions).",
                config.YOLO_WEIGHTS_PATH,
            )
        detector = ObjectDetector(
            model_path=config.YOLO_WEIGHTS_PATH,
            conf_threshold=config.YOLO_CONFIDENCE,
            device=config.YOLO_DEVICE,
        )
    return detector


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        log.info("Loading EasyOCR model (first run may take a while)...")
        ocr_reader = OCRReader(languages=config.OCR_LANGUAGES, gpu=config.OCR_USE_GPU)
    return ocr_reader


def get_currency_detector():
    global currency_detector
    if currency_detector is None:
        currency_detector = CurrencyDetector(reference_dir=config.CURRENCY_DATASET_DIR)
    return currency_detector


def get_qr_reader():
    global qr_reader
    if qr_reader is None:
        qr_reader = QRReader()
    return qr_reader


# ---------------------------------------------------------------------------
# Camera / detection lifecycle
# ---------------------------------------------------------------------------
def start_camera():
    global camera, _detection_thread, _previous_labels
    global _last_nav_instruction, _pending_nav_instruction, _pending_nav_since

    if camera is None or not camera.is_opened:
        camera = VideoCamera(
            source=config.CAMERA_SOURCE, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT
        )
        if not camera.is_opened:
            log.error("Could not open camera source %s", config.CAMERA_SOURCE)
            return False

    state.camera_active = True
    _previous_labels = set()
    _last_nav_instruction = None
    _pending_nav_instruction = None
    _pending_nav_since = 0.0

    if _detection_thread is None or not _detection_thread.is_alive():
        _detection_thread = threading.Thread(target=_detection_loop, daemon=True)
        _detection_thread.start()

    return True


def stop_camera():
    global camera
    state.camera_active = False
    if camera is not None:
        camera.release()
        camera = None


def _detection_loop():
    """Continuously grab frames, run inference, update shared state and
    trigger spoken feedback. Runs on its own daemon thread while the
    camera is active."""
    global _previous_labels, _annotated_frame
    global _last_nav_instruction, _pending_nav_instruction, _pending_nav_since

    fps_tracker = FPSTracker()
    frame_count = 0

    while state.camera_active and camera is not None:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        if state.frame_skip > 1 and frame_count % state.frame_skip != 0:
            time.sleep(config.DETECTION_LOOP_SLEEP)
            continue

        proc_frame = resize_frame(frame, state.resize_scale)
        mode = state.mode

        try:
            detections = get_detector().detect(proc_frame)
        except Exception as exc:
            log.error("Detection error: %s", exc)
            detections = []

        if mode == "color":
            for det in detections:
                det["color"] = detect_dominant_color(proc_frame, det["bbox"])
        elif mode == "qr":
            try:
                qr_results = get_qr_reader().read(proc_frame)
            except Exception as exc:
                log.error("QR read error: %s", exc)
                qr_results = []
            state.last_qr = qr_results
            for qr in qr_results:
                speaker.speak(f"QR code found: {qr['data']}", dedup_key=f"qr:{qr['data']}")

        annotated = get_detector().draw_boxes(proc_frame.copy(), detections)
        h, w = proc_frame.shape[:2]

        if mode == "navigation":
            instruction, zone_map = navigator.analyze(detections, w, h)
            state.update_navigation(instruction, zone_map)

            # Bounding boxes flicker frame-to-frame (detection noise can
            # briefly swap which object is "primary"), which used to
            # trigger a burst of rapid-fire speech - fast enough to
            # overwhelm pyttsx3's Windows driver into a stuck state that
            # silently stops producing audio. Require an instruction to
            # stay stable for NAV_DEBOUNCE_SECONDS before treating it as
            # a real change worth interrupting speech for.
            now = time.time()
            if instruction != _pending_nav_instruction:
                _pending_nav_instruction = instruction
                _pending_nav_since = now

            if now - _pending_nav_since >= NAV_DEBOUNCE_SECONDS:
                if instruction != _last_nav_instruction:
                    speaker.speak(instruction, force=True)
                    _last_nav_instruction = instruction
                else:
                    speaker.speak(instruction, dedup_key="nav")
        else:
            # Speak only newly-appeared object labels so the assistant
            # doesn't repeat the same object every frame while it stays
            # in view.
            current_labels = {d["label"] for d in detections}
            new_labels = current_labels - _previous_labels
            for label in new_labels:
                phrase = label
                if mode == "color":
                    color_for_label = next(
                        (d["color"] for d in detections if d["label"] == label), None
                    )
                    if color_for_label:
                        phrase = f"{color_for_label} {label}"
                speaker.speak(f"{phrase} detected", dedup_key=f"obj:{label}")
            _previous_labels = current_labels

        fps = fps_tracker.tick()
        state.update_detections(detections, fps)

        with _frame_lock:
            _annotated_frame = annotated

        time.sleep(config.DETECTION_LOOP_SLEEP)


def _generate_mjpeg():
    """Generator that yields the latest annotated frame as an MJPEG
    multipart stream for the <img> tag on the frontend."""
    while True:
        with _frame_lock:
            frame = None if _annotated_frame is None else _annotated_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        jpeg = encode_jpeg(frame)
        if jpeg is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.03)


# ---------------------------------------------------------------------------
# Voice assistant command wiring
# ---------------------------------------------------------------------------
def _voice_read_text():
    state.set_redirect("/ocr")
    if camera is None and not start_camera():
        speaker.speak("Could not open the camera.", force=True)
        return
    # Give the camera a brief moment to deliver its first frame if it
    # was just started.
    frame = None
    for _ in range(20):
        frame = camera.get_frame()
        if frame is not None:
            break
        time.sleep(0.1)
    if frame is None:
        speaker.speak("No camera frame available yet. Try again in a moment.", force=True)
        return
    text, _ = get_ocr_reader().read_text(frame, min_confidence=config.OCR_MIN_CONFIDENCE)
    state.last_ocr_text = text
    speaker.speak(text if text else "No text found.", force=True)


def _voice_start_navigation():
    state.set_redirect("/navigation")
    if camera is None:
        start_camera()
    state.mode = "navigation"
    speaker.speak("Navigation mode activated.", force=True)


def _voice_detect_objects():
    state.set_redirect("/live")
    if camera is None:
        start_camera()
    state.mode = "object"
    speaker.speak("Object detection mode activated.", force=True)


def _voice_stop_speaking():
    speaker.stop()


def _voice_exit():
    speaker.speak("Exiting voice assistant.", force=True)
    voice_listener.stop()
    state.voice_active = False


voice_listener.register("read text", _voice_read_text)
voice_listener.register("start navigation", _voice_start_navigation)
voice_listener.register("detect object", _voice_detect_objects)
voice_listener.register("stop speaking", _voice_stop_speaking)
voice_listener.register("exit", _voice_exit)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/live")
def live():
    return render_template("live.html", active="live")


@app.route("/ocr")
def ocr_page():
    return render_template("ocr.html", active="ocr")


@app.route("/navigation")
def navigation_page():
    return render_template("navigation.html", active="navigation")


@app.route("/voice")
def voice_page():
    return render_template("voice.html", active="voice")


@app.route("/about")
def about_page():
    return render_template("about.html", active="about")


# ---------------------------------------------------------------------------
# Camera / streaming API
# ---------------------------------------------------------------------------
@app.route("/camera/start", methods=["POST"])
def api_camera_start():
    ok = start_camera()
    return jsonify({"success": ok, "camera_active": state.camera_active})


@app.route("/camera/stop", methods=["POST"])
def api_camera_stop():
    stop_camera()
    return jsonify({"success": True, "camera_active": state.camera_active})


@app.route("/video_feed")
def video_feed():
    return Response(_generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/mode", methods=["POST"])
def api_set_mode():
    global _previous_labels, _last_nav_instruction, _pending_nav_instruction, _pending_nav_since
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "object")
    if mode not in ("object", "navigation", "color", "qr"):
        return jsonify({"success": False, "error": "invalid mode"}), 400
    state.mode = mode
    _previous_labels = set()
    _last_nav_instruction = None
    _pending_nav_instruction = None
    _pending_nav_since = 0.0
    return jsonify({"success": True, "mode": mode})


@app.route("/detections")
def api_detections():
    return jsonify(state.snapshot())


# ---------------------------------------------------------------------------
# OCR API
# ---------------------------------------------------------------------------
@app.route("/capture_ocr", methods=["POST"])
def api_capture_ocr():
    if camera is None:
        return jsonify({"success": False, "error": "Camera not active"}), 400
    frame = camera.get_frame()
    if frame is None:
        return jsonify({"success": False, "error": "No frame available"}), 400
    text, boxes = get_ocr_reader().read_text(frame, min_confidence=config.OCR_MIN_CONFIDENCE)
    state.last_ocr_text = text
    return jsonify({"success": True, "text": text, "boxes_found": len(boxes)})


# ---------------------------------------------------------------------------
# Bonus feature APIs
# ---------------------------------------------------------------------------
@app.route("/capture_currency", methods=["POST"])
def api_capture_currency():
    if camera is None:
        return jsonify({"success": False, "error": "Camera not active"}), 400
    frame = camera.get_frame()
    det = get_currency_detector()
    if not det.is_ready:
        return jsonify({
            "success": False,
            "error": "No reference currency images found. Add images to dataset/currency/.",
        })
    result = det.detect(frame)
    state.last_currency = result
    if result:
        speaker.speak(f"{result['denomination']} rupee note detected", force=True)
    else:
        speaker.speak("Currency not recognized", force=True)
    return jsonify({"success": True, "result": result})


@app.route("/sos", methods=["POST"])
def api_sos():
    state.sos_active = True
    state.last_sos_time = time.strftime("%Y-%m-%d %H:%M:%S")
    speaker.speak("Emergency S O S activated. Alerting your emergency contact.", force=True)
    log.warning("SOS TRIGGERED at %s", state.last_sos_time)
    return jsonify({
        "success": True,
        "message": "SOS alert triggered (simulated).",
        "contact_name": config.SOS_EMERGENCY_CONTACT,
        "contact_phone": config.SOS_EMERGENCY_PHONE,
        "time": state.last_sos_time,
    })


@app.route("/gps")
def api_gps():
    # Placeholder: no GPS hardware wired up. Swap this with a real serial
    # / USB GPS module reader (e.g. pynmea2 + pyserial) for a physical
    # device build.
    return jsonify({
        "success": True,
        "latitude": config.GPS_MOCK_LAT,
        "longitude": config.GPS_MOCK_LNG,
        "note": "Mock GPS coordinates - connect real GPS hardware for live location.",
    })


@app.route("/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True, silent=True) or {}
    if "battery_saver" in data:
        state.battery_saver = bool(data["battery_saver"])
        state.frame_skip = 3 if state.battery_saver else config.DEFAULT_FRAME_SKIP
        state.resize_scale = 0.6 if state.battery_saver else config.DEFAULT_RESIZE_SCALE
    return jsonify({
        "success": True,
        "settings": {
            "battery_saver": state.battery_saver,
            "frame_skip": state.frame_skip,
            "resize_scale": state.resize_scale,
        },
    })


# ---------------------------------------------------------------------------
# Speech API (used by the OCR "Read Aloud" / "Stop" buttons, etc.)
# ---------------------------------------------------------------------------
@app.route("/speak", methods=["POST"])
def api_speak():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    speaker.speak(text, force=True)
    return jsonify({"success": True})


@app.route("/stop_speaking", methods=["POST"])
def api_stop_speaking():
    speaker.stop()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Voice assistant API
# ---------------------------------------------------------------------------
@app.route("/voice/start", methods=["POST"])
def api_voice_start():
    voice_listener.start()
    state.voice_active = True
    return jsonify({"success": True})


@app.route("/voice/stop", methods=["POST"])
def api_voice_stop():
    voice_listener.stop()
    state.voice_active = False
    return jsonify({"success": True})


@app.route("/voice/status")
def api_voice_status():
    return jsonify({
        "active": voice_listener.is_running,
        "log": state.voice_log[-20:],
        "redirect": state.pop_redirect(),
    })


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting AI Blind Assistant server on http://127.0.0.1:5000")
    # use_reloader=False: avoids double-loading the YOLO/EasyOCR models
    # and spawning duplicate camera threads under Flask's auto-reloader.
    app.run(host="0.0.0.0", port=5000, debug=config.DEBUG, threaded=True, use_reloader=False)
