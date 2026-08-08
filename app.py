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
from utils.frame_utils import resize_frame, encode_jpeg, is_low_light, FPSTracker
from utils.logger import get_logger
from utils.event_log import event_log

from modules.object_detector import ObjectDetector
from modules.ocr_reader import OCRReader, find_important_keyword
from modules.speaker import Speaker
from modules.navigation import NavigationAssistant
from modules.voice_commands import VoiceCommandListener
from modules.color_detector import detect_dominant_color
from modules.currency_detector import CurrencyDetector
from modules.qr_reader import QRReader
from modules.memory import memory
from modules.scene_summary import build_scene_summary
from modules.fall_detector import FallDetector
from modules.gesture_recognizer import GESTURE_ACTIONS

log = get_logger()

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
# Set explicitly (not just passed to app.run() below) so it also takes
# effect under a production WSGI server like gunicorn, which imports
# `app` directly and never calls app.run() at all.
app.config["DEBUG"] = config.DEBUG

# ---------------------------------------------------------------------------
# Lazily-created singletons (heavy models load on first use, not at import
# time, so the dev server starts instantly).
# ---------------------------------------------------------------------------
camera = None
detector = None
ocr_reader = None
currency_detector = None
qr_reader = None
depth_estimator = None
depth_estimator_failed = False  # stop retrying every frame once it's clear it can't load
gesture_recognizer = None
gesture_recognizer_failed = False

speaker = Speaker(
    rate=config.TTS_RATE,
    volume=config.TTS_VOLUME,
    voice_id=config.TTS_VOICE_ID,
    cooldown=config.SPEAK_COOLDOWN_SECONDS,
)
navigator = NavigationAssistant(near_area_ratio=config.NAV_NEAR_AREA_RATIO)
fall_detector = FallDetector()
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

_cached_depth_map = None
_last_gesture_time = 0.0
_last_low_light_check = 0.0
_last_low_light_warning = 0.0

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


def get_depth_estimator():
    """Return the shared DepthEstimator, or None if disabled/unavailable
    (e.g. no internet on first use to download MiDaS weights). Failure
    is remembered so the detection loop doesn't retry the expensive
    download attempt every single frame."""
    global depth_estimator, depth_estimator_failed
    if depth_estimator_failed or not config.ENABLE_DEPTH_ESTIMATION:
        state.depth_enabled = False
        return None
    if depth_estimator is None:
        try:
            from modules.depth_estimator import DepthEstimator
            log.info("Loading MiDaS depth estimation model (first run may take a while)...")
            depth_estimator = DepthEstimator(device=config.YOLO_DEVICE if config.YOLO_DEVICE != "cpu" else "cpu")
        except Exception as exc:
            log.warning("Depth estimation unavailable (%s) - navigation will use the "
                        "bounding-box-size heuristic instead.", exc)
            depth_estimator_failed = True
            state.depth_enabled = False
            return None
    state.depth_enabled = True
    return depth_estimator


def get_gesture_recognizer():
    """Return the shared GestureRecognizer, or None if disabled/
    unavailable (e.g. no internet on first use to download the
    landmarker model)."""
    global gesture_recognizer, gesture_recognizer_failed
    if gesture_recognizer_failed or not config.ENABLE_GESTURES:
        return None
    if gesture_recognizer is None:
        try:
            from modules.gesture_recognizer import GestureRecognizer
            log.info("Loading hand gesture recognition model (first run may take a while)...")
            gesture_recognizer = GestureRecognizer()
        except Exception as exc:
            log.warning("Gesture recognition unavailable (%s).", exc)
            gesture_recognizer_failed = True
            return None
    return gesture_recognizer


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
    global _cached_depth_map, _last_gesture_time
    global _last_low_light_check, _last_low_light_warning

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
        now = time.time()

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
                event_log.record("qr_detected", data=qr["data"])

        annotated = get_detector().draw_boxes(proc_frame.copy(), detections)
        h, w = proc_frame.shape[:2]

        # --- Monocular depth estimation (navigation mode only, ------------
        # throttled to every DEPTH_FRAME_INTERVAL frames - MiDaS is fast
        # (~0.1-0.2s on CPU) but still not free enough to run every frame).
        depth_map = None
        if mode == "navigation" and config.ENABLE_DEPTH_ESTIMATION:
            if _cached_depth_map is None or frame_count % config.DEPTH_FRAME_INTERVAL == 0:
                estimator = get_depth_estimator()
                if estimator is not None:
                    try:
                        _cached_depth_map = estimator.estimate(proc_frame)
                    except Exception as exc:
                        log.error("Depth estimation error: %s", exc)
            depth_map = _cached_depth_map

        # --- Navigation guidance + zone map --------------------------------
        # Computed every frame regardless of mode (cheap, no model) so
        # scene summaries and "where is my X" work no matter what mode
        # is active; only *spoken* when mode == "navigation" below.
        instruction, zone_map = navigator.analyze(detections, w, h, depth_map=depth_map)
        state.update_navigation(instruction, zone_map)

        # --- Contextual memory: "where did I last see my bag?" -----------
        for det in detections:
            cx = (det["bbox"][0] + det["bbox"][2]) / 2
            memory.record(det["label"], zone=navigator._zone(cx, w))

        # --- Fall detection (heuristic, experimental) ----------------------
        if config.ENABLE_FALL_DETECTION:
            person_boxes = [d["bbox"] for d in detections if d["label"] == "person"]
            largest_person = (
                max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                if person_boxes else None
            )
            if fall_detector.update(largest_person):
                state.fall_detected = True
                state.last_fall_time = time.strftime("%Y-%m-%d %H:%M:%S")
                speaker.speak(
                    "Possible fall detected. Are you okay? This is an automated, "
                    "experimental check, not a medical alert - please seek help if needed.",
                    force=True,
                )
                event_log.record("possible_fall_detected")

        # --- Low-light warning (cheap heuristic, time-gated) ----------------
        if now - _last_low_light_check >= config.LOW_LIGHT_CHECK_INTERVAL:
            _last_low_light_check = now
            if (is_low_light(proc_frame, threshold=config.LOW_LIGHT_THRESHOLD)
                    and now - _last_low_light_warning >= config.LOW_LIGHT_WARNING_COOLDOWN):
                _last_low_light_warning = now
                speaker.speak("Low light detected. Detection accuracy may be reduced.", force=True)

        # --- Hand gesture recognition (optional, throttled) ------------------
        if state.gesture_active and frame_count % config.GESTURE_FRAME_INTERVAL == 0:
            recognizer = get_gesture_recognizer()
            if recognizer is not None:
                try:
                    gesture, annotated = recognizer.process(annotated, draw=True)
                except Exception as exc:
                    log.error("Gesture recognition error: %s", exc)
                    gesture = None
                if gesture and now - _last_gesture_time >= config.GESTURE_COOLDOWN_SECONDS:
                    _last_gesture_time = now
                    state.last_gesture = gesture
                    action_name = GESTURE_ACTIONS.get(gesture)
                    event_log.record("gesture_recognized", gesture=gesture, action=action_name)
                    handler = ACTION_HANDLERS.get(action_name)
                    if handler:
                        handler()

        # --- Spoken feedback -------------------------------------------------
        if mode == "navigation":
            # Bounding boxes flicker frame-to-frame (detection noise can
            # briefly swap which object is "primary"), which used to
            # trigger a burst of rapid-fire speech - fast enough to
            # overwhelm pyttsx3's Windows driver into a stuck state that
            # silently stops producing audio. Require an instruction to
            # stay stable for NAV_DEBOUNCE_SECONDS before treating it as
            # a real change worth interrupting speech for.
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
# Voice assistant / gesture command wiring
# ---------------------------------------------------------------------------
# Both voice commands and hand gestures ultimately trigger one of these
# named actions - this is the "multimodal control" layer: whichever
# input the user prefers, the effect is identical. Voice callbacks
# receive the full heard phrase (used only by _voice_where_is); gesture
# dispatch calls them with no argument, which is why they all default it.
def speak_ocr_result(text):
    """Speak OCR text, calling out important keywords (danger/exit/etc.)
    first rather than burying them in a long read-out - shared by the
    voice command and the Capture & Read Text button."""
    if not text:
        speaker.speak("No text found.", force=True)
        return
    keyword = find_important_keyword(text)
    if keyword:
        speaker.speak(f'Attention: this text mentions "{keyword}". {text}', force=True)
    else:
        speaker.speak(text, force=True)


def _voice_read_text(heard_text=""):
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
    speak_ocr_result(text)
    event_log.record("ocr_read", via="voice/gesture", chars=len(text))


def _voice_start_navigation(heard_text=""):
    state.set_redirect("/navigation")
    if camera is None:
        start_camera()
    state.mode = "navigation"
    speaker.speak("Navigation mode activated.", force=True)
    event_log.record("mode_change", mode="navigation", via="voice/gesture")


def _voice_detect_objects(heard_text=""):
    state.set_redirect("/live")
    if camera is None:
        start_camera()
    state.mode = "object"
    speaker.speak("Object detection mode activated.", force=True)
    event_log.record("mode_change", mode="object", via="voice/gesture")


def _voice_stop_speaking(heard_text=""):
    speaker.stop()


def _voice_repeat(heard_text=""):
    speaker.repeat_last()


def _voice_describe_scene(heard_text=""):
    if not state.camera_active:
        speaker.speak("Camera is not active.", force=True)
        return
    summary = build_scene_summary(state.detections, state.zone_map)
    state.last_scene_summary = summary
    speaker.speak(summary, force=True)
    event_log.record("scene_described")


def _voice_where_is(heard_text=""):
    entry = memory.find_last(heard_text)
    speaker.speak(memory.describe(entry), force=True)


def _action_toggle_detection(heard_text=""):
    if state.camera_active:
        stop_camera()
        speaker.speak("Detection stopped.", force=True)
    else:
        start_camera()
        speaker.speak("Detection started.", force=True)
    event_log.record("detection_toggled", via="voice/gesture", active=state.camera_active)


def _voice_exit(heard_text=""):
    speaker.speak("Exiting voice assistant.", force=True)
    voice_listener.stop()
    state.voice_active = False


voice_listener.register("read text", _voice_read_text)
voice_listener.register("start navigation", _voice_start_navigation)
voice_listener.register("detect object", _voice_detect_objects)
voice_listener.register("stop speaking", _voice_stop_speaking)
voice_listener.register("repeat", _voice_repeat)
voice_listener.register("describe the scene", _voice_describe_scene)
voice_listener.register("what is around", _voice_describe_scene)
voice_listener.register("what's in front", _voice_describe_scene)
voice_listener.register("where is", _voice_where_is)
voice_listener.register("where's", _voice_where_is)
voice_listener.register("exit", _voice_exit)

# Maps a gesture's associated action name (see modules/gesture_recognizer
# .GESTURE_ACTIONS) to the handler that actually performs it, so a hand
# gesture and its equivalent spoken command do exactly the same thing.
ACTION_HANDLERS = {
    "stop speaking": _voice_stop_speaking,
    "toggle detection": _action_toggle_detection,
    "describe scene": _voice_describe_scene,
    "read text": _voice_read_text,
    "repeat last announcement": _voice_repeat,
}


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


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", active="dashboard")


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
    event_log.record("mode_change", mode=mode, via="ui")
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
    important_keyword = find_important_keyword(text) if text else None
    event_log.record("ocr_read", via="ui", chars=len(text))
    return jsonify({
        "success": True,
        "text": text,
        "boxes_found": len(boxes),
        "important_keyword": important_keyword,
    })


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
        event_log.record("currency_detected", denomination=result["denomination"])
    else:
        speaker.speak("Currency not recognized", force=True)
    return jsonify({"success": True, "result": result})


@app.route("/describe_scene", methods=["POST"])
def api_describe_scene():
    """On-demand scene summary - the same thing the "describe the
    scene" voice command / one-finger gesture trigger, exposed as a
    button for sighted testers/graders too."""
    if not state.camera_active:
        return jsonify({"success": False, "error": "Camera not active"}), 400
    summary = build_scene_summary(state.detections, state.zone_map)
    state.last_scene_summary = summary
    speaker.speak(summary, force=True)
    event_log.record("scene_described", via="ui")
    return jsonify({"success": True, "summary": summary})


@app.route("/memory/where_is", methods=["POST"])
def api_memory_where_is():
    """Text-box equivalent of the "where is my X" voice command, for
    testing without a microphone."""
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
    entry = memory.find_last(query)
    answer = memory.describe(entry)
    speaker.speak(answer, force=True)
    return jsonify({"success": True, "answer": answer, "found": entry is not None})


@app.route("/sos", methods=["POST"])
def api_sos():
    state.sos_active = True
    state.last_sos_time = time.strftime("%Y-%m-%d %H:%M:%S")
    if state.user_location:
        loc = state.user_location
        location_phrase = f" Your location is latitude {loc['latitude']:.4f}, longitude {loc['longitude']:.4f}."
    else:
        location_phrase = " Location unavailable - enable location sharing in your browser for this to include it."
    speaker.speak(
        f"Emergency S O S activated. Alerting your emergency contact.{location_phrase}",
        force=True,
    )
    log.warning("SOS TRIGGERED at %s", state.last_sos_time)
    event_log.record("sos_triggered", location=state.user_location)
    return jsonify({
        "success": True,
        "message": "SOS alert triggered (simulated).",
        "contact_name": config.SOS_EMERGENCY_CONTACT,
        "contact_phone": config.SOS_EMERGENCY_PHONE,
        "time": state.last_sos_time,
        "location": state.user_location,
    })


@app.route("/location", methods=["POST"])
def api_location():
    """Receives real coordinates from the browser's Geolocation API
    (see static/js/main.js) - this is "real GPS" in the sense that
    modern browsers resolve it from the visitor's own device (Wi-Fi/
    cell/GPS chip), not from anything the Python server has access to
    directly."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"success": False, "error": "latitude/longitude required"}), 400
    accuracy = data.get("accuracy")
    state.set_location(latitude, longitude, accuracy)
    return jsonify({"success": True, "location": state.user_location})


@app.route("/gps")
def api_gps():
    if state.user_location:
        return jsonify({
            "success": True,
            "latitude": state.user_location["latitude"],
            "longitude": state.user_location["longitude"],
            "accuracy": state.user_location.get("accuracy"),
            "source": "browser_geolocation",
            "note": "Real location reported by your browser.",
        })
    # Fallback: no location reported by the browser yet (permission
    # denied, insecure context, or JS hasn't run). Swap this with a real
    # serial/USB GPS module reader (e.g. pynmea2 + pyserial) for a
    # physical, browser-independent device build.
    return jsonify({
        "success": True,
        "latitude": config.GPS_MOCK_LAT,
        "longitude": config.GPS_MOCK_LNG,
        "source": "mock",
        "note": "Mock coordinates - no browser location reported yet (check location permission).",
    })


@app.route("/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True, silent=True) or {}
    if "battery_saver" in data:
        state.battery_saver = bool(data["battery_saver"])
        state.frame_skip = 3 if state.battery_saver else config.DEFAULT_FRAME_SKIP
        state.resize_scale = 0.6 if state.battery_saver else config.DEFAULT_RESIZE_SCALE
    if "gesture_active" in data:
        state.gesture_active = bool(data["gesture_active"])
        event_log.record("gesture_control_toggled", active=state.gesture_active)
    return jsonify({
        "success": True,
        "settings": {
            "battery_saver": state.battery_saver,
            "frame_skip": state.frame_skip,
            "resize_scale": state.resize_scale,
            "gesture_active": state.gesture_active,
        },
    })


@app.route("/api/metrics")
def api_metrics():
    """Backs the /dashboard page: live FPS/mode, event counters, recent
    event log, and basic process resource usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        cpu_percent, memory_mb = None, None

    return jsonify({
        "success": True,
        "uptime_seconds": round(event_log.uptime_seconds, 1),
        "fps": state.fps,
        "mode": state.mode,
        "camera_active": state.camera_active,
        "voice_active": state.voice_active,
        "gesture_active": state.gesture_active,
        "depth_enabled": state.depth_enabled,
        "cpu_percent": cpu_percent,
        "memory_mb": memory_mb,
        "event_counters": event_log.counters(),
        "recent_events": event_log.recent(30),
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
