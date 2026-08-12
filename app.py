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
from modules.navigation import NavigationAssistant, footpath_walkability
from modules.voice_commands import VoiceCommandListener
from modules.color_detector import detect_dominant_color
from modules.currency_detector import CurrencyDetector
from modules.qr_reader import QRReader
from modules.memory import memory
from modules.scene_summary import build_scene_summary
from modules.fall_detector import FallDetector
from modules.gesture_recognizer import GESTURE_ACTIONS
from modules.tracking import ObjectTracker
from modules.priority_engine import select_most_relevant, classify_priority
from modules.announcement_manager import AnnouncementManager
from modules.model_router import router as model_router
from utils.pipeline_settings import settings as pipeline_settings

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

# --- Auto-assistance pipeline (Sections 12/25/26/33 of the master spec) ---
# Temporal confirmation -> priority engine -> announcement manager,
# driven by whichever domain model model_router actually has available.
# Never falls back to the generic COCO model in production - see
# modules/model_router.py.
_tracking_cfg = pipeline_settings["tracking"]
object_tracker = ObjectTracker(
    min_confidence=0.4,
    min_consecutive_frames=_tracking_cfg["minimum_frames"],
    iou_match_threshold=_tracking_cfg["position_tolerance"],
    stale_after_seconds=_tracking_cfg["stale_after_seconds"],
)
announcer = AnnouncementManager(
    speaker,
    cooldown_seconds=_tracking_cfg["cooldown"],
    unknown_object_cooldown=pipeline_settings["unknown_object"]["announcement_cooldown"],
)

# Domains tried in this order when picking which custom model drives
# auto-assistance - first one that's actually AVAILABLE wins. Voice
# commands can still switch state.mode away from "auto" at any time.
AUTO_DOMAIN_PRIORITY = ["indoor", "household", "outdoor"]

# Wire config/config.yaml's confidence thresholds into the router - this
# was previously silently unused (the router's set_domain_confidence()
# was never actually called), so every domain ran at the ObjectDetector
# default (0.45) no matter what config.yaml said. `hazard` in the yaml
# maps to the router's `road_hazards` domain key.
_confidence_cfg = dict(pipeline_settings.get("confidence", {}))
if "hazard" in _confidence_cfg:
    _confidence_cfg["road_hazards"] = _confidence_cfg.pop("hazard")
model_router.set_domain_confidence(_confidence_cfg)


# Set by "indoor mode"/"outdoor mode" (voice or dashboard) to pin
# auto-assistance to one domain instead of AUTO_DOMAIN_PRIORITY's
# automatic pick; cleared by "automatic mode". None = automatic.
_manual_domain_override = None


def refresh_model_status():
    """Recompute which domain models are available and pick the active
    one for auto-assistance mode. Safe to call again later if models
    get trained/placed while the app is running."""
    state.model_status = model_router.status_report()
    if _manual_domain_override and model_router.is_available(_manual_domain_override):
        state.active_domain = _manual_domain_override
    else:
        state.active_domain = next(
            (d for d in AUTO_DOMAIN_PRIORITY if model_router.is_available(d)), None
        )
    return state.active_domain


refresh_model_status()

_detection_thread = None

_cached_depth_map = None
_last_gesture_time = 0.0
# (mode, label, zone, tier) of the last thing auto-assistance actually
# said - lets the spoken-feedback block tell "this is new information"
# apart from "the same static obstacle is still there", so a stationary
# object doesn't get re-announced on every announcer.cooldown_seconds
# (Section 10: "if nothing changes, do not repeat" - a short cooldown
# alone only limits repeat *rate*, not repetition itself).
_last_situation_key = None
# Repeat interval for an *unchanged* situation - deliberately much
# longer than announcer.cooldown_seconds (which governs how fast a
# genuinely NEW situation can interrupt); CRITICAL gets a shorter
# repeat since an ongoing hazard is worth the occasional reminder.
SAME_SITUATION_REPEAT_SECONDS = {"CRITICAL": 12.0, "HIGH": 20.0, "MEDIUM": 30.0, "LOW": 30.0}
_last_low_light_check = 0.0
_last_low_light_warning = 0.0
_low_light_session_active = False  # True while the scene has stayed dark
                                    # since the last warning/light-return

_frame_lock = threading.Lock()
_annotated_frame = None

# NOTE: there is deliberately no get_detector()/generic-COCO helper here
# any more. Every live detection path goes through modules/model_router.py,
# which only ever uses a domain's real trained model and reports honestly
# ("Detection model unavailable.") when one isn't trained yet - see the
# audit notes in README.md's Final Status Report for why the old
# COCO-backed "object" mode was removed rather than kept as a fallback.


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
    global camera, _detection_thread
    global _low_light_session_active

    if camera is None or not camera.is_opened:
        camera = VideoCamera(
            source=config.CAMERA_SOURCE, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT
        )
        if not camera.is_opened:
            log.error("Could not open camera source %s", config.CAMERA_SOURCE)
            return False

    state.camera_active = True
    _low_light_session_active = False
    object_tracker.reset()
    announcer.reset()

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
    global _annotated_frame, _last_situation_key
    global _cached_depth_map, _last_gesture_time
    global _last_low_light_check, _last_low_light_warning, _low_light_session_active

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

        active_domain = state.active_domain

        if mode in ("auto", "navigation", "color"):
            # Every live detection path routes through model_router -
            # the generic COCO model is never used here (Section 3/9/43).
            # "object" mode was removed as a separate mode: it was the
            # one remaining place COCO still ran, gated only by a weak
            # "was this label present last frame" dedup with no
            # confidence/position stability check at all - exactly the
            # gap that let a one-frame "fan" misread as "airplane" (a
            # real, well-documented COCO false positive - COCO has no
            # "fan" class, and its "airplane" class does occasionally
            # trigger on blurred spinning-blade shapes) reach the user
            # instead of being filtered out by temporal confirmation.
            detections = []
            if active_domain:
                domain_detector = model_router.get_detector(active_domain)
                if domain_detector is not None:
                    try:
                        detections = domain_detector.detect(proc_frame)
                    except Exception as exc:
                        log.error("Detection error (%s): %s", active_domain, exc)
                        detections = []
                else:
                    # The model just failed to load - re-scan so
                    # active_domain/model_status reflect reality instead
                    # of retrying every single frame.
                    refresh_model_status()
            if mode == "color":
                for det in detections:
                    det["color"] = detect_dominant_color(proc_frame, det["bbox"])
            annotated = ObjectDetector.draw_boxes(proc_frame.copy(), detections)
        else:  # mode == "qr" - independent of the object detector entirely
            detections = []
            try:
                qr_results = get_qr_reader().read(proc_frame)
            except Exception as exc:
                log.error("QR read error: %s", exc)
                qr_results = []
            state.last_qr = qr_results
            for qr in qr_results:
                speaker.speak(f"QR code found: {qr['data']}", dedup_key=f"qr:{qr['data']}")
                event_log.record("qr_detected", data=qr["data"])
            annotated = proc_frame.copy()

        h, w = proc_frame.shape[:2]

        # --- Monocular depth estimation (auto + navigation, throttled to --
        # every DEPTH_FRAME_INTERVAL frames - MiDaS is fast (~0.1-0.2s on
        # CPU) but still not free enough to run every frame).
        depth_map = None
        if mode in ("auto", "navigation") and config.ENABLE_DEPTH_ESTIMATION:
            if _cached_depth_map is None or frame_count % config.DEPTH_FRAME_INTERVAL == 0:
                estimator = get_depth_estimator()
                if estimator is not None:
                    try:
                        _cached_depth_map = estimator.estimate(proc_frame)
                    except Exception as exc:
                        log.error("Depth estimation error: %s", exc)
            depth_map = _cached_depth_map

        # --- Footpath walkability (supplementary signal, Section 6/13) ----
        # Only tried when a trained footpath model actually exists; cheap
        # single-class detector, run alongside the domain detector rather
        # than instead of it. footpath_zones stays None (not "all False")
        # when the model isn't available or hasn't confirmed any walkable
        # region in this frame, so downstream code can tell "no signal"
        # apart from "signal says nothing's walkable" - important since
        # indoors, footpath legitimately detects nothing everywhere, and
        # that must never be misread as "the path is blocked".
        footpath_zones = None
        if mode in ("auto", "navigation") and model_router.is_available("footpath"):
            footpath_detector = model_router.get_detector("footpath")
            if footpath_detector is not None:
                try:
                    footpath_dets = footpath_detector.detect(proc_frame)
                    zones = footpath_walkability(footpath_dets, w)
                    if any(zones.values()):
                        footpath_zones = zones
                except Exception as exc:
                    log.error("Footpath detection error: %s", exc)

        # --- Navigation guidance + zone map --------------------------------
        # Computed every frame regardless of mode (cheap, no model) so
        # scene summaries, "where is my X", and the Navigation page's
        # Left/Center/Right panel work no matter what mode is active.
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
        # Speaks once when the scene *becomes* dark, then stays quiet
        # about it - repeating the same "it's dark" fact every couple of
        # seconds for as long as it stays dark is exactly the kind of
        # non-repetitive-speech violation Section 10 rules out. Only a
        # long-session reminder (LOW_LIGHT_REMINDER_INTERVAL) or the
        # light actually returning resets it.
        if now - _last_low_light_check >= config.LOW_LIGHT_CHECK_INTERVAL:
            _last_low_light_check = now
            dark_now = is_low_light(proc_frame, threshold=config.LOW_LIGHT_THRESHOLD)
            if dark_now:
                if (not _low_light_session_active
                        or now - _last_low_light_warning >= config.LOW_LIGHT_REMINDER_INTERVAL):
                    _last_low_light_warning = now
                    _low_light_session_active = True
                    speaker.speak("Low light detected. Detection accuracy may be reduced.", force=True)
            else:
                _low_light_session_active = False

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
        # One unified pipeline for every detection-driven mode (Section 2):
        # temporal confirmation -> priority engine -> zone/footpath-aware
        # phrasing -> announcement manager. A detection only becomes
        # eligible for speech after appearing consistently for several
        # consecutive frames at real position stability, and even then
        # only the single most relevant one gets spoken - never every
        # detection, every frame, and never a raw single-frame reading.
        if mode in ("auto", "navigation", "color"):
            if active_domain is None:
                # No domain model trained/available at all - say so
                # honestly (Section 3) rather than staying mysteriously
                # silent or quietly falling back to a generic model.
                announcer.announce(
                    "Detection model unavailable.", key="__model_unavailable__",
                    cooldown=pipeline_settings["unknown_object"]["announcement_cooldown"],
                )
            else:
                confirmed = object_tracker.update(detections, now=now)
                candidates = []
                for track in confirmed:
                    # stable_bbox (rolling mean of recent frames) for
                    # zone/size decisions, not the single latest bbox -
                    # otherwise ordinary per-frame detection jitter near
                    # a zone boundary flips "on your left"/"ahead"/"on
                    # your right" every few seconds for an object that
                    # hasn't actually moved (confirmed during live
                    # testing - see modules/tracking.py).
                    x1, y1, x2, y2 = track.stable_bbox
                    cx = (x1 + x2) / 2
                    area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, w * h)
                    zone = navigator._zone(cx, w)
                    is_very_close = area_ratio >= pipeline_settings["navigation"]["very_near_threshold"]
                    # Only trust footpath's "not walkable here" as a real
                    # obstruction signal if it also confirmed walkable
                    # ground somewhere else in the same frame - otherwise
                    # "nothing walkable anywhere" almost always just means
                    # the footpath model doesn't apply to this scene (e.g.
                    # indoors), not that the path is blocked.
                    blocks_path = bool(footpath_zones) and not footpath_zones.get(zone, True)
                    candidates.append({
                        "label": track.label, "zone": zone, "area_ratio": area_ratio,
                        "is_very_close": is_very_close, "blocks_path": blocks_path,
                    })
                primary = select_most_relevant(candidates)
                if primary:
                    tier = classify_priority(primary["label"], primary["is_very_close"], primary["blocks_path"])
                    descriptor = None
                    if mode == "color":
                        match = next(
                            (d for d in detections if d["label"] == primary["label"] and d.get("color")),
                            None,
                        )
                        descriptor = match["color"] if match else None
                    label_text = f"{descriptor} {primary['label']}" if descriptor else primary["label"]
                    text = navigator.phrase_for(
                        label_text, primary["zone"], is_close=primary["is_very_close"],
                        tier=tier, blocked=primary["blocks_path"],
                    )
                    # Section 10: "if nothing changes, do not repeat" - a
                    # short cooldown alone only limits repeat *rate*, not
                    # repetition itself, which is exactly what let a
                    # stationary "very close" door get re-announced every
                    # ~6s indefinitely during live testing. A genuinely
                    # NEW situation (different label/zone/tier) still
                    # gets announced right away; an UNCHANGED one falls
                    # back to a much longer, tier-scaled repeat interval.
                    situation_key = (mode, primary["label"], primary["zone"], tier)
                    is_new_situation = situation_key != _last_situation_key
                    # force=True for a genuinely new situation bypasses
                    # the announcer's own per-key cooldown outright (that
                    # cooldown is keyed by label only, not zone/tier, so
                    # without this a door moving from "ahead" to "left"
                    # within 6s of the last announcement would otherwise
                    # still be suppressed as if it were a plain repeat).
                    if announcer.announce(text, key=f"{mode}:{primary['label']}", tier=tier,
                                           force=is_new_situation,
                                           cooldown=SAME_SITUATION_REPEAT_SECONDS.get(tier, 30.0)):
                        _last_situation_key = situation_key
                        state.last_auto_announcement = text
                        event_log.record("auto_announcement", text=text, tier=tier,
                                          domain=active_domain, mode=mode)
                elif detections and not confirmed:
                    # Something was detected but never stabilized into a
                    # confirmed track (position/class not stable across
                    # consecutive frames) - prefer honest uncertainty over
                    # a confident-sounding wrong guess (Section 2/15).
                    announcer.announce_unknown()
                elif footpath_zones and not footpath_zones.get("center", True) and mode != "color":
                    # Nothing detected as an obstacle, but the footpath
                    # model - having confirmed walkable ground exists
                    # somewhere in frame - says the way directly ahead
                    # specifically isn't part of it.
                    announcer.announce("Path blocked.", key="__path_blocked__", tier="HIGH")
                elif mode != "color":
                    # Genuinely clear: nothing detected, and if footpath
                    # info is available it confirms the way ahead is part
                    # of the walkable path (Section 5's "Path ahead is
                    # clear." example) - rate-limited like any other
                    # announcement so it doesn't compete for airtime.
                    announcer.announce(
                        "Path ahead is clear.", key="__clear__", tier="LOW",
                        cooldown=pipeline_settings["unknown_object"]["announcement_cooldown"],
                    )
        # mode == "qr": handled entirely above, alongside QR detection -
        # no separate object-detection speech path needed.

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
    state.processing = True
    try:
        text, _ = get_ocr_reader().read_text(frame, min_confidence=config.OCR_MIN_CONFIDENCE)
    finally:
        state.processing = False
    state.last_ocr_text = text
    speak_ocr_result(text)
    event_log.record("ocr_read", via="voice/gesture", chars=len(text))


def _voice_start_navigation(heard_text=""):
    state.set_redirect("/navigation")
    if camera is None:
        start_camera()
    state.mode = "navigation"
    object_tracker.reset()
    announcer.reset()
    speaker.speak("Navigation mode activated.", force=True)
    event_log.record("mode_change", mode="navigation", via="voice/gesture")


def _voice_detect_objects(heard_text=""):
    state.set_redirect("/live")
    if camera is None:
        start_camera()
    state.mode = "auto"
    object_tracker.reset()
    announcer.reset()
    speaker.speak("Automatic detection mode activated.", force=True)
    event_log.record("mode_change", mode="auto", via="voice/gesture")


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


def _voice_what_is_ahead(heard_text=""):
    """"What is ahead?" / "What is this?" - Section 8's situational-
    awareness query, distinct from "describe the scene": this repeats
    the single most relevant thing auto-assistance has already
    confirmed and announced, not a full object listing."""
    if not state.camera_active:
        speaker.speak("Camera is not active.", force=True)
        return
    if state.last_auto_announcement:
        speaker.speak(state.last_auto_announcement, force=True)
    else:
        speaker.speak("Nothing has been clearly identified yet.", force=True)


def _voice_identify_currency(heard_text=""):
    if camera is None and not start_camera():
        speaker.speak("Could not open the camera.", force=True)
        return
    frame = camera.get_frame()
    if frame is None:
        speaker.speak("No camera frame available yet. Try again in a moment.", force=True)
        return
    det = get_currency_detector()
    if not det.is_ready:
        speaker.speak("Currency detection is unavailable - no reference images configured.", force=True)
        return
    state.processing = True
    try:
        result = det.detect(frame)
    finally:
        state.processing = False
    state.last_currency = result
    if result:
        speaker.speak(f"{result['denomination']} rupees.", force=True)
        event_log.record("currency_detected", denomination=result["denomination"], via="voice")
    else:
        # Section 11: never guess a denomination.
        speaker.speak("Currency not recognized clearly.", force=True)


def _voice_indoor_mode(heard_text=""):
    global _manual_domain_override
    if not model_router.is_available("indoor"):
        speaker.speak("Indoor detection model is unavailable.", force=True)
        return
    _manual_domain_override = "indoor"
    refresh_model_status()
    object_tracker.reset()
    announcer.reset()
    speaker.speak("Indoor mode activated.", force=True)
    event_log.record("domain_override", domain="indoor", via="voice")


def _voice_outdoor_mode(heard_text=""):
    global _manual_domain_override
    if not model_router.is_available("outdoor"):
        speaker.speak("Outdoor detection model is unavailable.", force=True)
        return
    _manual_domain_override = "outdoor"
    refresh_model_status()
    object_tracker.reset()
    announcer.reset()
    speaker.speak("Outdoor mode activated.", force=True)
    event_log.record("domain_override", domain="outdoor", via="voice")


def _voice_automatic_mode(heard_text=""):
    global _manual_domain_override
    _manual_domain_override = None
    refresh_model_status()
    state.mode = "auto"
    object_tracker.reset()
    announcer.reset()
    speaker.speak("Automatic mode activated.", force=True)
    event_log.record("mode_change", mode="auto", via="voice/gesture")


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
voice_listener.register("read this", _voice_read_text)
voice_listener.register("start navigation", _voice_start_navigation)
voice_listener.register("detect object", _voice_detect_objects)
# Longer/more-specific phrases registered before "stop speaking" so the
# substring-matching in voice_commands.py doesn't also fire the generic
# stop-speaking handler for these more specific "stop" requests. Since
# _match_and_dispatch() dispatches to every phrase that matches (not
# just the first), a bare "stop" WILL still additionally trigger
# stop-speaking whenever one of these longer phrases is heard - that's
# harmless here (interrupting speech is a reasonable side effect of any
# "stop ..." command), so it's left as-is rather than engineered around.
voice_listener.register("stop speaking", _voice_stop_speaking)
voice_listener.register("stop", _voice_stop_speaking)
voice_listener.register("repeat", _voice_repeat)
voice_listener.register("describe the scene", _voice_describe_scene)
voice_listener.register("what is around", _voice_describe_scene)
voice_listener.register("what's in front", _voice_describe_scene)
voice_listener.register("what is ahead", _voice_what_is_ahead)
voice_listener.register("what's ahead", _voice_what_is_ahead)
voice_listener.register("what is this", _voice_what_is_ahead)
voice_listener.register("what's this", _voice_what_is_ahead)
voice_listener.register("identify this currency", _voice_identify_currency)
voice_listener.register("identify currency", _voice_identify_currency)
voice_listener.register("indoor mode", _voice_indoor_mode)
voice_listener.register("outdoor mode", _voice_outdoor_mode)
voice_listener.register("automatic mode", _voice_automatic_mode)
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
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "auto")
    # "object" is intentionally not a valid mode any more - it was the
    # one remaining place the app used the generic COCO model with no
    # temporal confirmation; see the audit notes in README.md. Existing
    # clients that still request it are treated as "auto" rather than
    # rejected outright.
    if mode == "object":
        mode = "auto"
    if mode not in ("auto", "navigation", "color", "qr"):
        return jsonify({"success": False, "error": "invalid mode"}), 400
    state.mode = mode
    if mode in ("auto", "navigation", "color"):
        # Fresh mode, fresh "what have we already said" state - carrying
        # tracks/cooldowns over from a different mode risks either a
        # stale announcement or an unnecessary re-confirmation delay.
        object_tracker.reset()
        announcer.reset()
    event_log.record("mode_change", mode=mode, via="ui")
    return jsonify({"success": True, "mode": mode})


@app.route("/detections")
def api_detections():
    return jsonify(state.snapshot())


@app.route("/status")
def api_status():
    """Minimal, focused status for the User Mode home page (Section 8) -
    deliberately a much smaller payload than /detections (which backs
    the full Developer Mode dashboard): just what a blind user's status
    view needs - is assistance active, is the camera working, what's
    the assistant doing right now, what did it last say, and what mode
    is it in. No bounding boxes, no FPS, no event log."""
    if speaker.is_speaking:
        activity = "Speaking"
    elif state.processing:
        activity = "Processing"
    elif state.voice_active:
        activity = "Listening"
    else:
        activity = "Idle"

    mode_labels = {"auto": "Automatic", "navigation": "Navigation", "color": "Color", "qr": "QR Code"}

    return jsonify({
        "assistance_active": state.camera_active or state.voice_active,
        "camera_active": state.camera_active,
        "activity": activity,
        "current_situation": state.last_auto_announcement or "Nothing identified yet.",
        "last_spoken": speaker.last_text or "",
        "mode": mode_labels.get(state.mode, state.mode.capitalize() if state.mode else "-"),
        "active_domain": state.active_domain,
        "model_status": dict(state.model_status),
    })


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
    state.processing = True
    try:
        text, boxes = get_ocr_reader().read_text(frame, min_confidence=config.OCR_MIN_CONFIDENCE)
    finally:
        state.processing = False
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
    state.processing = True
    try:
        result = det.detect(frame)
    finally:
        state.processing = False
    state.last_currency = result
    if result:
        speaker.speak(f"{result['denomination']} rupees.", force=True)
        event_log.record("currency_detected", denomination=result["denomination"])
    else:
        # Section 11: never guess a denomination.
        speaker.speak("Currency not recognized clearly.", force=True)
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
        # Model router honesty (Section 43): which domain models are
        # actually trained/loadable, which one auto-assistance is using,
        # and the last thing it actually said - never fabricated.
        "model_status": state.model_status,
        "active_domain": state.active_domain,
        "last_auto_announcement": state.last_auto_announcement,
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


def auto_start():
    """Startup sequence for voice-first, buttonless operation (Sections
    3/29/42): the assistant starts itself - camera, voice recognition,
    and AUTO ASSISTANCE mode - the instant the process launches. The
    web UI that used to require clicking "Start Camera" etc. still
    works exactly as before, but it's now a developer/debugging
    dashboard layered on top of an assistant that's already running,
    not a prerequisite for the assistant to do anything."""
    log.info("=" * 60)
    log.info("STARTUP: validating environment")
    log.info("Model availability:")
    for domain, status in state.model_status.items():
        log.info("  %-14s %s", domain, status)
    if state.active_domain:
        log.info("Auto-assistance will use the '%s' domain model.", state.active_domain)
    else:
        log.warning("No domain-specific model is available yet - auto-assistance will "
                    "stay silent on object detection until one is trained/placed under "
                    "models/<domain>/best.pt. It will NOT silently fall back to a "
                    "generic COCO model (see modules/model_router.py).")

    camera_ok = start_camera()
    if camera_ok:
        log.info("Camera: OK")
    else:
        log.warning("Camera: unavailable - the assistant will keep running (voice "
                    "commands, OCR-on-demand, currency, etc. still work), but automatic "
                    "visual detection has nothing to see until a camera is connected.")

    voice_listener.start()
    state.voice_active = True
    log.info("Voice recognition: starting")

    log.info("=" * 60)
    speaker.speak("AI assistance started.", force=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting AI Blind Assistant server on http://127.0.0.1:5000")
    auto_start()
    # use_reloader=False: avoids double-loading the YOLO/EasyOCR models
    # and spawning duplicate camera threads under Flask's auto-reloader.
    app.run(host="0.0.0.0", port=5000, debug=config.DEBUG, threaded=True, use_reloader=False)
