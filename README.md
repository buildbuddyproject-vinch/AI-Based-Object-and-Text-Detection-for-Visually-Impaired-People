# 👁️ AI-Based Object, Text, Currency and Navigation Assistant for Visually Impaired People

A final-year engineering project that helps visually impaired users
understand their surroundings using real-time object detection, printed
text recognition, spoken navigation guidance, and a hands-free voice
assistant — all running **offline** on a regular laptop with a webcam.

**The app is voice-first and buttonless by design.** Running
`python app.py` auto-opens the camera, starts text-to-speech, starts
the microphone, and begins **AUTO ASSISTANCE** immediately — no click
required. The web pages described below still exist and are fully
functional, but they are an *optional developer/debugging dashboard*,
not a requirement for the assistant to work. See
[Auto-Start / Voice-First Operation](#-auto-start--voice-first-operation).

**No face recognition of any kind is implemented or planned.** The
system can say "Person ahead." — it will never say a name. See
[No Face Recognition](#-no-face-recognition-by-design).

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Auto Assistance | Starts automatically on launch — camera, TTS, mic, detection loop, all with zero clicks |
| 🧠 Model Router | Loads a real trained model per domain (indoor/household/outdoor/footpath/currency); a domain with no trained model is honestly reported `NOT AVAILABLE`, never silently swapped for a generic COCO model |
| 🎯 Temporal Confirmation | An object must be seen consistently for several consecutive frames before it's announced — a single noisy misread (e.g. a spinning fan momentarily read as "airplane") is filtered out, never spoken |
| 📢 Priority Engine + Announcement Manager | Only the single most relevant detection is spoken per cycle (CRITICAL obstacles pre-empt everything else), with cooldowns so the same object isn't repeated every frame |
| 🏠 Home Page | Optional dashboard landing page with large buttons and clear navigation |
| 📷 Live Camera | Real-time YOLOv8 object detection with bounding boxes + confidence (dashboard view of the same detection loop driving auto assistance) |
| 📖 OCR Mode | Capture printed text with EasyOCR, reading-order-sorted, hear it read aloud |
| ⚠️ Important-Text Priority | Warnings/exits/hazards in OCR text are called out first, not buried |
| 🧭 Navigation Mode | Left / Center / Right obstacle guidance, spoken instructions |
| 📏 Monocular Depth Estimation | MiDaS-based relative depth refines "very close" / "farther away" |
| 🗣️ Scene Summary | On-demand natural-language description of everything currently detected |
| 🧠 Contextual Memory | "Where is my bag?" — answers from a short-term detection history |
| 🎙️ Voice Assistant | Hands-free control via microphone commands (this is the primary interface, not a bonus mode) |
| ✋ Hand Gesture Control | Static gestures (fist, open palm, peace, thumbs up) trigger the same actions as voice |
| 🤕 Fall Detection (experimental) | Heuristic bounding-box-collapse check with a spoken safety prompt |
| 🔊 Audio Feedback | Offline pyttsx3 speech, de-duplicated so it doesn't repeat itself |
| ⚡ Performance | Threaded camera capture, frame-skip, battery saver mode, optional ONNX inference |
| 📊 System Dashboard | Live FPS/CPU/memory, per-domain model AVAILABLE/NOT AVAILABLE status, feature status, and an event log |
| 🎨 Color Detection (bonus) | Names the dominant color of a detected object |
| 💵 Currency Detection (bonus) | ORB feature-matching against real reference note images in `dataset/currency/` |
| 🔳 QR Code Reader (bonus) | Detects and reads QR codes aloud |
| 🆘 Emergency SOS (bonus) | One-tap simulated emergency alert, includes real location if available |
| 📍 Real GPS (bonus) | Browser Geolocation API reports the visitor's actual location |
| 🔋 Battery Saver (bonus) | Reduces frame rate / resolution to save CPU/battery |

---

## 🧰 Tech Stack

**Backend:** Python, Flask
**AI/CV:** YOLOv8 (Ultralytics, optionally ONNX Runtime), EasyOCR, MiDaS (monocular depth,
via torch.hub), MediaPipe (hand gestures), OpenCV, pyttsx3, SpeechRecognition
**Frontend:** HTML, CSS, JavaScript (responsive, accessibility-first, dark mode)

---

## 📁 Project Structure

```
AI-Based Object and Text Detection for Visually Impaired People/
│
├── app.py                     # Entry point: auto_start() runs camera+TTS+mic+detection
│                               # loop before the Flask dev server even starts listening
├── config.py                  # Legacy central configuration (weights paths, thresholds)
├── config/
│   └── config.yaml            # Pipeline settings: tracking, priority, speech, navigation
├── download_weights.py        # Convenience script to fetch yolov8n.pt
├── export_onnx.py             # Optional: export YOLOv8 to ONNX for faster CPU inference
├── requirements.txt
├── requirements-render.txt    # Trimmed dependency set for cloud deployment
├── render.yaml                # Render Blueprint config
├── README.md
├── .gitignore
│
├── modules/
│   ├── object_detector.py     # YOLOv8 wrapper (loads .pt or .onnx transparently)
│   ├── model_router.py        # Per-domain model registry - lazy load, never a silent
│   │                           # COCO fallback (see Model Router section below)
│   ├── tracking.py            # Temporal confirmation (IoU tracker) - filters one-frame noise
│   ├── priority_engine.py     # CRITICAL/HIGH/MEDIUM/LOW tiers, picks the most relevant object
│   ├── announcement_manager.py# Cooldown/dedup speech gate, CRITICAL interrupts current speech
│   ├── ocr_reader.py          # EasyOCR wrapper + reading-order sort + keyword priority
│   ├── speaker.py             # Subprocess-isolated, de-duplicated pyttsx3 TTS
│   ├── navigation.py          # Left/Center/Right guidance + optional depth-aware distance
│   ├── depth_estimator.py     # MiDaS-small monocular relative-depth estimation
│   ├── voice_commands.py      # Microphone command listener
│   ├── gesture_recognizer.py  # MediaPipe static hand-gesture recognition
│   ├── fall_detector.py       # Heuristic, experimental fall detection
│   ├── memory.py              # "Where is my X" contextual detection memory
│   ├── scene_summary.py       # Template-based natural-language scene description
│   ├── color_detector.py      # (bonus) dominant color naming
│   ├── currency_detector.py   # (bonus) ORB feature-matching against dataset/currency/
│   └── qr_reader.py           # (bonus) QR code detection
│
├── utils/
│   ├── camera_stream.py       # Threaded webcam reader
│   ├── frame_utils.py         # Resize / JPEG encode / FPS tracker / low-light check
│   ├── logger.py              # Console logger
│   ├── event_log.py           # Structured event log powering /dashboard
│   ├── pipeline_settings.py   # Loads config/config.yaml (lives outside config/ package -
│   │                           # see the comment in the file for why)
│   └── state.py                # Thread-safe shared app state (mode, model_status, active_domain...)
│
├── templates/                 # Jinja2 pages - now labelled as the optional dashboard
├── static/
│   ├── css/style.css           # Blue accessibility theme + dark mode
│   └── js/                     # user_mode.js (Home/status), camera.js, ocr.js,
│                               # navigation.js, voice.js, dashboard.js, main.js
│
├── tools/
│   ├── dataset_common.py       # Shared watermark-filtering + VOC-box-conversion helpers
│   ├── analyze_datasets.py     # Inspects dataset/ and reports real format/classes/counts (read-only)
│   ├── prepare_datasets.py     # Converts dataset/ -> dataset_prepared/<domain>/ in YOLO format
│   └── test_false_positives.py # Scripted false-positive/safety-behavior scenarios
│
├── training/
│   ├── train_indoor.py         # Trains models/indoor/best.pt
│   ├── train_household.py      # Trains models/household/best.pt
│   ├── train_footpath.py       # Trains models/footpath/best.pt
│   └── train_outdoor.py        # Documents WHY outdoor can't train yet (no source images)
│
├── weights/                    # Generic COCO yolov8n.pt (DEVELOPMENT_MODE only, never production)
├── models/                     # Per-domain trained weights - models/<domain>/best.pt
│   ├── indoor/ household/ outdoor/ road_hazards/ currency/ footpath/
├── dataset/                    # Real source datasets (never modified/reorganized by any tool)
│   ├── currency/                # Reference note images (ORB matching, not a trainable set)
│   ├── Footpath/                # Pascal VOC XML, single class "footpath"
│   ├── household/                # Supervisely JSON, 138 theme folders, 94 classes
│   ├── indoor/                   # Roboflow YOLO export, 10 classes, train/valid/test
│   └── outdoor/                  # Pascal VOC XML (13 real classes) + unpaired OBB labels
├── dataset_prepared/            # Generated by tools/prepare_datasets.py - git-ignored, YOLO-ready
├── runs/                        # Ultralytics training run output (results.csv, PR curves, etc.)
├── logs/                        # Structured event log output (git-ignored)
├── screenshots/                 # Add demo screenshots here
└── tests/                       # Unit tests (no camera/mic required)
```

---

## 🛠️ Installation Guide

### 1. Prerequisites

- **Python 3.10 or 3.11** recommended (best current compatibility with
  `torch`/`ultralytics`/`easyocr` wheels).
- A working webcam.
- (Optional, for the voice assistant) a working microphone.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The first install downloads `torch`/`torchvision` (used internally by
> Ultralytics and EasyOCR) and can take several minutes / a few hundred
> MB depending on your connection.

### 4. Download the YOLOv8 weights

```bash
python download_weights.py
```

This saves `yolov8n.pt` (~6 MB) into `weights/`. If you skip this step,
the app will still work — `ObjectDetector` automatically falls back to
Ultralytics' built-in auto-download the first time you open the Live
Camera page — but running the script ahead of time keeps startup fast
and the folder structure tidy.

*(Manual alternative: download `yolov8n.pt` from
https://github.com/ultralytics/assets/releases and place it at
`weights/yolov8n.pt`.)*

**Other models download automatically on first use**, the same
lazy-download pattern - no separate setup step needed: EasyOCR's models
(first OCR capture), MiDaS depth estimation weights (first time
Navigation mode runs, via `torch.hub`), and the hand gesture landmarker
model (first time Gesture Control is enabled). All need internet access
the *first* time only; everything is cached locally afterward.

### 5. (Optional) Export YOLOv8 to ONNX for faster CPU inference

```bash
python export_onnx.py
```

Then point `YOLO_WEIGHTS_PATH` in `config.py` at the resulting
`weights/yolov8n.onnx` - Ultralytics loads either format transparently,
no other code changes needed. Measured on this project's dev machine:
~2.2x faster per-frame inference on CPU (0.061s vs 0.136s).

### 6. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## ▶️ How to Use

**None of the steps below are required to get assistance** — `python
app.py` already starts the camera, TTS, mic, and AUTO ASSISTANCE by
itself (see [Auto-Start / Voice-First Operation](#-auto-start--voice-first-operation)).
What follows documents the **optional dashboard** pages, for anyone
who wants to see/demo what the assistant is doing, and the voice
commands that work regardless of whether the dashboard is ever opened.

1. **Home (User Mode)** — the actual primary interface (Section 8):
   a minimal, read-only status view - "AI ASSISTANCE ACTIVE", camera
   status, Listening/Processing/Speaking, current situation, last
   spoken message, current mode. Nothing on it needs to be clicked.
2. **Developer Mode** (Live Camera / OCR / Navigation / Voice /
   Dashboard, clearly labelled as such in the nav) — visual debugging
   views of the same auto-started pipeline, for demos/development, not
   for the blind end user:
   - **Live Camera** — mode switch (Auto / Color / QR Code), bounding
     boxes + confidence, **💵 Detect Currency**, **🗣️ Describe Scene**,
     **✋ Gesture Control** toggle.
   - **OCR** — **Capture & Read Text** then **Read Aloud**, same as
     saying "read this". Reading order is top-to-bottom, and a toast
     calls out safety-relevant keywords (exit, danger, wet floor, ...)
     before you even read the rest.
   - **Navigation** — visual Left/Center/Right obstacle panel; guidance
     is spoken automatically once this mode is active, using real
     per-pixel MiDaS depth when it's loaded, never a fabricated distance.
   - **Dashboard** — live FPS/CPU/memory, per-domain model
     AVAILABLE/NOT AVAILABLE status, active domain, last spoken
     announcement, and a running event log.
3. **Voice Assistant** (works everywhere, no page needs to be open):
   - `"what is ahead"` / `"what is this"` — repeat the last confirmed situational announcement
   - `"read this"` / `"read text"` — capture + read printed text aloud
   - `"identify this currency"` — capture + run currency detection
   - `"describe the scene"` / `"what is around"` — full summary of everything currently detected
   - `"start navigation"` / `"automatic mode"` / `"indoor mode"` / `"outdoor mode"` — switch modes
   - `"stop speaking"` / `"stop"` — interrupt current speech
   - `"repeat"` — say the last announcement again
   - `"where is my bag"` (or any recently-seen object) — answered from short-term memory
   - `"exit"` — stop the voice assistant
4. **Hand Gestures** — enable "Gesture Control" on the Live Camera page,
   then hold a hand up to the camera: **fist** = start/stop detection,
   **open palm** = stop speaking, **peace sign** = read text, **one
   finger** = describe the scene, **thumbs up** = repeat. Same actions
   as the voice commands above — use whichever input suits the moment.
5. Dashboard also shows: live FPS, CPU/memory usage, which features are
   currently active, and a running event log (mode changes, SOS
   triggers, detections, gestures, etc.) — useful for a demo/report.
6. **SOS** — the red **SOS** button in the header is available on every
   page and triggers a simulated emergency alert (spoken + logged),
   including your real location if the browser reported one.
7. **Dark Mode** — toggle via the 🌙/☀️ icon in the header; your choice
   is remembered.
8. **Battery Saver** — toggle on the Live Camera page to reduce frame
   rate and resolution for lower CPU/battery usage.

---

## 🎙️ Auto-Start / Voice-First Operation

Running `python app.py` does all of the following **before you touch
anything**:

1. Scans `models/<domain>/best.pt` for every domain and honestly logs
   which are `AVAILABLE` vs `NOT AVAILABLE` (see [Model Router](#-model-router--no-silent-coco-fallback)).
2. Opens the default camera.
3. Starts the background detection loop, which runs temporal
   confirmation ([modules/tracking.py](modules/tracking.py)) and the
   priority engine ([modules/priority_engine.py](modules/priority_engine.py))
   on every frame.
4. Starts the microphone / voice-command listener.
5. Speaks **"AI assistance started."** and begins **AUTO ASSISTANCE**
   — the assistant proactively announces the single most relevant
   thing in front of the camera (never everything at once - see
   [Priority Engine + Announcement Manager](#-priority-engine--announcement-manager)),
   using whichever domain model is available (indoor → household →
   outdoor, in that priority order, per `AUTO_DOMAIN_PRIORITY` in `app.py`).

No page load, no click, and no keyboard input is required for any of
this. The Flask server that starts afterward (`http://127.0.0.1:5000`)
exposes the exact same running state as an **optional** visual
dashboard - useful for demoing/debugging what the assistant is
currently seeing and saying - not as a prerequisite for the assistant
to function. Voice commands (see [How to Use](#️-how-to-use), step 5)
work identically whether or not anyone ever opens the dashboard.

If **no** domain model is available yet (a fresh checkout before any
training has been run), the app still auto-starts everything except
detection announcements, and says so honestly rather than silently
running a generic COCO model — see the next section.

---

## 🧠 Model Router — No Silent COCO Fallback

[`modules/model_router.py`](modules/model_router.py) is the single
place that decides which weights file backs each domain
(`indoor`, `household`, `outdoor`, `road_hazards`, `currency`,
`footpath`). Its rule, enforced in code, not just documentation:

> **A domain with no trained `models/<domain>/best.pt` is reported
> `NOT AVAILABLE` and stays disabled. It is never silently replaced by
> the generic COCO-pretrained `yolov8n.pt`.**

A generic COCO detector can still be loaded for manual
development/testing via `get_development_coco_detector()` — but only
if the `DEVELOPMENT_MODE=1` environment variable is set explicitly;
without it, calling that method raises instead of running, so
production code paths can never reach it by accident.

Check what's currently available:
```bash
python -c "from modules.model_router import router; print(router.status_report())"
```
or watch the app's own startup log, or open the dashboard's home page
(model status is included in the `/detections` API response as
`model_status` / `active_domain`).

**Two of the six listed domains need a caveat, to avoid the status
report being misread:**
- **`currency`** will always show `NOT AVAILABLE` here - there is no
  trained YOLO classifier for it. Currency detection still genuinely
  works, but through a completely separate, pre-existing code path
  (`modules/currency_detector.py`, ORB feature-matching against
  `dataset/currency/*.png`) that never goes through `model_router` at
  all. `NOT AVAILABLE` here means "no trained currency *object
  detector* exists," not "currency detection doesn't work."
- **`road_hazards`** is a placeholder domain slot with no dataset, no
  prepare/train script, and no `AUTO_DOMAIN_PRIORITY` entry yet - none
  of the 5 real datasets under `dataset/` map cleanly to it today
  (the closest candidate, pothole/speed-breaker classes, isn't present
  in any of them - see [Final Status Report](#-final-status-report)).
  It's listed now so the router's shape doesn't need to change later.

**Update (post-launch audit):** an earlier version of this README
described the dashboard's Live Camera "Object" mode as a deliberate
exception that still used the generic `yolov8n.pt`. A live-testing
audit (see [Audit & Redesign Findings](#-audit--redesign-findings))
found this was actually the root cause of a real "fan detected as
airplane"-style false positive, and worse, that mode had **no
temporal confirmation at all** - just a weak "was this label present
last frame" check. Both the generic-COCO usage and the standalone
"object" mode have since been **removed entirely**. Every live
detection path - `auto`, `navigation`, and `color` - now goes through
the exact same `model_router` + temporal-confirmation + priority-engine
pipeline described above, with zero exceptions. If no domain model is
available, the dashboard now honestly says "Detection model
unavailable." instead of ever touching a generic model.

---

## 🎯 Temporal Confirmation & 📢 Priority Engine + Announcement Manager

Two problems a raw per-frame YOLO announcement loop has in practice:

1. **Single-frame misclassification.** A spinning ceiling fan can, for
   one noisy frame, score higher for "airplane" than for its real
   class. Announcing every single frame's raw detections would say
   "airplane" out loud from a false positive most detectors produce
   occasionally.
2. **Announcement spam.** With a person standing in frame, a raw loop
   says "Person ahead" 20+ times a second - useless and actively
   dangerous if it drowns out something newly relevant.

[`modules/tracking.py`](modules/tracking.py)'s `ObjectTracker` fixes
(1): a detection only becomes "confirmed" after appearing in
`min_consecutive_frames` (default 4, `config/config.yaml` →
`tracking.minimum_frames`) consecutive frames, matched frame-to-frame
by label + IoU overlap. A one-frame "airplane" blip never reaches 4
consecutive frames and is never spoken. See
[`tools/test_false_positives.py`](tools/test_false_positives.py)
Scenario 1 for this exact case, verified against the real module.

[`modules/priority_engine.py`](modules/priority_engine.py) fixes (2)
from the other direction: every confirmed detection is classified into
a `CRITICAL` / `HIGH` / `MEDIUM` / `LOW` tier (obstacles that block the
path or are very close outrank everything else), and
`select_most_relevant()` picks exactly one candidate per cycle.
[`modules/announcement_manager.py`](modules/announcement_manager.py)
then applies a per-object cooldown (default 6s) so that one candidate
isn't repeated every cycle either — except `CRITICAL` announcements,
which interrupt whatever is currently being spoken (`speaker.stop()`)
regardless of cooldown, since an active hazard must never wait behind
an unrelated sentence.

When nothing is confidently recognized, the assistant says **"Object
not clearly recognized."** — a fixed, honest phrase — rather than
guessing a label with low confidence.

---

## 🚫 No Face Recognition, By Design

This project deliberately implements **zero** face recognition, face
identification, face embeddings, or any form of "who is this person"
capability - and never will. Verified by grepping the entire codebase
for `face_recognition`, `deepface`, `facenet`, `dlib` face-encoding
APIs, and any local "known faces" database: none exist.

The generic object detector can and does say **"Person ahead."** when
a person is confidently, temporally-confirmed detected — that's a
normal COCO/custom-model object class like any other. It will never
say a name, because there is no code path anywhere that could produce
one.

---

## 📊 Dataset Pipeline

Real datasets already exist under `dataset/` — `currency/`,
`Footpath/`, `household/`, `indoor/`, `outdoor/`. Two tools work
against them, in order, and **neither ever renames, deletes, moves, or
reorganizes anything inside `dataset/`** — both are strictly read +
write-elsewhere:

### 1. Analyze (read-only)

```bash
python tools/analyze_datasets.py --json tools/dataset_report.json
```

Inspects each dataset's actual on-disk format (no assumptions) and
reports image/annotation counts, the real class list, splits, and data
quality problems (missing labels, watermark/junk classes, empty
annotations). Findings that mattered for this project, and directly
contradicted some initial assumptions:

- **Footpath is bounding-box object detection, not segmentation** —
  every XML has `<segmented>0</segmented>` despite the folder name.
- **household is rich multi-class object detection (94 real classes:
  Shoe, Cup, Cooking pot, Hand, Toothbrush, Plate, ...), not simple
  per-folder image classification** — the theme folder name (e.g.
  `Objects__armchairs`) is only the photo's *source collection*, not
  the object's label; several unrelated classes are annotated per photo.
- **outdoor's `XML Files/` mixes 1,013 real annotations (car, pole,
  truck, flyover, hoarding, traffic symbols, pedestrian, ...) with
  2,899 Roboflow watermark/version-label annotations** that must be
  filtered out before training (see `tools/dataset_common.py`).
- **outdoor's `labels/` OBB set has undocumented, inconsistent
  class-id mappings** across its Day/Foggy/Rainy subsets (Day uses ids
  0-3, Rainy uses ids 0-13, no `classes.txt` anywhere) — not used for
  training until that's resolved.

### 2. Prepare (writes only to `dataset_prepared/`)

```bash
python tools/prepare_datasets.py --dataset indoor
python tools/prepare_datasets.py --dataset footpath
python tools/prepare_datasets.py --dataset household --min-class-count 20 --max-images 6000
python tools/prepare_datasets.py --dataset outdoor
python tools/prepare_datasets.py --dataset all
```

Converts each dataset's real, native format into a standard YOLO
layout (`dataset_prepared/<domain>/{train,val}/{images,labels}` +
`data.yaml`), filtering out watermark/junk classes and rare classes
(`--min-class-count`, default 20 examples) along the way. `dataset/`
itself is never touched — confirmed by every prepare function copying
files into `dataset_prepared/`, never writing back.

---

## 🏋️ Model Training

Each trainable domain has its own script under `training/`, run after
the matching `prepare_datasets.py` step:

```bash
python training/train_indoor.py     [--epochs 30] [--imgsz 640] [--batch 16]
python training/train_footpath.py   [--epochs 50]   # small dataset, heavier augmentation
python training/train_household.py  [--epochs 25] [--imgsz 416]
python training/train_outdoor.py    # does NOT train - explains why, see below
```

All start from `yolov8n.pt` as a transfer-learning backbone (standard
practice — the pretrained COCO weights are a generic feature-extractor
starting point, not used for inference) and fine-tune entirely on the
domain's own real classes. Each prints real, measured
precision/recall/mAP50/mAP50-95 from `model.val()` on a held-out split
— never a fabricated or assumed number — and copies the trained
`best.pt` to `models/<domain>/best.pt`, where `model_router.py` picks
it up automatically on the app's next restart.

See [Final Status Report](#-final-status-report) for this project's
actual, currently-measured results per domain, including what's
**not** trainable yet and why.

---

## 🧪 How to Test

Automated unit tests cover the pure-logic modules (navigation heuristics
including depth-aware distance, color naming, QR/currency wrappers,
fall detection, contextual memory, scene summary, OCR reading order and
keyword priority, temporal tracking, priority engine, announcement
manager, model router, pipeline settings) without needing a camera,
GPU, or microphone:

```bash
python -m unittest discover -s tests -v
```

Scripted false-positive / safety-behavior scenarios (the fan/airplane
single-frame-misread case, cooldown/CRITICAL-interrupt behavior, the
model router's no-silent-fallback guarantee, priority-tier ordering)
against the real modules, with a readable PASS/FAIL report:

```bash
python tools/test_false_positives.py
```

**Manual end-to-end test checklist:**
- [ ] `python app.py` starts without errors and prints the server URL.
- [ ] Home page loads with working navigation links and dark mode toggle.
- [ ] Live Camera: clicking "Start Camera" shows a live video feed within
      a few seconds; bounding boxes + labels appear on recognized objects;
      new objects trigger spoken announcements.
- [ ] OCR: capturing a page of printed text extracts readable text and
      "Read Aloud" speaks it.
- [ ] Navigation: moving an object left/right across the frame changes
      the spoken Left/Center/Right guidance.
- [ ] Voice Assistant: saying "read text" while the camera is on triggers
      an OCR capture + speech.
- [ ] Voice Assistant: saying "where is my bag" after a backpack has been
      detected answers with how long ago and where it was seen.
- [ ] Gesture Control: enabling it and holding up a fist/open palm/peace
      sign/thumbs up in good lighting triggers the matching action.
- [ ] Navigation: after MiDaS loads (check the console log), guidance
      mentions "very close" for a large, near object.
- [ ] Dashboard shows live FPS/CPU/memory and updates its event log as
      you interact with other pages.
- [ ] SOS button shows a toast with the configured emergency contact and,
      if location permission was granted, real coordinates.
- [ ] Toggling Battery Saver reduces CPU usage (check Task Manager).

---

## 📸 Expected Output

- The **Live Camera** page shows your webcam feed with colored bounding
  boxes around detected objects (e.g. `person 92%`, `chair 87%`), and
  each newly-seen object is announced once, e.g. *"person detected"*.
- The **OCR** page displays the extracted text in a textbox after
  capture, and reads it aloud on request.
- The **Navigation** page shows a short spoken sentence like *"Chair on
  your left."* or *"Person ahead. It's very close. Move slightly
  right."* (the "It's very close" part only appears once depth
  estimation has loaded), plus a Left/Center/Right panel listing
  everything currently in each zone.
- The **Voice Assistant** page logs every recognized phrase and which
  command (if any) it matched.
- The **Dashboard** page shows live FPS/CPU/memory numbers, ON/OFF
  badges for camera/voice/gesture/depth, and a running event log.

See `screenshots/` for where to add your own captured screenshots for a
project report.

---

## 🩺 Troubleshooting

**`PyAudio` fails to install on Windows**
```powershell
pip install pipwin
pipwin install pyaudio
```
or download the matching pre-built wheel from Christoph Gohlke's
Unofficial Windows Binaries page and `pip install <file>.whl`.

**Webcam doesn't open / `Could not open camera`**
- Close any other app using the camera (Zoom, Teams, etc.).
- Try a different `CAMERA_SOURCE` in `config.py` (0, 1, 2...).

**No sound from pyttsx3 even though the app reports success and no errors**
- Root cause (confirmed): `pyttsx3`'s Windows (SAPI5) driver has a
  well-documented bug where **reusing one long-lived engine instance**
  across many `say()`/`runAndWait()` calls works for the very first call
  and then silently stops producing audio for every call after that -
  `runAndWait()` returns almost instantly without ever actually
  rendering speech, and raises no exception. Worse, on this project's
  test machine, repeatedly recreating engines *inside one long-running
  process* eventually **segfaulted the whole server** outright, from
  inside pyttsx3's native COM driver, uncatchable by any Python
  try/except. `modules/speaker.py` avoids both problems by re-invoking
  itself as a **short-lived subprocess for every single utterance**
  (`python -m modules.speaker --speak-worker`) - one process, one
  engine, one utterance, then it exits. A crash there can never take
  the server down, and every utterance always starts from clean state.
- A separate, unrelated Windows quirk can also silently mute output: the
  OS persists a per-application volume in the Volume Mixer keyed by
  executable name, and `python.exe`/`python3.11.exe` can end up pinned
  to a near-zero level left over from an unrelated earlier session,
  sometimes not fixable by dragging the mixer slider or clicking
  "Reset". Because each utterance is now a brand-new process, this can
  recur on *every* utterance, not just once - so `_force_full_volume_windows()`
  runs inside the worker subprocess itself (via `pycaw`), concurrently
  polling for ~0.1s intervals for as long as that one utterance plays
  (confirmed by direct measurement: the audio session isn't visible to
  WASAPI until playback actually starts, so a single check beforehand
  isn't enough). This is safe specifically *because* it's scoped to one
  subprocess that only ever speaks once - the same approach inside the
  old long-lived process was what caused the instability above.
- If you still hear nothing: confirm your system's default *output
  device* (Settings → System → Sound → Output) is actually your
  speakers/headphones and not an unplugged HDMI/virtual device, and
  check for audio-enhancement software (e.g. Nahimic) that mutes
  unrecognized apps in its own mixer.

**No sound from pyttsx3 on Linux (e.g. a cloud host)**
- pyttsx3's Linux backend needs `espeak`/`espeak-ng` installed as an
  actual system package - most cloud Python buildpacks don't include
  it. The TTS worker fails gracefully (logs a clear message, doesn't
  crash) in that case; see `README.md` -> "Deployment".

**MiDaS (depth estimation) or the hand gesture model fails to load**
- Both download their model files on first use - MiDaS via `torch.hub`
  (`modules/depth_estimator.py`), the hand landmarker via a direct URL
  (`modules/gesture_recognizer.py`) - and need internet access the
  first time. If the download fails (no internet, or a corporate
  proxy/firewall blocking GitHub/Google Cloud Storage), the feature
  fails closed: navigation falls back to the bounding-box-size
  heuristic, and gesture control simply reports nothing recognized.
  Neither failure crashes the app - see `get_depth_estimator()` /
  `get_gesture_recognizer()` in `app.py`.
- `torch.hub` may print an interactive trust-confirmation prompt for
  MiDaS's EfficientNet backbone dependency - already patched around in
  `modules/depth_estimator.py` (see the comment there) since it would
  otherwise hang forever waiting for stdin in a server process.

**Voice commands aren't recognized**
- The default recognizer (`recognize_google`) needs an internet
  connection. For a fully offline setup, integrate
  [Vosk](https://alphacephei.com/vosk/) inside
  `modules/voice_commands.py::_recognize()` (left as an extension point).
- Check your microphone is set as the default input device.

**YOLO / EasyOCR is slow on first run**
- Both download model weights on first use (EasyOCR caches to
  `~/.EasyOCR`). Subsequent runs are much faster.
- Enable **Battery Saver** mode, or lower `YOLO_CONFIDENCE` /
  `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py` for higher FPS on modest
  hardware.

**No GPU available**
- Everything defaults to `device="cpu"` in `config.py` — a GPU is not
  required, though inference will run faster with one (`YOLO_DEVICE=0`).

---

## 🌐 Offline Mode Notes

- Object detection (YOLOv8), OCR (EasyOCR), text-to-speech (pyttsx3),
  depth estimation (MiDaS), and hand gesture recognition (MediaPipe) all
  run **fully offline** once their model weights are downloaded on
  first use.
- Only the default voice-command speech-to-text step
  (`SpeechRecognition`'s Google Web Speech API) requires internet at
  runtime; swap it for Vosk (see Troubleshooting) to go fully offline
  end-to-end.
- The **real GPS** feature (browser Geolocation API) needs the
  visitor's device to have network/GPS connectivity to resolve a
  location - that's inherent to what "real location" means, not
  something this app can make offline.

---

## ☁️ Deployment

### ⚠️ Read this before deploying to Render, Vercel, or any cloud host

This app talks to hardware **on the machine running `app.py`**, not the
visitor's browser:

- `cv2.VideoCapture(0)` in `utils/camera_stream.py` opens whatever
  camera is physically attached to that machine.
- `pyttsx3` (offline TTS) and the voice assistant's microphone
  (`SpeechRecognition`) need local audio output/input hardware.
- Hand gesture control needs a camera frame to look at, so it's
  unavailable for the same reason.

Run locally, that machine is your own laptop/PC, so "the camera" and
"the speakers" correctly mean *your* webcam and speakers. Deployed to a
cloud server, "that machine" is Render's container - which has no
webcam, no microphone, and no speakers. The pages will load and the
UI/API will respond, but **Live Camera, OCR, Navigation, Voice
Assistant, Gesture Control, and all audio narration will not work for
anyone visiting the deployed URL** - `camera.start()` fails to open a
nonexistent device, the mic fails to initialize, and TTS has no audio
driver to speak through. Both failures are handled gracefully (a
toast/log message, not a crash) but the features are simply
unavailable.

One genuine exception: **real GPS location works fine on a cloud
deployment**, because it's resolved by the *visitor's own browser*
(`navigator.geolocation`), not the server - `/sos` and `/gps` will
correctly reflect wherever the person viewing the page actually is.

This is fine for demoing the UI, the object-detection/OCR/navigation
*code paths* against your own machine, or the non-hardware endpoints
(Home, About, SOS, GPS, Settings, Dashboard) - it is not a way to give
remote visitors a working camera feed. Making that actually work for
remote visitors would mean moving camera capture to the browser
(`getUserMedia()`, frames POSTed to the server for inference) and
TTS/STT to the browser's Web Speech API - a real frontend rework, not a
config change.

### Deploying to Render (demo / UI only, per above)

1. Push this repo to GitHub (already done if you're reading this from
   there).
2. In Render: **New -> Blueprint**, point it at this repo - it will
   read `render.yaml` and set everything up automatically. Or configure
   a **Web Service** manually with:
   - Build command: `pip install -r requirements-render.txt && python download_weights.py`
   - Start command: `gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`
   - Env var: `FLASK_DEBUG=0`
3. `requirements-render.txt` (not `requirements.txt`) is used
   deliberately - it drops `PyAudio` (needs system build headers Render's
   Python buildpack doesn't have) and `mediapipe` (pulls in
   `opencv-contrib-python`, which needs system X11/libGL libraries a
   minimal container usually lacks and could break the whole app's
   ability to import `cv2`). Both would otherwise risk failing the
   *entire* install; neither is a real loss since both need a camera the
   cloud host doesn't have anyway.
4. **Memory**: `torch` + `ultralytics` + `easyocr` (+ `timm` for MiDaS if
   depth estimation loads) loaded together can approach or exceed
   Render's free-tier 512 MB RAM limit, especially once a request
   actually triggers OCR/detection - measured locally with everything
   loaded (YOLO + EasyOCR + MiDaS), this app used **~960 MB**. If the app
   gets OOM-killed under load, that's why - a paid tier with more RAM
   avoids this, or set `ENABLE_DEPTH_ESTIMATION=0` to trim it down.
5. **Cold starts**: the free tier spins down after inactivity; the
   first request after that will be slow (YOLO/EasyOCR loading, and
   `download_weights.py` already ran at build time so at least that
   part is cached).
6. `pyttsx3` on Linux needs an `espeak`/`espeak-ng` binary to have any
   voice at all - Render's default Python environment doesn't include
   one, so speech will no-op (logged, not fatal) even for API calls
   that don't need a camera.

### Deploying to Vercel

**Not supported for this app**, independent of the hardware issue
above. Vercel is a serverless-functions platform with per-invocation
time limits and package-size limits; this app needs a persistent,
stateful, long-running process (a background detection thread, a live
MJPEG stream, in-memory state shared across requests) and ships with
`torch`/`ultralytics`/`easyocr`, which are far too large for a
serverless function bundle. Use Render (or any host that runs a
persistent Python process - Railway, Fly.io, a plain VPS, etc.)
instead.

---

## 🔍 Audit & Redesign Findings

After the initial auto-assistance build (above), real live-camera
testing surfaced two categories of problems: detection reliability and
a UI that read like a visual dashboard rather than an assistive tool.
This section documents what was actually found and fixed - every claim
below was verified against real running logs, not assumed.

### A duplicate-process bug was corrupting test results

The very first thing the audit found wasn't a code bug at all: **two
`app.py` processes were both bound to port 5000 simultaneously** - the
current one, and an hours-old zombie left running from a much earlier
testing session, before any of the model-router/tracking work existed.
Requests were being non-deterministically routed to whichever process
the OS picked, which explains a lot of the inconsistent behavior seen
before this audit (e.g. a `/camera/stop` call silently landing on the
dead process while the real one kept running). Confirmed via the
zombie's own logs, which used the pre-refactor `"<label> detected"`
speech format instead of the current `"<label> ahead."` format. Fixed
by killing it; the broader lesson (always verify only one server
instance is running before testing) is now something to check first,
not assume.

### The fan → airplane problem: found and fixed at the root

**Cause:** the dashboard's old "Object" mode (`mode == "object"`) was
a separate code path from `mode == "auto"` that always ran the
generic, COCO-pretrained `yolov8n.pt` via a module-level `get_detector()`
helper - completely bypassing `model_router` and its no-silent-fallback
guarantee. Worse, that mode's spoken-feedback logic had **no temporal
confirmation at all**: it only checked "was this label present in the
immediately-previous single frame," with no confidence-stability,
position-stability, or cooldown beyond that. COCO has an `airplane`
class and no `fan` class, and is a well-documented source of
spinning-blade-blur false positives against `airplane` - so a single
noisy frame in that mode could be spoken immediately, with nothing to
stop it.

**Fix:** `get_detector()` and the standalone `"object"` mode were
**deleted entirely**, not patched. Every live detection path -
`auto`, `navigation`, and `color` - now shares one pipeline: `model_router`
(real domain model or an honest "Detection model unavailable.", never
COCO) → `modules/tracking.py`'s temporal confirmation → the priority
engine → zone-aware phrasing. `mode="object"` is still accepted from
old clients for backward compatibility, but is silently treated as
`"auto"` rather than resurrecting the old behavior.

### Other real bugs found via live testing, and their fixes

| Finding | Fix |
|---|---|
| `config/config.yaml`'s `confidence:` section was **never actually wired in** - a `router.set_domain_confidence()` call was documented in a comment but never made, so every domain silently ran at the hardcoded 0.45 default regardless of config. The yaml also used `hazard` while the router's key is `road_hazards`. | Added `ModelRouter.set_domain_confidence()` and call it at startup with the (now key-corrected) config values. |
| The low-light warning re-spoke the identical sentence every ~30 seconds for as long as the scene stayed dark - confirmed live over a 20+ minute dark session, 20+ repeats of the same phrase. Directly violates "if nothing changes, do not repeat." | Speaks once when the scene *becomes* dark, then stays quiet until either light returns or `LOW_LIGHT_REMINDER_INTERVAL` (5 min) passes. |
| `speaker.stop()`'s `terminate()` call on an in-flight TTS subprocess (e.g. a CRITICAL announcement interrupting a lower-priority one) produced a nonzero exit code logged as `ERROR - TTS worker exited with code 1` - working-as-designed behavior that looked like a crash in the logs. | `Speaker` now tracks whether it deliberately terminated the process and logs that case as `INFO`, not `ERROR`. |
| **The exact same confirmed announcement repeated every ~6 seconds indefinitely** for a stationary object (e.g. "Warning. Door ahead. It's very close." 4x in 24 seconds) - a short per-key cooldown limits repeat *rate*, not repetition itself. Directly reproduced live, then fixed and re-verified live. | Auto/navigation/color announcements now track the last *situation* (mode+label+zone+tier) spoken. An unchanged situation falls back to a much longer, tier-scaled repeat interval (12-30s); a genuinely new one (different label, zone, or escalating tier) still announces immediately. |
| **A stationary object's announced zone flickered** between "on your left" / "ahead" / "on your right" every few seconds - each individual detection was itself correct, but zone was computed from only the single latest frame's bounding box, which jitters near zone boundaries even for a confirmed, unmoving object. Reproduced live, then fixed. | `TrackedObject` now exposes `stable_bbox` (a rolling mean of the last 5 frames), used for zone/distance decisions instead of the single latest frame's box. Reduces, but per honest live re-testing does not fully eliminate, boundary-adjacent flicker - see Remaining Work. |
| The trained `models/footpath/best.pt` existed but was never actually called anywhere in the live pipeline - a real trained model sitting unused. | Wired into `auto`/`navigation` modes: runs alongside the domain detector, computes per-zone walkability, and feeds a `blocks_path` signal into the priority engine (only trusted when footpath confirmed walkable ground exists *somewhere* in frame, so "nothing walkable anywhere" - the normal indoor case, where footpath doesn't apply - is never misread as "the path is blocked"). |
| `currency`/`OCR` were already on-demand only (not continuous per-frame), and currency already reported "not recognized" honestly on a miss - both already matched Sections 11/12's intent without changes. Phrasing was tightened to the exact requested wording ("Currency not recognized clearly.", "`<denomination>` rupees."). | Phrasing fix only; no architecture change was needed here. |

### UI redesign: User Mode vs. Developer Mode

The Home page (`/`) is now the literal thing Section 8 asked for -
`AI ASSISTANCE ACTIVE`, camera status, a `Listening`/`Processing`/
`Speaking` indicator, current situation, last spoken message, current
mode - backed by a new, deliberately tiny `/status` API (not the much
heavier `/detections` payload). Nothing on it needs to be clicked.
Every other page (Live Camera, OCR, Navigation, Voice Assistant,
Dashboard) is now explicitly labelled **"Developer Mode"** in its
heading and in the nav bar, with a one-line note pointing back to Home
for the real assistant status - existing functionality kept, nothing
rebuilt, just honestly relabelled as what it actually is: a debugging
view, not the primary interface.

---

## 📋 Final Status Report

Honest, per-domain status as actually built and measured against this
project's real datasets - no invented classes, no assumed accuracy.
`tools/dataset_report.json` (from `tools/analyze_datasets.py`) backs
every count below; training numbers come from each `training/train_*.py`
script's own `model.val()` output, never hand-typed.

> **Training status:** all three trainable domains (indoor, footpath,
> household) have been trained and validated with real measurements -
> no numbers below are estimated or assumed. Household ran 8 epochs
> instead of the script's default 25 (measured at ~942s/epoch on this
> CPU - 93 classes × 5,400 images is far heavier than indoor's 10
> classes × 1,012 images - so 25 epochs would have taken ~6.5 hours;
> 8 was a disclosed, honest scope reduction to fit a reasonable
> session, not a hidden shortcut). `training/train_household.py
> --epochs 25` (or higher) remains available for anyone with more CPU
> time or a GPU, and should improve on the numbers below - household's
> mAP50 was still climbing at a steady, unplateaued rate through all 8
> epochs (0.054 -> 0.225), unlike indoor/footpath which were closer to
> convergence by the time training stopped.

### Domain-by-domain

| Domain | Dataset | Status | Notes |
|---|---|---|---|
| **indoor** | 1,012 train / 230 val / 107 test images, 10 classes (door, cabinetDoor, refrigeratorDoor, window, chair, table, cabinet, couch, openedDoor, pole) | ✅ TRAINED | 20 epochs, `yolov8n.pt` base, CPU, imgsz 416. **Real measured validation results: mAP50 = 0.456, mAP50-95 = 0.303, precision = 0.672, recall = 0.430** (230 val images, 1,289 instances) - see per-class breakdown below. |
| **household** | 5,400 train / 600 val images (capped from 27,519 real photos), 93 classes (Shoe, Cup, Cooking pot, Hand, Toothbrush, Plate, Toy, Cutlery, Book, Power outlet, ...) | ✅ TRAINED (reduced scope) | 8 epochs (scoped down from 25 - see note above), `yolov8n.pt` base, CPU, imgsz 416. **Real measured validation results: mAP50 = 0.226, mAP50-95 = 0.151, precision = 0.523, recall = 0.233** (600 val images). Still improving steadily at epoch 8, not converged - expect meaningfully better numbers from `--epochs 25`+. 1 class ("Cleaning floor", 3 examples) dropped for having fewer than 20 examples. Per-class breakdown across 93 classes is too large for this table - see `runs/household/train/confusion_matrix.png` and `results.csv` for the full picture; rarest classes (near the 20-example minimum) are expected to be weakest. |
| **footpath** | 40 train / 9 val images, 1 class (footpath) | ✅ TRAINED | Stopped early at epoch 17/50 (`patience=15` - no improvement since epoch 2, exactly as configured). **Real measured validation results: mAP50 = 0.573, mAP50-95 = 0.375, recall = 0.900, precision = 0.003** (9 val images, 10 instances). The very low precision alongside high recall is a real, honest signature of overfitting on ~50 images - the model finds real footpath regions (high recall) but also throws a lot of spurious low-confidence boxes (YOLO's `val()` sweeps confidence thresholds for the mAP curve, which is why precision looks this extreme at the reported operating point). Treat this model as a proof-of-concept only - see Remaining Work. Two corrupt JPEGs in the val split were auto-repaired by Pillow/OpenCV during validation, a genuine data-quality issue worth fixing at the source. |
| **outdoor** | 861 real XML annotations (1,013 objects after stripping 2,899 watermark labels), 13 classes (car, pole, truck, flyover, hoarding, traffic symbols, pedestrian, traffic signal, bus, building, bike, auto rickshaw, caravan) | 🔴 NOT TRAINABLE | **Zero source images exist for any of the 861 annotated XML files anywhere in `dataset/`.** See `training/train_outdoor.py` for the full explanation and what's needed to unblock it. |
| **currency** | 6 reference images (one per denomination) | 🟢 WORKING (not a trained model) | ORB feature-matching (`modules/currency_detector.py`), not YOLO - reports "not recognized" honestly on a miss rather than guessing. |
| **road_hazards** | — | ⚪ NOT STARTED | Placeholder domain slot only - no dataset maps to it yet; see the Model Router section above. |

### indoor per-class validation results (real, from `model.val()`)

| Class | Val images | Val instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| cabinetDoor | 99 | 765 | 0.803 | 0.661 | 0.769 | 0.428 |
| refrigeratorDoor | 85 | 192 | 0.668 | 0.807 | 0.767 | 0.516 |
| chair | 24 | 49 | 0.701 | 0.526 | 0.584 | 0.299 |
| door | 63 | 97 | 0.551 | 0.423 | 0.471 | 0.283 |
| cabinet | 28 | 32 | 0.493 | 0.406 | 0.414 | 0.299 |
| window | 39 | 91 | 0.511 | 0.352 | 0.316 | 0.194 |
| table | 30 | 40 | 0.323 | 0.125 | 0.150 | 0.082 |
| couch | 1 | 1 | 0.671 | 1.000 | 0.995 | 0.895 |
| openedDoor | 13 | 13 | 1.000 | 0.000 | 0.073 | 0.031 |
| pole | 4 | 9 | 1.000 | 0.000 | 0.020 | 0.005 |

**Read this honestly, not optimistically:** `cabinetDoor`,
`refrigeratorDoor`, and `chair` are genuinely reliable (enough
validation instances, solid mAP50). `couch` (1 instance),
`openedDoor` (13 instances), and `pole` (9 instances) have too few
validation examples for their metrics to mean anything - `pole`'s
0.020 mAP50 reflects the model essentially never detecting it
correctly at this training budget, not a class that's somehow "5%
accurate" in any meaningful sense. More images for the underrepresented
classes (openedDoor, pole, couch, table) is the clear next step for
this domain, not more epochs on the current data.

### MISSING CLASS report

**MISSING CLASS: fan**
**RECOMMENDATION:** Add a custom dataset for fan (photos of ceiling
fans / table fans from varied angles, labeled `fan`, merged into the
indoor or household training set). Verified absent by searching every
class name in `tools/dataset_report.json` across all 5 datasets -
"fan" does not appear anywhere. This is the exact failure case the
temporal-confirmation tracker (`modules/tracking.py`) is designed to
contain: without a trained `fan` class, a spinning fan cannot be
correctly announced as "fan," but the tracker's consecutive-frame
requirement stops it from being mis-announced as something else
(e.g. "airplane") off a single noisy frame either - see
`tools/test_false_positives.py` Scenario 1.

### What's WORKING right now

- Auto-start / voice-first operation (camera, TTS, mic, AUTO ASSISTANCE
  with zero clicks) - verified live.
- Honest model-router status reporting (no domain silently substitutes
  COCO) - verified via `tools/test_false_positives.py`.
- Temporal confirmation, priority engine, announcement manager (cooldowns,
  CRITICAL interrupts, honest "not clearly recognized") - all 34 unit
  tests + 11 scripted scenarios passing.
- Dataset analysis/preparation tooling for all 5 real datasets.
- OCR, navigation (+ optional MiDaS depth), voice assistant, gesture
  control, scene summary, contextual memory, fall detection, currency
  (ORB), QR, SOS, real GPS, battery saver - all pre-existing features,
  unaffected by this phase's changes, still covered by their own unit
  tests.
- No face recognition anywhere - verified by full-codebase search.

### Remaining Work

1. **Train household further** - `models/household/best.pt` exists and
   works, but only 8 epochs were run (see the scope-reduction note
   above); mAP50 was still climbing steadily, not plateaued, so
   `training/train_household.py --epochs 25` (or more) on a faster
   machine/GPU should meaningfully improve on 0.226 mAP50.
2. **Collect more footpath images.** ~50 source images produced a
   model that overfits (0.9 recall but 0.003 precision at `val()`'s
   reported operating point) - more images from varied locations/
   lighting is the fix, not more epochs on the same 50.
3. **Fix the 2 corrupt JPEGs** found in `dataset/Footpath/footpath_images/`
   during validation (auto-repaired at load time by Pillow/OpenCV, but
   worth fixing at the source).
4. **Improve indoor's underrepresented classes** (`openedDoor`: 13
   instances, `pole`: 9, `couch`: 1) - their mAP50/mAP50-95 numbers
   reflect too little validation data to be meaningful, not a model
   that's "somehow bad at doors." More images for exactly these classes
   is the highest-value next step for indoor.
5. **Add a `fan` class** to indoor or household training data - see
   MISSING CLASS report above.
6. **Unblock outdoor training** - either capture/source real photos to
   pair with the existing 861 XML annotations, or obtain the
   Day/Foggy/Rainy `labels/` OBB set's class-id-to-name mapping from
   whoever produced it.
7. **`road_hazards` domain** - currently just a placeholder slot; no
   dataset maps to it yet (closest candidate, pothole/speed-breaker
   classes, isn't present in any of the 5 real datasets).
8. **Currency**: current ORB matching is lighting/angle-sensitive by
   nature; a trained classifier (once enough real photos per
   denomination exist, not just 1 reference image each) would be more
   robust - see `dataset/currency/README.md`.
9. **household's rarest classes** (several sit right at the 20-example
   minimum) will likely have weak per-class accuracy - check
   `runs/household/train/confusion_matrix.png` once trained before
   trusting them individually.
10. **Residual zone-boundary flicker** - `stable_bbox`'s 5-frame rolling
    average (see Audit & Redesign Findings) measurably reduces but does
    not fully eliminate an object's announced zone changing when its
    true position sits right at a left/center or center/right boundary.
    A hysteresis-based zone state machine (require N consecutive frames
    in the *new* zone before treating it as a real transition, not just
    a smoothed average crossing the line) would fix this more
    completely, at the cost of slightly slower reporting of genuine
    movement - a real safety/annoyance tradeoff worth deciding
    deliberately rather than defaulting silently.
11. **Class confusion between visually similar trained classes** (e.g.
    indoor's `door` vs `cabinetDoor`/`refrigeratorDoor` for the same
    physical object) was observed during live audit testing - each
    individual detection clears its confidence threshold, so the
    tracker correctly confirms both as separate tracks, but the result
    can alternate which label gets spoken. This is a genuine model
    discriminability limit at the current training budget (consistent
    with indoor's measured 0.456 mAP50), not a pipeline bug - more
    training data distinguishing these classes is the real fix.

---

## ⚠️ Disclaimer

This is an academic prototype built for a final-year engineering
project. Navigation guidance is a **heuristic, monocular** estimate - even
with MiDaS depth estimation enabled, it only ever reports *relative*
closeness ("very close" / "farther away"), never a calibrated distance
in metres, because a single uncalibrated camera cannot honestly measure
one. **Fall detection is an experimental heuristic** based only on 2D
bounding-box shape change, not a validated medical device - it prompts a
spoken check-in, and should never be relied on as an automatic emergency
alert. The Emergency SOS feature is a **simulated** integration (no real
contact is notified); GPS location is genuinely real when reported by
the browser, but has no fallback if the visitor denies location
permission. None of this project should be relied upon as a certified
mobility, safety, or medical device.

---

## 📄 License

Provided as-is for educational purposes.
