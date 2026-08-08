# 👁️ AI-Based Object and Text Detection for Visually Impaired People

A final-year engineering project that helps visually impaired users
understand their surroundings using real-time object detection, printed
text recognition, spoken navigation guidance, and a hands-free voice
assistant — all running **offline** on a regular laptop with a webcam.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏠 Home Page | Accessible landing page with large buttons and clear navigation |
| 📷 Live Camera | Real-time YOLOv8 object detection with bounding boxes + confidence |
| 📖 OCR Mode | Capture printed text with EasyOCR, reading-order-sorted, hear it read aloud |
| ⚠️ Important-Text Priority | Warnings/exits/hazards in OCR text are called out first, not buried |
| 🧭 Navigation Mode | Left / Center / Right obstacle guidance, spoken instructions |
| 📏 Monocular Depth Estimation | MiDaS-based relative depth refines "very close" / "farther away" |
| 🗣️ Scene Summary | On-demand natural-language description of everything currently detected |
| 🧠 Contextual Memory | "Where is my bag?" — answers from a short-term detection history |
| 🎙️ Voice Assistant | Hands-free control via microphone commands |
| ✋ Hand Gesture Control | Static gestures (fist, open palm, peace, thumbs up) trigger the same actions as voice |
| 🤕 Fall Detection (experimental) | Heuristic bounding-box-collapse check with a spoken safety prompt |
| 🔊 Audio Feedback | Offline pyttsx3 speech, de-duplicated so it doesn't repeat itself |
| ⚡ Performance | Threaded camera capture, frame-skip, battery saver mode, optional ONNX inference |
| 📊 System Dashboard | Live FPS/CPU/memory, feature status, and an event log |
| 🎨 Color Detection (bonus) | Names the dominant color of a detected object |
| 💵 Currency Detection (bonus) | Heuristic Indian currency note recognition |
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
├── app.py                     # Flask app entry point (routes + detection loop)
├── config.py                  # Central configuration
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
│   ├── currency_detector.py   # (bonus) heuristic currency recognition
│   └── qr_reader.py           # (bonus) QR code detection
│
├── utils/
│   ├── camera_stream.py       # Threaded webcam reader
│   ├── frame_utils.py         # Resize / JPEG encode / FPS tracker / low-light check
│   ├── logger.py              # Console logger
│   ├── event_log.py           # Structured event log powering /dashboard
│   └── state.py                # Thread-safe shared app state
│
├── templates/                 # Jinja2 pages (Home, Live, OCR, Navigation, Voice, Dashboard, About)
├── static/
│   ├── css/style.css           # Blue accessibility theme + dark mode
│   └── js/                     # main.js, camera.js, ocr.js, navigation.js, voice.js, dashboard.js
│
├── weights/                    # Place yolov8n.pt here (+ auto-downloaded .onnx/.task files)
├── dataset/
│   └── currency/                # Optional reference images for currency detection
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

1. **Home** — click "Start Camera" or "Voice Assistant" to jump right in.
2. **Live Camera** — click **Start Camera**, choose a mode (Object /
   Color / QR Code), and detected items are spoken aloud as they appear.
   - **💵 Detect Currency** to scan a note (needs reference images —
     see `dataset/currency/README.md`).
   - **🗣️ Describe Scene** for an on-demand natural-language summary of
     everything currently detected.
   - **✋ Gesture Control** toggle enables hand-gesture commands (see below).
3. **OCR** — start the camera, aim at printed text, click **Capture &
   Read Text**, then **Read Aloud**. Text is read in proper top-to-bottom
   reading order, and a toast warns you if it contains a safety-relevant
   keyword (exit, danger, wet floor, etc.) before you even read it.
4. **Navigation** — start the camera, click **Start Navigation** to get
   spoken Left/Center/Right obstacle guidance. If MiDaS depth estimation
   loaded successfully (needs internet the first time), guidance also
   distinguishes "very close" from "farther away" using real per-pixel
   depth, not just bounding-box size.
5. **Voice Assistant** — click the mic button and say:
   - `"read text"` — capture + read text aloud (auto-starts the camera)
   - `"start navigation"` — switch to navigation mode (auto-starts the camera)
   - `"detect objects"` — switch to object detection mode
   - `"stop speaking"` — interrupt current speech
   - `"repeat"` — say the last announcement again
   - `"describe the scene"` — summarize what's currently detected
   - `"where is my bag"` (or any recently-seen object) — answered from
     a short-term detection memory
   - `"exit"` — stop the voice assistant
6. **Hand Gestures** — enable "Gesture Control" on the Live Camera page,
   then hold a hand up to the camera: **fist** = start/stop detection,
   **open palm** = stop speaking, **peace sign** = read text, **one
   finger** = describe the scene, **thumbs up** = repeat. Same actions
   as the voice commands above — use whichever input suits the moment.
7. **Dashboard** — live FPS, CPU/memory usage, which features are
   currently active, and a running event log (mode changes, SOS
   triggers, detections, gestures, etc.) — useful for a demo/report.
8. **SOS** — the red **SOS** button in the header is available on every
   page and triggers a simulated emergency alert (spoken + logged),
   including your real location if the browser reported one.
9. **Dark Mode** — toggle via the 🌙/☀️ icon in the header; your choice
   is remembered.
10. **Battery Saver** — toggle on the Live Camera page to reduce frame
    rate and resolution for lower CPU/battery usage.

---

## 🧪 How to Test

Automated unit tests cover the pure-logic modules (navigation heuristics
including depth-aware distance, color naming, QR/currency wrappers,
fall detection, contextual memory, scene summary, OCR reading order and
keyword priority) without needing a camera, GPU, or microphone:

```bash
python -m unittest discover -s tests -v
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
