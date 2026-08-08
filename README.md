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
| 📖 OCR Mode | Capture printed text with EasyOCR and hear it read aloud |
| 🧭 Navigation Mode | Left / Center / Right obstacle guidance, spoken instructions |
| 🎙️ Voice Assistant | Hands-free control via microphone commands |
| 🔊 Audio Feedback | Offline pyttsx3 speech, de-duplicated so it doesn't repeat itself |
| ⚡ Performance | Threaded camera capture, frame-skip & battery saver mode |
| 🎨 Color Detection (bonus) | Names the dominant color of a detected object |
| 💵 Currency Detection (bonus) | Heuristic Indian currency note recognition |
| 🔳 QR Code Reader (bonus) | Detects and reads QR codes aloud |
| 🆘 Emergency SOS (bonus) | One-tap simulated emergency alert |
| 📍 GPS (bonus, placeholder) | Mock coordinates endpoint, ready for real GPS hardware |
| 🔋 Battery Saver (bonus) | Reduces frame rate / resolution to save CPU/battery |

---

## 🧰 Tech Stack

**Backend:** Python, Flask
**AI/CV:** YOLOv8 (Ultralytics), EasyOCR, OpenCV, pyttsx3, SpeechRecognition
**Frontend:** HTML, CSS, JavaScript (responsive, accessibility-first, dark mode)

---

## 📁 Project Structure

```
AI-Based Object and Text Detection for Visually Impaired People/
│
├── app.py                     # Flask app entry point (routes + detection loop)
├── config.py                  # Central configuration
├── download_weights.py        # Convenience script to fetch yolov8n.pt
├── requirements.txt
├── README.md
├── .gitignore
│
├── modules/
│   ├── object_detector.py     # YOLOv8 wrapper
│   ├── ocr_reader.py          # EasyOCR wrapper
│   ├── speaker.py             # Threaded, de-duplicated pyttsx3 TTS
│   ├── navigation.py          # Left/Center/Right obstacle guidance
│   ├── voice_commands.py      # Microphone command listener
│   ├── color_detector.py      # (bonus) dominant color naming
│   ├── currency_detector.py   # (bonus) heuristic currency recognition
│   └── qr_reader.py           # (bonus) QR code detection
│
├── utils/
│   ├── camera_stream.py       # Threaded webcam reader
│   ├── frame_utils.py         # Resize / JPEG encode / FPS tracker
│   ├── logger.py              # Console logger
│   └── state.py                # Thread-safe shared app state
│
├── templates/                 # Jinja2 HTML pages (Home, Live, OCR, Navigation, Voice, About)
├── static/
│   ├── css/style.css           # Blue accessibility theme + dark mode
│   └── js/                     # main.js, camera.js, ocr.js, navigation.js, voice.js
│
├── weights/                    # Place yolov8n.pt here
├── dataset/
│   └── currency/                # Optional reference images for currency detection
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

### 5. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## ▶️ How to Use

1. **Home** — click "Start Camera" or "Voice Assistant" to jump right in.
2. **Live Camera** — click **Start Camera**, choose a mode (Object /
   Color / QR Code), and detected items are spoken aloud as they appear.
   Use **💵 Detect Currency** to scan a note (needs reference images —
   see `dataset/currency/README.md`).
3. **OCR** — start the camera, aim at printed text, click **Capture &
   Read Text**, then **Read Aloud**.
4. **Navigation** — start the camera, click **Start Navigation** to get
   spoken Left/Center/Right obstacle guidance.
5. **Voice Assistant** — click the mic button and say:
   - `"read text"` — capture + read text aloud
   - `"start navigation"` — switch to navigation mode
   - `"detect objects"` — switch to object detection mode
   - `"stop speaking"` — interrupt current speech
   - `"exit"` — stop the voice assistant
6. **SOS** — the red **SOS** button in the header is available on every
   page and triggers a simulated emergency alert (spoken + logged).
7. **Dark Mode** — toggle via the 🌙/☀️ icon in the header; your choice
   is remembered.
8. **Battery Saver** — toggle on the Live Camera page to reduce frame
   rate and resolution for lower CPU/battery usage.

---

## 🧪 How to Test

Automated unit tests cover the pure-logic modules (navigation heuristics,
color naming, QR/currency wrappers) without needing a camera, GPU, or
microphone:

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
- [ ] SOS button shows a toast with the configured emergency contact.
- [ ] Toggling Battery Saver reduces CPU usage (check Task Manager).

---

## 📸 Expected Output

- The **Live Camera** page shows your webcam feed with colored bounding
  boxes around detected objects (e.g. `person 92%`, `chair 87%`), and
  each newly-seen object is announced once, e.g. *"person detected"*.
- The **OCR** page displays the extracted text in a textbox after
  capture, and reads it aloud on request.
- The **Navigation** page shows a short spoken sentence like *"Chair on
  your left."* or *"Person ahead. Move slightly right."*, plus a
  Left/Center/Right panel listing everything currently in each zone.
- The **Voice Assistant** page logs every recognized phrase and which
  command (if any) it matched.

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
  rendering speech, and raises no exception. `modules/speaker.py` avoids
  this entirely by building a **brand-new `pyttsx3.init()` engine for
  every utterance** and letting it be garbage-collected right after,
  instead of reusing one engine across the app's lifetime. If you're
  extending this module, keep that pattern - reusing an engine across
  multiple calls is the one thing to avoid.
- A separate, unrelated Windows quirk can also silently mute output: the
  OS persists a per-application volume in the Volume Mixer keyed by
  executable name, and `python.exe`/`python3.11.exe` can end up pinned
  to a near-zero level left over from an unrelated earlier session,
  sometimes not fixable by dragging the mixer slider or clicking
  "Reset". `_force_full_volume_windows()` in `modules/speaker.py` fixes
  this once at Speaker startup via `pycaw` (see `requirements.txt`) -
  deliberately only once, not per-utterance, since calling it repeatedly
  around active playback was found to itself destabilize `pyttsx3` and
  reproduce the exact same "instant silent finish" symptom.
- If you still hear nothing: confirm your system's default *output
  device* (Settings → System → Sound → Output) is actually your
  speakers/headphones and not an unplugged HDMI/virtual device, and
  check for audio-enhancement software (e.g. Nahimic) that mutes
  unrecognized apps in its own mixer.

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

- Object detection (YOLOv8), OCR (EasyOCR) and text-to-speech (pyttsx3)
  all run **fully offline** once their model weights are downloaded.
- Only the default voice-command speech-to-text step
  (`SpeechRecognition`'s Google Web Speech API) requires internet; swap
  it for Vosk (see Troubleshooting) to go fully offline end-to-end.

---

## ☁️ Deployment

### ⚠️ Read this before deploying to Render, Vercel, or any cloud host

This app talks to hardware **on the machine running `app.py`**, not the
visitor's browser:

- `cv2.VideoCapture(0)` in `utils/camera_stream.py` opens whatever
  camera is physically attached to that machine.
- `pyttsx3` (offline TTS) and the voice assistant's microphone
  (`SpeechRecognition`) need local audio output/input hardware.

Run locally, that machine is your own laptop/PC, so "the camera" and
"the speakers" correctly mean *your* webcam and speakers. Deployed to a
cloud server, "that machine" is Render's container - which has no
webcam, no microphone, and no speakers. The pages will load and the
UI/API will respond, but **Live Camera, OCR, Navigation, Voice
Assistant, and all audio narration will not work for anyone visiting
the deployed URL** - `camera.start()` fails to open a nonexistent
device, the mic fails to initialize, and TTS has no audio driver to
speak through. Both failures are handled gracefully (a toast/log
message, not a crash) but the features are simply unavailable.

This is fine for demoing the UI, the object-detection/OCR/navigation
*code paths* against your own machine, or the non-hardware endpoints
(Home, About, SOS, GPS placeholder, Settings) - it is not a way to give
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
   deliberately - it drops `PyAudio`, which needs system build headers
   Render's Python buildpack doesn't have and would otherwise fail the
   *entire* install.
4. **Memory**: `torch` + `ultralytics` + `easyocr` loaded together can
   approach or exceed Render's free-tier 512 MB RAM limit, especially
   once a request actually triggers OCR or detection - if the app gets
   OOM-killed under load, that's why. A paid tier with more RAM avoids
   this.
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
project. Navigation guidance is a **heuristic, monocular** estimate (no
true depth sensor) and the Emergency SOS / GPS features are
**simulated/placeholder** integrations. This project should not be relied
upon as a certified mobility or safety aid.

---

## 📄 License

Provided as-is for educational purposes.
