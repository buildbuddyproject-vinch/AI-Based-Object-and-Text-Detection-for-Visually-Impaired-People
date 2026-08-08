// OCR page: start/stop camera, capture a frame and extract text, then
// read it aloud (or stop speaking) via the offline TTS backend.
(function () {
  const videoFeed = document.getElementById("videoFeed");
  const placeholder = document.getElementById("cameraPlaceholder");
  const toggleBtn = document.getElementById("toggleCameraBtn");
  const captureBtn = document.getElementById("captureOcrBtn");
  const ocrText = document.getElementById("ocrText");
  const readAloudBtn = document.getElementById("readAloudBtn");
  const stopBtn = document.getElementById("stopSpeakingBtn");

  let cameraOn = false;

  // The camera is server-side state that outlives any single page. If
  // it was started from another page (Live Camera, Navigation...) and
  // the user then navigates here, this page's JS starts fresh and would
  // otherwise think the camera is off - blocking "Capture & Read Text"
  // even though it's actually running. Sync with the server on load.
  async function syncCameraState() {
    try {
      const res = await fetch("/detections");
      const data = await res.json();
      if (data.camera_active) {
        cameraOn = true;
        videoFeed.src = "/video_feed?_=" + Date.now();
        placeholder.classList.add("hidden");
        toggleBtn.textContent = "Stop Camera";
      }
    } catch (err) {
      /* server not reachable yet - leave the "camera is off" default */
    }
  }
  syncCameraState();

  async function startCamera() {
    const res = await fetch("/camera/start", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      cameraOn = true;
      videoFeed.src = "/video_feed?_=" + Date.now();
      placeholder.classList.add("hidden");
      toggleBtn.textContent = "Stop Camera";
    } else {
      window.showToast("Could not open camera.");
    }
  }

  async function stopCamera() {
    await fetch("/camera/stop", { method: "POST" });
    cameraOn = false;
    videoFeed.src = "";
    placeholder.classList.remove("hidden");
    toggleBtn.textContent = "Start Camera";
  }

  toggleBtn.addEventListener("click", () => (cameraOn ? stopCamera() : startCamera()));

  captureBtn.addEventListener("click", async () => {
    if (!cameraOn) {
      window.showToast("Start the camera first.");
      return;
    }
    ocrText.value = "Reading text...";
    const res = await fetch("/capture_ocr", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      ocrText.value = data.text || "No text detected.";
    } else {
      ocrText.value = "";
      window.showToast(data.error || "OCR failed.");
    }
  });

  readAloudBtn.addEventListener("click", async () => {
    if (!ocrText.value.trim()) return;
    await fetch("/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: ocrText.value }),
    });
  });

  stopBtn.addEventListener("click", async () => {
    await fetch("/stop_speaking", { method: "POST" });
  });
})();
