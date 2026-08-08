// Live Camera page: start/stop the webcam stream, switch detection mode,
// poll for detections/FPS/QR/currency results, and toggle battery saver.
(function () {
  const videoFeed = document.getElementById("videoFeed");
  const placeholder = document.getElementById("cameraPlaceholder");
  const toggleBtn = document.getElementById("toggleCameraBtn");
  const modeButtons = document.querySelectorAll(".mode-btn");
  const detectionList = document.getElementById("detectionList");
  const fpsBadge = document.getElementById("fpsBadge");
  const captureCurrencyBtn = document.getElementById("captureCurrencyBtn");
  const currencyResult = document.getElementById("currencyResult");
  const qrResult = document.getElementById("qrResult");
  const batteryToggle = document.getElementById("batterySaverToggle");
  const describeSceneBtn = document.getElementById("describeSceneBtn");
  const sceneSummaryResult = document.getElementById("sceneSummaryResult");
  const gestureToggle = document.getElementById("gestureToggle");
  const gestureResult = document.getElementById("gestureResult");

  let cameraOn = false;
  let pollTimer = null;
  let lastGesture = null;

  // The camera/mode are server-side state that outlives any single
  // page. Sync with the server on load so arriving here after starting
  // the camera on another page shows the live feed immediately instead
  // of a misleading "Camera is off" placeholder.
  async function syncCameraState() {
    try {
      const res = await fetch("/detections");
      const data = await res.json();
      if (data.camera_active) {
        cameraOn = true;
        videoFeed.src = "/video_feed?_=" + Date.now();
        placeholder.classList.add("hidden");
        toggleBtn.textContent = "Stop Camera";
        startPolling();
      }
      modeButtons.forEach((b) => b.classList.toggle("active", b.dataset.mode === data.mode));
      gestureToggle.checked = !!data.gesture_active;
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
      startPolling();
      window.showToast("Camera started.");
    } else {
      window.showToast("Could not open camera. Check webcam connection.");
    }
  }

  async function stopCamera() {
    await fetch("/camera/stop", { method: "POST" });
    cameraOn = false;
    videoFeed.src = "";
    placeholder.classList.remove("hidden");
    toggleBtn.textContent = "Start Camera";
    stopPolling();
  }

  toggleBtn.addEventListener("click", () => (cameraOn ? stopCamera() : startCamera()));

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      await fetch("/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: btn.dataset.mode }),
      });
    });
  });

  captureCurrencyBtn.addEventListener("click", async () => {
    currencyResult.classList.remove("hidden");
    currencyResult.textContent = "Scanning...";
    const res = await fetch("/capture_currency", { method: "POST" });
    const data = await res.json();
    if (data.success && data.result) {
      currencyResult.textContent = `💵 Detected: ₹${data.result.denomination}`;
    } else {
      currencyResult.textContent = data.error || "Currency not recognized.";
    }
  });

  batteryToggle.addEventListener("change", async () => {
    await fetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ battery_saver: batteryToggle.checked }),
    });
    window.showToast(batteryToggle.checked ? "Battery saver on." : "Battery saver off.");
  });

  describeSceneBtn.addEventListener("click", async () => {
    if (!cameraOn) {
      window.showToast("Start the camera first.");
      return;
    }
    sceneSummaryResult.classList.remove("hidden");
    sceneSummaryResult.textContent = "Looking...";
    const res = await fetch("/describe_scene", { method: "POST" });
    const data = await res.json();
    sceneSummaryResult.textContent = data.success ? `🗣️ ${data.summary}` : data.error || "Could not describe the scene.";
  });

  gestureToggle.addEventListener("change", async () => {
    await fetch("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gesture_active: gestureToggle.checked }),
    });
    window.showToast(
      gestureToggle.checked
        ? "Gesture control on - try a fist, open palm, thumbs up, or peace sign."
        : "Gesture control off."
    );
  });

  function renderDetections(snapshot) {
    fpsBadge.textContent = `${snapshot.fps.toFixed(1)} FPS`;

    if (!snapshot.detections.length) {
      detectionList.innerHTML = '<li class="empty-state">No detections yet.</li>';
    } else {
      detectionList.innerHTML = snapshot.detections
        .map(
          (d) =>
            `<li><span class="det-label">${d.label}${d.color ? " (" + d.color + ")" : ""}</span>` +
            `<span class="det-conf">${Math.round(d.confidence * 100)}%</span></li>`
        )
        .join("");
    }

    if (snapshot.last_qr && snapshot.last_qr.length) {
      qrResult.classList.remove("hidden");
      qrResult.innerHTML = snapshot.last_qr.map((q) => `🔳 ${q.data}`).join("<br>");
    }

    if (snapshot.last_gesture && snapshot.last_gesture !== lastGesture) {
      lastGesture = snapshot.last_gesture;
      gestureResult.classList.remove("hidden");
      gestureResult.textContent = `✋ Gesture recognized: ${snapshot.last_gesture.replace("_", " ")}`;
    }
  }

  function startPolling() {
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/detections");
        const data = await res.json();
        renderDetections(data);
      } catch (err) {
        /* ignore transient network errors while polling */
      }
    }, 700);
  }

  function stopPolling() {
    clearInterval(pollTimer);
  }
})();
