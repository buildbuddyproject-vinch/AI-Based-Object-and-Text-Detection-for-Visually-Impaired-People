// User Mode home page (Section 8): read-only status view, no controls
// to click. Polls the small /status endpoint (not /detections, which
// backs the much heavier Developer Mode dashboard) and updates the
// page text - nothing here starts/stops anything, since the assistant
// is already running by the time this page can even load.
(function () {
  const heading = document.getElementById("assistanceHeading");
  const cameraText = document.getElementById("statusCameraText");
  const activityText = document.getElementById("statusActivityText");
  const modeText = document.getElementById("statusModeText");
  const situation = document.getElementById("currentSituation");
  const lastSpoken = document.getElementById("lastSpoken");

  async function poll() {
    try {
      const res = await fetch("/status");
      const data = await res.json();
      heading.textContent = data.assistance_active ? "AI ASSISTANCE ACTIVE" : "Starting…";
      cameraText.textContent = data.camera_active ? "On" : "Off";
      activityText.textContent = data.activity;
      modeText.textContent = data.mode;
      situation.textContent = data.current_situation;
      lastSpoken.textContent = data.last_spoken || "-";
    } catch (err) {
      heading.textContent = "Connecting…";
    }
  }

  poll();
  setInterval(poll, 1500);
})();
