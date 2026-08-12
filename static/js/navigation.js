// Navigation page: start/stop camera, toggle navigation mode, and poll
// for the current spoken instruction + left/center/right obstacle map.
(function () {
  const videoFeed = document.getElementById("videoFeed");
  const placeholder = document.getElementById("cameraPlaceholder");
  const toggleBtn = document.getElementById("toggleCameraBtn");
  const navBtn = document.getElementById("toggleNavBtn");
  const navInstruction = document.getElementById("navInstruction");
  const zoneLeft = document.querySelector("#zoneLeft ul");
  const zoneCenter = document.querySelector("#zoneCenter ul");
  const zoneRight = document.querySelector("#zoneRight ul");

  let cameraOn = false;
  let navOn = false;
  let pollTimer = null;

  // The camera/mode are server-side state that outlives any single
  // page. Sync with the server on load so arriving here after starting
  // the camera on another page shows the live feed (and correct
  // Start/Stop Navigation label) immediately.
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
      navOn = data.mode === "navigation";
      navBtn.textContent = navOn ? "⏹ Stop Navigation" : "🧭 Start Navigation";
      navBtn.classList.toggle("active", navOn);
    } catch (err) {
      /* server not reachable yet - leave the defaults */
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
    stopPolling();
  }

  toggleBtn.addEventListener("click", () => (cameraOn ? stopCamera() : startCamera()));

  navBtn.addEventListener("click", async () => {
    navOn = !navOn;
    await fetch("/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: navOn ? "navigation" : "auto" }),
    });
    navBtn.textContent = navOn ? "⏹ Stop Navigation" : "🧭 Start Navigation";
    navBtn.classList.toggle("active", navOn);
  });

  function renderZones(zoneMap) {
    zoneLeft.innerHTML =
      (zoneMap.left || []).map((l) => `<li>${l}</li>`).join("") ||
      "<li class='empty-state'>Clear</li>";
    zoneCenter.innerHTML =
      (zoneMap.center || []).map((l) => `<li>${l}</li>`).join("") ||
      "<li class='empty-state'>Clear</li>";
    zoneRight.innerHTML =
      (zoneMap.right || []).map((l) => `<li>${l}</li>`).join("") ||
      "<li class='empty-state'>Clear</li>";
  }

  function startPolling() {
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/detections");
        const data = await res.json();
        navInstruction.textContent = data.navigation_instruction;
        renderZones(data.zone_map || {});
      } catch (err) {
        /* ignore transient network errors while polling */
      }
    }, 700);
  }

  function stopPolling() {
    clearInterval(pollTimer);
  }
})();
