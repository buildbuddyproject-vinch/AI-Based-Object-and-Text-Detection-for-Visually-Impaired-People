// Shared utilities loaded on every page: dark mode toggle, toast
// notifications, and the Emergency SOS button.
(function () {
  const root = document.documentElement;
  const toggleBtn = document.getElementById("darkModeToggle");
  const stored = localStorage.getItem("theme");

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (toggleBtn) toggleBtn.textContent = theme === "dark" ? "☀️" : "🌙";
  }

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(stored || (prefersDark ? "dark" : "light"));

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      localStorage.setItem("theme", next);
    });
  }

  window.showToast = function (message, duration) {
    duration = duration || 4000;
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("visible");
    clearTimeout(window._toastTimer);
    window._toastTimer = setTimeout(() => toast.classList.remove("visible"), duration);
  };

  const sosBtn = document.getElementById("sosBtn");
  if (sosBtn) {
    sosBtn.addEventListener("click", async () => {
      sosBtn.disabled = true;
      try {
        const res = await fetch("/sos", { method: "POST" });
        const data = await res.json();
        const locPart = data.location
          ? ` Location: ${data.location.latitude.toFixed(4)}, ${data.location.longitude.toFixed(4)}.`
          : " Location unavailable.";
        window.showToast(
          `🆘 SOS sent! Contact: ${data.contact_name} (${data.contact_phone}).${locPart}`,
          7000
        );
      } catch (err) {
        window.showToast("Failed to trigger SOS. Is the server running?");
      } finally {
        setTimeout(() => (sosBtn.disabled = false), 3000);
      }
    });
  }

  // Real GPS via the browser's own Geolocation API - resolved from the
  // visitor's device (Wi-Fi/cell/GPS chip), not anything the Python
  // server has access to on its own. Reported on every page load and
  // refreshed periodically so /sos can include an up-to-date location.
  // Requires a secure context (HTTPS, or localhost for local dev) and
  // the user granting the browser's location permission prompt.
  function reportLocation() {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        fetch("/location", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: pos.coords.latitude,
            longitude: pos.coords.longitude,
            accuracy: pos.coords.accuracy,
          }),
        }).catch(() => {});
      },
      () => {
        /* permission denied / unavailable - SOS/GPS silently fall back
           to the mock coordinates server-side, no user-facing error */
      },
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 60000 }
    );
  }
  reportLocation();
  setInterval(reportLocation, 60000);
})();
