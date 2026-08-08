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
        window.showToast(
          `🆘 SOS sent! Contact: ${data.contact_name} (${data.contact_phone})`,
          6000
        );
      } catch (err) {
        window.showToast("Failed to trigger SOS. Is the server running?");
      } finally {
        setTimeout(() => (sosBtn.disabled = false), 3000);
      }
    });
  }
})();
