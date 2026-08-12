// Dashboard page: polls /api/metrics and renders live system stats,
// feature status, and recent event activity.
(function () {
  const statFps = document.getElementById("statFps");
  const statMode = document.getElementById("statMode");
  const statUptime = document.getElementById("statUptime");
  const statCpu = document.getElementById("statCpu");
  const statMemory = document.getElementById("statMemory");
  const statusCamera = document.getElementById("statusCamera");
  const statusVoice = document.getElementById("statusVoice");
  const statusGesture = document.getElementById("statusGesture");
  const statusDepth = document.getElementById("statusDepth");
  const eventCounters = document.getElementById("eventCounters");
  const recentEvents = document.getElementById("recentEvents");
  const modelStatusList = document.getElementById("modelStatusList");
  const activeDomain = document.getElementById("activeDomain");
  const lastAnnouncement = document.getElementById("lastAnnouncement");

  function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  }

  function badge(el, active, onText, offText) {
    el.textContent = active ? onText || "ON" : offText || "OFF";
    el.classList.toggle("status-on", !!active);
    el.classList.toggle("status-off", !active);
  }

  function render(data) {
    statFps.textContent = data.fps.toFixed(1);
    statMode.textContent = data.mode;
    statUptime.textContent = formatUptime(data.uptime_seconds);
    statCpu.textContent = data.cpu_percent != null ? `${data.cpu_percent}%` : "n/a";
    statMemory.textContent = data.memory_mb != null ? `${data.memory_mb} MB` : "n/a";

    badge(statusCamera, data.camera_active);
    badge(statusVoice, data.voice_active);
    badge(statusGesture, data.gesture_active);
    badge(statusDepth, data.depth_enabled);

    const modelStatus = Object.entries(data.model_status || {});
    modelStatusList.innerHTML = modelStatus.length
      ? modelStatus
          .map(([domain, status]) => {
            const on = status === "AVAILABLE";
            return `<li>${domain} <span class="status-badge ${on ? "status-on" : "status-off"}">${status}</span></li>`;
          })
          .join("")
      : '<li class="empty-state">No model status reported.</li>';
    activeDomain.textContent = data.active_domain || "none (no domain model available)";
    lastAnnouncement.textContent = data.last_auto_announcement || "(nothing spoken yet)";

    const counters = Object.entries(data.event_counters || {});
    eventCounters.innerHTML = counters.length
      ? counters.map(([type, count]) => `<li><span>${type}</span><span>${count}</span></li>`).join("")
      : '<li class="empty-state">No events yet.</li>';

    const events = data.recent_events || [];
    recentEvents.innerHTML = events.length
      ? events
          .slice()
          .reverse()
          .map((e) => {
            const details = Object.entries(e)
              .filter(([k]) => k !== "time" && k !== "type")
              .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
              .join(" ");
            return `<li><span class="log-time">${e.time}</span> <strong>${e.type}</strong> ${details}</li>`;
          })
          .join("")
      : '<li class="empty-state">No activity yet.</li>';
  }

  async function poll() {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      render(data);
    } catch (err) {
      /* ignore transient network errors while polling */
    }
  }

  poll();
  setInterval(poll, 2000);
})();
