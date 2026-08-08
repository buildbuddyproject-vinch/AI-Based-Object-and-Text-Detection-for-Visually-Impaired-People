// Voice Assistant page: start/stop the microphone listener and poll for
// the recognized-speech / command activity log.
(function () {
  const micBtn = document.getElementById("micBtn");
  const micLabel = document.getElementById("micLabel");
  const voiceLog = document.getElementById("voiceLog");

  let listening = false;
  let pollTimer = null;

  // The listener is server-side state that outlives any single page
  // (including surviving the redirect a voice command can trigger).
  // Sync with the server on load so coming back to this page doesn't
  // show a stale "Start Listening" button while it's already running -
  // clicking that stale button would otherwise stop a listener the
  // user thinks they're starting.
  async function syncListenerState() {
    try {
      const res = await fetch("/voice/status");
      const data = await res.json();
      listening = !!data.active;
      micBtn.classList.toggle("active", listening);
      micBtn.setAttribute("aria-pressed", String(listening));
      micLabel.textContent = listening ? "Stop Listening" : "Start Listening";
      renderLog(data.log || []);
      if (listening) startPolling();
    } catch (err) {
      /* server not reachable yet - leave the defaults */
    }
  }
  syncListenerState();

  micBtn.addEventListener("click", async () => {
    listening = !listening;
    const endpoint = listening ? "/voice/start" : "/voice/stop";
    await fetch(endpoint, { method: "POST" });
    micBtn.classList.toggle("active", listening);
    micBtn.setAttribute("aria-pressed", String(listening));
    micLabel.textContent = listening ? "Stop Listening" : "Start Listening";
    if (listening) startPolling();
    else stopPolling();
  });

  function renderLog(entries) {
    if (!entries.length) {
      voiceLog.innerHTML = '<li class="empty-state">No activity yet.</li>';
      return;
    }
    voiceLog.innerHTML = entries
      .slice()
      .reverse()
      .map((e) => `<li><span class="log-time">${e.time}</span> ${e.message}</li>`)
      .join("");
  }

  function startPolling() {
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/voice/status");
        const data = await res.json();
        renderLog(data.log || []);
        // A command like "start navigation" or "read text" sets a
        // pending redirect server-side so the user actually lands on
        // the right page instead of the mode silently changing behind
        // the scenes while they stay on the Voice Assistant page.
        if (data.redirect) {
          window.location.href = data.redirect;
        }
      } catch (err) {
        /* ignore transient network errors while polling */
      }
    }, 1000);
  }

  function stopPolling() {
    clearInterval(pollTimer);
  }
})();
