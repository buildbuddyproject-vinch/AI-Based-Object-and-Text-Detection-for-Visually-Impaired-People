"""
Speaker Module
===============
Offline text-to-speech using pyttsx3, running on a dedicated background
thread with a queue so multiple parts of the app (object detector, OCR,
navigation, voice assistant) can request speech without blocking Flask
requests or clashing with each other.

Each utterance is spoken in its own short-lived subprocess (this file
re-invokes itself with `--speak-worker`) rather than reusing one
long-lived pyttsx3 engine in-process. That is not just tidiness: reusing
one engine across many say()/runAndWait() calls is a well-documented
pyttsx3-on-Windows bug (works once, then silently renders no audio for
every call after that), and repeatedly creating/destroying engines
*inside the same long-running process* was observed here to eventually
segfault the entire Flask server outright, from inside pyttsx3's native
COM driver, uncatchable by any Python try/except. Giving each utterance
its own OS process means a crash there can never take the server down,
and every utterance always starts from clean COM state.

Includes simple de-duplication (via `dedup_key` + cooldown) so the same
phrase (e.g. "chair detected") isn't repeated every single frame while
the object stays in view.
"""
import os
import platform
import queue
import subprocess
import sys
import threading
import time

from utils.logger import get_logger

# Reuse the app's single top-level logger rather than a dotted child
# name - a child logger would propagate records up to the parent's
# handler too, printing every line twice.
log = get_logger()

# Project root (parent of this modules/ directory) - the worker
# subprocess is launched with this as its working directory via `python
# -m modules.speaker`, so `utils`/`modules` resolve as top-level
# packages the same way they do for the main Flask process.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _force_full_volume_windows(pid=None):
    """Best-effort fix for a real-world Windows quirk: the OS persists a
    per-application volume in the Volume Mixer keyed by executable, and
    on some machines python.exe/python3.11.exe gets silently pinned to a
    near-zero level from an unrelated earlier session - independent of
    pyttsx3's own `volume` property, and not always fixable by dragging
    the mixer slider or clicking "Reset". This reaches into the Windows
    Core Audio API directly and forces the given process's (default:
    this process's) audio session(s) to full, unmuted volume.

    Returns True if a matching audio session was found and fixed, False
    otherwise (nothing to fix yet, wrong platform, or pycaw unavailable)
    - callers that run this before the process has produced any audio
    yet (no session exists to find) should retry a few times rather than
    treating one False as final. See callers in _run_as_speak_worker().

    No-op (silently) on non-Windows platforms or if `pycaw` isn't
    installed - it is an optional, Windows-only convenience, not a hard
    dependency.
    """
    if platform.system() != "Windows":
        return False
    try:
        from pycaw.pycaw import AudioUtilities

        target_pid = pid if pid is not None else os.getpid()
        matched = False
        for session in AudioUtilities.GetAllSessions():
            if session.Process and session.Process.pid == target_pid:
                volume = session.SimpleAudioVolume
                volume.SetMasterVolume(1.0, None)
                volume.SetMute(0, None)
                matched = True
        return matched
    except Exception as exc:
        log.warning("pycaw volume-forcing failed (%s) - falling back to whatever "
                    "volume Windows already has set for this process", exc)
        return False


class Speaker:
    def __init__(self, rate=170, volume=1.0, voice_id=None, cooldown=6.0):
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.cooldown = cooldown

        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._last_spoken = {}
        self._stop_event = threading.Event()
        # The subprocess currently speaking, if any - only set while an
        # utterance is actively playing, so stop() has something to
        # terminate.
        self._current_process = None
        # The last non-dedup-skipped phrase actually queued, so a
        # "repeat that" voice command / gesture has something to say
        # again - see repeat_last().
        self._last_text = None
        # Lets other components (the voice command listener, in
        # particular) know not to have the microphone listen while the
        # assistant is talking - otherwise it hears its own voice and
        # tries to parse it as a new command.
        self._speaking_event = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not text:
                continue
            self._speak_once(text)

    def _speak_once(self, text):
        args = [sys.executable, "-m", "modules.speaker", "--speak-worker",
                str(self.rate), str(self.volume), self.voice_id or ""]

        self._speaking_event.set()
        t0 = time.time()
        try:
            log.info('Speaking: "%s"', text)
            proc = subprocess.Popen(
                args,
                cwd=_PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._current_process = proc
            try:
                _, stderr = proc.communicate(input=text, timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                log.error("TTS worker timed out for %r", text)
                return

            if proc.returncode != 0:
                log.error("TTS worker exited with code %s for %r: %s",
                          proc.returncode, text, (stderr or "").strip())
            else:
                log.info("Finished speaking (%.2fs)", time.time() - t0)
                # Surface the worker's own diagnostics (e.g. whether the
                # volume-forcing fix found a session to fix) even on
                # success - it prints to stderr precisely so this isn't
                # silently swallowed by stdout=DEVNULL above.
                if stderr and stderr.strip():
                    log.info("TTS worker diagnostics: %s", stderr.strip())
        except Exception as exc:
            log.error("Failed to run TTS worker for %r: %s", text, exc)
        finally:
            self._current_process = None
            self._speaking_event.clear()

    @property
    def is_speaking(self):
        return self._speaking_event.is_set()

    def speak(self, text, dedup_key=None, force=False):
        """Queue `text` to be spoken.

        If `dedup_key` is given, the phrase is skipped when the same key
        was already spoken within `self.cooldown` seconds (unless
        `force=True`), preventing repetitive narration.
        """
        if not text:
            return

        if dedup_key and not force:
            with self._lock:
                last = self._last_spoken.get(dedup_key, 0)
                now = time.time()
                if now - last < self.cooldown:
                    log.debug('Skipped (cooldown, key=%s): "%s"', dedup_key, text)
                    return
                self._last_spoken[dedup_key] = now

        # Cap the queue so a burst of detections can't pile up minutes of
        # stale speech behind the live feed.
        if self._queue.qsize() < 10:
            self._queue.put(text)
            self._last_text = text
        else:
            log.warning('Speech queue full - dropped: "%s"', text)

    def repeat_last(self):
        """Speak the last phrase that was actually queued again (for a
        "repeat that" voice command or gesture). No-op if nothing has
        been spoken yet."""
        if self._last_text:
            self.speak(self._last_text, force=True)
        else:
            self.speak("I haven't said anything yet.", force=True)

    def stop(self):
        """Clear any pending speech and interrupt what is currently
        being spoken."""
        with self._queue.mutex:
            self._queue.queue.clear()
        if self._current_process is not None:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def shutdown(self):
        self.stop()
        self._stop_event.set()
        self._thread.join(timeout=1)


def _run_as_speak_worker(argv):
    """Entry point used when this file is re-invoked as a standalone TTS
    worker subprocess (see Speaker._speak_once). Reads the text to speak
    from stdin and exits as soon as it's done - one utterance per
    process, then gone."""
    import pyttsx3

    rate = int(argv[0]) if len(argv) > 0 and argv[0] else 170
    volume = float(argv[1]) if len(argv) > 1 and argv[1] else 1.0
    voice_id = argv[2] if len(argv) > 2 and argv[2] else None

    text = sys.stdin.read()
    if not text:
        return

    try:
        engine = pyttsx3.init()
    except Exception as exc:
        # No usable TTS driver on this machine - e.g. a cloud host with
        # no `espeak`/`espeak-ng` installed (pyttsx3's Linux backend) or
        # no SAPI5 voices (Windows). Fail quietly with a clear message
        # rather than a raw traceback; the app keeps working visually,
        # it just can't speak on a box with no audio stack at all.
        print(f"[tts-worker] no TTS driver available: {exc}", file=sys.stderr)
        sys.exit(1)

    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    if voice_id:
        engine.setProperty("voice", voice_id)

    # Confirmed by direct measurement: the audio session for a brand-new
    # process isn't visible to WASAPI's session enumerator merely by
    # having called pyttsx3.init() - 10 retries (500ms) right after
    # init() and before say() never found it. It only appears once the
    # engine actually starts rendering audio, i.e. *during*
    # runAndWait(). So the fix has to poll concurrently with playback,
    # not just once beforehand. That's normally a real risk (earlier,
    # concurrent Windows Audio COM calls during playback were the
    # direct cause of a segfault) - but that was specifically from
    # reusing one engine for many utterances in a single long-running
    # process; here every subprocess speaks exactly once and then exits,
    # so there's no accumulation across calls for it to destabilize.
    fixed = threading.Event()
    stop_guard = threading.Event()

    def _volume_guard():
        while not stop_guard.is_set():
            if _force_full_volume_windows():
                fixed.set()
            stop_guard.wait(0.1)

    guard_thread = threading.Thread(target=_volume_guard, daemon=True)
    guard_thread.start()
    try:
        engine.say(text)
        engine.runAndWait()
    finally:
        stop_guard.set()
        guard_thread.join(timeout=1)

    # Printed to stderr (not the `log` object - that would go to this
    # process's own stdout, which the parent discards) so the parent
    # Speaker can surface it regardless of success/failure.
    print(f"[tts-worker] volume-forced={fixed.is_set()} pid={os.getpid()}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--speak-worker":
        _run_as_speak_worker(sys.argv[2:])
