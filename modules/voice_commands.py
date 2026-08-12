"""
Voice Command Listener
========================
Listens on the microphone in a background thread using SpeechRecognition
and dispatches recognized phrases to registered callback functions.

Supported commands (registered by app.py):
    "read text" / "read this"      -> capture the current frame with OCR and read it
    "start navigation"              -> switch live detection into navigation mode
    "detect object"                 -> switch live detection into automatic mode
    "automatic mode"                -> automatic mode + clear any indoor/outdoor override
    "indoor mode" / "outdoor mode"  -> pin auto-assistance to one domain model
    "what is ahead" / "what is this" -> repeat the last confirmed situational announcement
    "identify this currency"        -> capture + run currency detection
    "stop speaking" / "stop"        -> interrupt/clear current speech
    "repeat"                        -> speak the last announcement again
    "describe the scene" / "what is around" -> summarize what's currently detected
    "where is my <object>"          -> answer from recent detection memory
    "exit"                          -> stop the voice assistant thread

Each registered callback receives the full recognized phrase as its
only argument (e.g. "where is my backpack") so commands like "where is
my X" can parse out what comes after the fixed prefix - callbacks that
don't need it can just ignore the parameter.

The default recognizer (`recognize_google`) requires an internet
connection. To run fully offline, install `vosk` and swap the body of
`_recognize()` for a Vosk-based recognizer - that is the intended
extension point for offline speech-to-text.
"""
import threading
import time

import speech_recognition as sr


class VoiceCommandListener:
    def __init__(self, on_log=None, is_speaking_check=None):
        self.recognizer = sr.Recognizer()
        # Default pause_threshold (0.8s) can cut a phrase off mid-way
        # through a brief natural pause (e.g. "start" ... "navigation"
        # gets heard as just "start"). Give it more room to breathe.
        self.recognizer.pause_threshold = 1.2
        self.microphone = None
        self._callbacks = {}
        self._thread = None
        self._running = threading.Event()
        self.on_log = on_log or (lambda msg: None)
        # Accepted but intentionally unused for gating the listen loop -
        # see the note in _listen_loop() for why. Kept as a constructor
        # parameter so callers (app.py) don't need to change.
        self._is_speaking_check = is_speaking_check or (lambda: False)

    def register(self, phrase, callback):
        """Register `callback` to fire whenever `phrase` is heard as a
        substring of the recognized speech (case-insensitive). `callback`
        is invoked as `callback(heard_text)` with the full recognized
        phrase, so pattern-style commands (e.g. "where is my X") can
        extract what follows the fixed prefix."""
        self._callbacks[phrase.lower()] = callback

    def _log(self, msg):
        try:
            self.on_log(msg)
        except Exception:
            pass

    def _recognize(self, audio):
        """Speech-to-text step. Swap this for an offline engine (e.g.
        Vosk) if internet access cannot be guaranteed on the target
        device."""
        return self.recognizer.recognize_google(audio)

    def _match_and_dispatch(self, text):
        text = text.lower().strip()
        self._log(f'Heard: "{text}"')
        matched = False
        for phrase, callback in self._callbacks.items():
            if phrase in text:
                matched = True
                try:
                    callback(text)
                except Exception as exc:
                    self._log(f"Command '{phrase}' failed: {exc}")
        if not matched:
            self._log("No matching command.")
        return matched

    def _listen_loop(self):
        try:
            self.microphone = sr.Microphone()
        except Exception as exc:
            self._log(f"Microphone unavailable: {exc}")
            self._running.clear()
            return

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception as exc:
            self._log(f"Could not calibrate microphone: {exc}")

        # Deliberately NOT pausing the microphone while the assistant is
        # talking. An earlier version did exactly that to avoid the mic
        # picking up the assistant's own confirmation speech - but in
        # navigation mode, guidance is spoken every few seconds, so
        # pausing on every utterance left almost no window to actually
        # hear the user, which is a far worse problem than the
        # occasional misheard echo. That echo is low-risk in practice:
        # none of the phrases this app speaks ("Navigation mode
        # activated.", "Person ahead...", "X detected", etc.) contain
        # any of the registered command phrases as a substring, so at
        # worst it produces a harmless "No matching command" log line.
        self._log("Voice assistant listening...")
        while self._running.is_set():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=6)
                text = self._recognize(audio)
                self._match_and_dispatch(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                self._log(f"Speech service error: {exc}")
                time.sleep(2)
            except Exception as exc:
                self._log(f"Listener error: {exc}")
                time.sleep(1)

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._running.is_set():
            self._running.clear()
            self._log("Voice assistant stopped.")

    @property
    def is_running(self):
        return self._running.is_set()
