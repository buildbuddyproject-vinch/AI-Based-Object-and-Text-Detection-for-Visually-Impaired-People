"""
Threaded Camera Stream
========================
Continuously grabs frames from the webcam on its own thread so frame
capture is never blocked by (slower) AI inference, keeping the live
preview smooth. Call `get_frame()` from any thread to fetch a copy of the
most recently captured frame.
"""
import platform
import threading
import time

import cv2


class VideoCamera:
    def __init__(self, source=0, width=640, height=480):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, backend)

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None

        if self.cap.isOpened():
            self._running = True
            self._thread = threading.Thread(target=self._update, daemon=True)
            self._thread.start()

    @property
    def is_opened(self):
        return self.cap.isOpened()

    def _update(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.05)

    def get_frame(self):
        """Return a copy of the latest captured frame, or None if no
        frame has been captured yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        time.sleep(0.1)
        if self.cap.isOpened():
            self.cap.release()
