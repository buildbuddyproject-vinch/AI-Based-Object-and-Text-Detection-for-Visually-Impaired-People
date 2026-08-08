"""
QR Code Reader Module (bonus feature)
========================================
Detects and decodes QR codes using OpenCV's built-in QRCodeDetector, so no
extra native dependency (e.g. zbar) is required.
"""
import cv2


class QRReader:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def read(self, frame):
        """Return a list of {"data": str, "bbox": [[x, y], ...]} for
        every QR code found in `frame`."""
        results = []
        if frame is None:
            return results

        try:
            ok, decoded_info, points, _ = self.detector.detectAndDecodeMulti(frame)
        except cv2.error:
            return results

        if ok and points is not None:
            for text, box in zip(decoded_info, points):
                if text:
                    results.append({"data": text, "bbox": box.tolist()})
        return results
