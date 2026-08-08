"""
OCR Reader Module
===================
Uses EasyOCR to detect and extract printed text from a captured frame, so
the assistant can read signs, labels, packaging and documents aloud.
"""
import cv2
import numpy as np
import easyocr


class OCRReader:
    def __init__(self, languages=None, gpu=False):
        self.languages = languages or ["en"]
        # EasyOCR downloads its detection/recognition weights on first use
        # and caches them under ~/.EasyOCR - this can take a minute the
        # very first time it runs.
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def read_text(self, frame, min_confidence=0.4):
        """Return (full_text: str, boxes: list[dict]) for text found in
        `frame` whose recognition confidence is >= min_confidence."""
        results = self.reader.readtext(frame)
        lines, boxes = [], []
        for bbox, text, conf in results:
            if conf >= min_confidence and text.strip():
                lines.append(text.strip())
                boxes.append({
                    "bbox": bbox,
                    "text": text.strip(),
                    "confidence": round(float(conf), 2),
                })
        return " ".join(lines), boxes

    @staticmethod
    def draw_text_boxes(frame, boxes, color=(66, 135, 245)):
        """Draw the polygon around each recognized text region."""
        for item in boxes:
            pts = np.array(item["bbox"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        return frame
