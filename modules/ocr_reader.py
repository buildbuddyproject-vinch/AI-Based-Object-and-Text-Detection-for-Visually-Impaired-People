"""
OCR Reader Module
===================
Uses EasyOCR to detect and extract printed text from a captured frame, so
the assistant can read signs, labels, packaging and documents aloud.

Beyond raw extraction, this module also:
  - orders text into a sensible top-to-bottom, left-to-right reading
    order (EasyOCR's own result order roughly follows detection
    confidence/position, not layout) - see `_group_into_lines()`.
  - flags "important" text (warnings, exits, hazards, prices, dates) so
    the caller can announce it first / more emphatically instead of
    burying it in the middle of a long read-out - see
    `find_important_keyword()`.
"""
import cv2
import numpy as np
import easyocr

# Words worth calling out before the rest of the text - deliberately a
# small, high-signal list (safety/orientation/money-related) rather than
# an exhaustive one, so it doesn't fire on every third sign.
IMPORTANT_KEYWORDS = [
    "danger", "warning", "caution", "emergency", "fire", "hazard",
    "poison", "toxic", "evacuate", "restricted",
    "exit", "no entry", "do not enter", "stop", "help",
    "wet floor", "slippery",
]


def find_important_keyword(text):
    """Return the first important keyword found in `text` (case
    insensitive), or None if it doesn't contain any."""
    lowered = text.lower()
    for kw in IMPORTANT_KEYWORDS:
        if kw in lowered:
            return kw
    return None


def _group_into_lines(items):
    """Cluster OCR results into text lines by vertical proximity, then
    sort each line left-to-right and the lines themselves top-to-bottom
    - a simple stand-in for real document-layout analysis, using only
    the bounding boxes EasyOCR already returns."""
    items_sorted = sorted(items, key=lambda it: it["cy"])
    lines = []
    for it in items_sorted:
        placed = False
        for line in lines:
            ref_cy = sum(x["cy"] for x in line) / len(line)
            ref_h = sum(x["h"] for x in line) / len(line)
            if abs(it["cy"] - ref_cy) <= max(ref_h, it["h"]) * 0.6:
                line.append(it)
                placed = True
                break
        if not placed:
            lines.append([it])

    lines.sort(key=lambda line: sum(x["cy"] for x in line) / len(line))
    for line in lines:
        line.sort(key=lambda x: x["cx"])
    return lines


class OCRReader:
    def __init__(self, languages=None, gpu=False):
        self.languages = languages or ["en"]
        # EasyOCR downloads its detection/recognition weights on first use
        # and caches them under ~/.EasyOCR - this can take a minute the
        # very first time it runs.
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def read_text(self, frame, min_confidence=0.4):
        """Return (full_text: str, boxes: list[dict]) for text found in
        `frame` whose recognition confidence is >= min_confidence.
        `full_text` is ordered into reading lines (see
        `_group_into_lines`), not just EasyOCR's raw result order."""
        results = self.reader.readtext(frame)
        items, boxes = [], []
        for bbox, text, conf in results:
            text = text.strip()
            if conf >= min_confidence and text:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                items.append({
                    "text": text,
                    "cx": sum(xs) / len(xs),
                    "cy": sum(ys) / len(ys),
                    "h": max(ys) - min(ys),
                })
                boxes.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": round(float(conf), 2),
                })

        if not items:
            return "", boxes

        lines = _group_into_lines(items)
        full_text = "\n".join(" ".join(it["text"] for it in line) for line in lines)
        return full_text, boxes

    @staticmethod
    def draw_text_boxes(frame, boxes, color=(66, 135, 245)):
        """Draw the polygon around each recognized text region."""
        for item in boxes:
            pts = np.array(item["bbox"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        return frame
