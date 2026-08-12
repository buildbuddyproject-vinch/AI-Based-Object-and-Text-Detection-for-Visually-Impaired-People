"""
Shared helpers for tools/analyze_datasets.py and tools/prepare_datasets.py
so the "what counts as a real class vs. export junk" logic lives in
exactly one place instead of being duplicated (and drifting out of
sync, as it briefly did) across both scripts.
"""
import re

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Substrings that indicate a Roboflow free-tier watermark leaked into an
# annotation as if it were a real object class - seen concretely in
# dataset/outdoor/XML Files/.
WATERMARK_MARKERS = [
    "roboflow",
    "collaborate with your team",
    "end-to-end computer vision platform",
]

# A Roboflow export can also embed its own project/version name as a
# stray label, e.g. "day2 - v3 2025-02-05 12-27pm" - real object class
# names never contain a 4-digit year or a "v<digit>" version marker, so
# that combination is a reliable enough signal to treat as junk too.
_VERSION_LABEL_PATTERN = re.compile(r"\b(19|20)\d{2}\b|\bv\d+\b", re.IGNORECASE)


def is_watermark_class(name):
    lowered = name.strip().lower()
    if lowered in ("", "-"):
        return True
    if any(marker in lowered for marker in WATERMARK_MARKERS):
        return True
    return bool(_VERSION_LABEL_PATTERN.search(lowered))


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert a Pascal VOC pixel box to normalized YOLO (cx, cy, w, h)."""
    xmin, xmax = sorted((max(0, xmin), min(img_w, xmax)))
    ymin, ymax = sorted((max(0, ymin), min(img_h, ymax)))
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h
