"""
Scene Summary Module
======================
Composes a short natural-language description of the current scene
from raw YOLO detections and their left/center/right zones, e.g. "A
person is ahead of you. A chair is on your left."

This is deliberately template-based, not a captioning/vision-language
model - it's built entirely from data the app already computes (labels
+ zones), so it's fast enough to run on demand with no extra model
weights or GPU/CPU cost beyond what object detection already pays.
"""


def _zone_phrase(zone):
    if zone == "left":
        return "on your left"
    if zone == "right":
        return "on your right"
    if zone == "center":
        return "ahead of you"
    return "nearby"


def build_scene_summary(detections, zone_map=None, max_items=4):
    """Return a spoken-friendly summary sentence for the current
    detections. `zone_map` (as produced by NavigationAssistant.analyze)
    is optional - without it, every object is just described as
    "nearby"."""
    if not detections:
        return "I don't see anything notable right now."

    counts = {}
    for det in detections:
        counts[det["label"]] = counts.get(det["label"], 0) + 1

    zone_for_label = {}
    if zone_map:
        for zone, labels in zone_map.items():
            for label in labels:
                zone_for_label.setdefault(label, zone)

    phrases = []
    for label, count in list(counts.items())[:max_items]:
        location = _zone_phrase(zone_for_label.get(label))
        if count > 1:
            phrases.append(f"{count} {label}s {location}")
        else:
            article = "An" if label[:1].lower() in "aeiou" else "A"
            phrases.append(f"{article} {label} {location}")

    remaining = len(counts) - max_items
    sentence = ". ".join(p[0].upper() + p[1:] for p in phrases) + "."
    if remaining > 0:
        sentence += f" And {remaining} other thing" + ("s" if remaining != 1 else "") + "."
    return sentence
