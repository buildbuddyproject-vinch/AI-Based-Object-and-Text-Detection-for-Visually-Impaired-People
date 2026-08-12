"""
Priority Engine
==================
Given a list of currently-confirmed tracked objects (see
modules/tracking.py), decides which ONE thing, if any, is most
important to tell the user about right now. The system must never
narrate every detection every frame - this is what enforces that.

Priority tiers (highest first):
    CRITICAL - approaching vehicle, pothole, speed breaker, an obstacle
               that is very close or fully blocks the path, dangerous
               stairs
    HIGH     - person, motorcycle/bicycle, pole/pillar, a door in the way
    MEDIUM   - chair, table, cabinet and similar furniture
    LOW      - everything else (bottle, cup, small objects, ...)

Within a tier, the closer/larger object (bigger bbox area) wins - a
CRITICAL item always beats a HIGH item regardless of size.
"""

CRITICAL_LABELS = {"pothole", "speed breaker", "speed_breaker"}
HIGH_LABELS = {
    "person", "motorcycle", "bike", "bicycle", "pole", "pillar",
    "car", "truck", "bus", "auto rickshaw", "door", "openeddoor", "openeddoor",
}
MEDIUM_LABELS = {"chair", "table", "cabinet", "sofa", "couch", "armchair", "bed"}

TIER_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def classify_priority(label, is_very_close=False, blocks_path=False):
    """Return "CRITICAL"/"HIGH"/"MEDIUM"/"LOW" for a label. Escalates
    to CRITICAL regardless of the label's normal tier if it's
    dangerously close or fully blocking the path."""
    if blocks_path or is_very_close:
        return "CRITICAL"
    lowered = label.lower()
    if lowered in CRITICAL_LABELS:
        return "CRITICAL"
    if lowered in HIGH_LABELS:
        return "HIGH"
    if lowered in MEDIUM_LABELS:
        return "MEDIUM"
    return "LOW"


def select_most_relevant(candidates):
    """`candidates`: list of dicts with at least {"label"}, optionally
    "tier" (computed via classify_priority if absent) and "area_ratio"
    (0-1, used as a tiebreaker within a tier - bigger wins). Returns the
    single most relevant candidate dict, or None if the list is empty.
    Always returns at most one - the announcement manager is what
    decides whether/when to actually speak it."""
    if not candidates:
        return None

    def sort_key(c):
        tier = c.get("tier") or classify_priority(
            c["label"], c.get("is_very_close", False), c.get("blocks_path", False)
        )
        return (TIER_RANK.get(tier, 3), -c.get("area_ratio", 0.0))

    return min(candidates, key=sort_key)
