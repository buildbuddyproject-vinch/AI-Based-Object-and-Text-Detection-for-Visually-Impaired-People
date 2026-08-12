"""
False-Positive Prevention Test
=================================
Exercises the real modules/tracking.py, modules/priority_engine.py,
modules/announcement_manager.py and modules/model_router.py against
synthetic frame sequences that reproduce the specific failure modes
this pipeline exists to prevent - most importantly the "fan mistaken
for airplane on a single noisy frame" scenario the master prompt uses
as its motivating example.

This is a standalone script (not pytest) so it can be run on demand
and prints a readable PASS/FAIL report - see also tests/test_tracking.py,
tests/test_priority_engine.py, tests/test_announcement_manager.py and
tests/test_model_router.py for the unit-test-level coverage of each
module in isolation.

Run:
    python tools/test_false_positives.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tracking import ObjectTracker  # noqa: E402
from modules.priority_engine import classify_priority, select_most_relevant  # noqa: E402
from modules.announcement_manager import AnnouncementManager  # noqa: E402
from modules.model_router import ModelRouter  # noqa: E402

_PASS = []
_FAIL = []


def check(name, condition, detail=""):
    if condition:
        _PASS.append(name)
        print(f"  PASS  {name}")
    else:
        _FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


class FakeSpeaker:
    """Stands in for utils/text_to_speech.py's Speaker - records every
    utterance instead of touching pyttsx3/the audio hardware, so this
    script can run headless with no speakers/mic attached."""

    def __init__(self):
        self.said = []
        self.stopped = 0

    def speak(self, text, dedup_key=None, force=False):
        self.said.append(text)

    def stop(self):
        self.stopped += 1


def box(cx, cy, w=0.2, h=0.3):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---------------------------------------------------------------------------
# Scenario 1: the exact fan/airplane example from the master prompt - a
# single noisy misclassified frame must NOT reach the user.
# ---------------------------------------------------------------------------
def scenario_single_frame_misclassification():
    print("\n[Scenario 1] Single-frame 'airplane' misread must never be announced")
    tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=4,
                             iou_match_threshold=0.3, stale_after_seconds=1.5)
    t = 0.0
    confirmed_labels_seen = set()

    # Frames 1-2: a spinning fan is briefly misread as "airplane".
    for _ in range(2):
        confirmed = tracker.update(
            [{"label": "airplane", "bbox": box(0.5, 0.3), "confidence": 0.55}], now=t
        )
        confirmed_labels_seen.update(c.label for c in confirmed)
        t += 0.2

    # Frame 3: confidence drops below the floor entirely (real detector
    # noise) - detection is dropped outright, track goes unmatched.
    confirmed = tracker.update([], now=t)
    confirmed_labels_seen.update(c.label for c in confirmed)
    t += 0.2

    # Frame 4: nothing detected at all.
    confirmed = tracker.update([], now=t)
    confirmed_labels_seen.update(c.label for c in confirmed)

    check(
        "'airplane' never reaches consecutive-frame confirmation threshold",
        "airplane" not in confirmed_labels_seen,
        detail=f"saw={confirmed_labels_seen}",
    )


# ---------------------------------------------------------------------------
# Scenario 2: a REAL object (e.g. a fan, once a fan model exists) that
# is consistently detected across consecutive frames MUST be confirmed
# - the tracker should not be so strict it suppresses genuine objects.
# ---------------------------------------------------------------------------
def scenario_consistent_detection_confirms():
    print("\n[Scenario 2] A consistently-detected real object IS confirmed")
    tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=4,
                             iou_match_threshold=0.3, stale_after_seconds=1.5)
    confirmed = []
    t = 0.0
    for conf in (0.70, 0.78, 0.84, 0.86):
        confirmed = tracker.update(
            [{"label": "chair", "bbox": box(0.5, 0.3), "confidence": conf}], now=t
        )
        t += 0.2
    check(
        "'chair' is confirmed after 4 consecutive consistent frames",
        any(c.label == "chair" for c in confirmed),
        detail=f"confirmed={[c.label for c in confirmed]}",
    )


# ---------------------------------------------------------------------------
# Scenario 3: low-confidence detections must never start a track at
# all, regardless of how many frames they appear in.
# ---------------------------------------------------------------------------
def scenario_below_floor_never_tracked():
    print("\n[Scenario 3] Below-confidence-floor detections never accumulate")
    tracker = ObjectTracker(min_confidence=0.4, min_consecutive_frames=4)
    confirmed = []
    t = 0.0
    for _ in range(10):
        confirmed = tracker.update(
            [{"label": "dog", "bbox": box(0.5, 0.3), "confidence": 0.2}], now=t
        )
        t += 0.2
    check(
        "sub-floor 'dog' detections never confirm even after 10 frames",
        len(confirmed) == 0,
        detail=f"confirmed={[c.label for c in confirmed]}",
    )


# ---------------------------------------------------------------------------
# Scenario 4: the announcement manager must not spam the same
# confirmed object every single frame - and must honor a CRITICAL
# interrupt (e.g. "Obstacle ahead, stop.") even mid-cooldown.
# ---------------------------------------------------------------------------
def scenario_announcement_cooldown_and_interrupt():
    print("\n[Scenario 4] Cooldown suppresses repeats; CRITICAL interrupts speech")
    speaker = FakeSpeaker()
    mgr = AnnouncementManager(speaker, cooldown_seconds=6.0, unknown_object_cooldown=15.0)

    mgr.announce("Chair ahead.", key="chair", tier="MEDIUM")
    mgr.announce("Chair ahead.", key="chair", tier="MEDIUM")
    mgr.announce("Chair ahead.", key="chair", tier="MEDIUM")
    check(
        "repeated MEDIUM-tier announcement is suppressed by cooldown, not repeated 3x",
        speaker.said.count("Chair ahead.") == 1,
        detail=f"said={speaker.said}",
    )

    mgr.announce("Obstacle ahead, stop.", key="obstacle", tier="CRITICAL")
    check(
        "CRITICAL announcement calls speaker.stop() to interrupt current speech",
        speaker.stopped >= 1,
    )
    check(
        "CRITICAL announcement is spoken even though 'chair' cooldown is active",
        "Obstacle ahead, stop." in speaker.said,
    )


# ---------------------------------------------------------------------------
# Scenario 5: an unrecognized/low-confidence object must produce an
# honest "not clearly recognized" message, never a fabricated guess.
# ---------------------------------------------------------------------------
def scenario_unknown_object_is_honest():
    print("\n[Scenario 5] Unrecognized objects get an honest message, never a guess")
    speaker = FakeSpeaker()
    mgr = AnnouncementManager(speaker, cooldown_seconds=6.0, unknown_object_cooldown=15.0)
    mgr.announce_unknown()
    check(
        "unknown-object announcement uses the honest fixed phrase",
        speaker.said and speaker.said[-1] == "Object not clearly recognized.",
        detail=f"said={speaker.said}",
    )


# ---------------------------------------------------------------------------
# Scenario 6: the model router must NEVER silently return a COCO
# detector for a domain whose custom model isn't trained yet.
# ---------------------------------------------------------------------------
def scenario_router_never_silently_falls_back():
    print("\n[Scenario 6] Missing domain model -> None, never a silent COCO substitute")
    fake_models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
    )
    router = ModelRouter(models_dir=fake_models_dir, device="cpu")
    # 'road_hazards' and 'currency' have no trained best.pt yet (per the
    # actual models/ directory state) - confirm the router is honest
    # about that instead of quietly substituting something else.
    for domain in ("road_hazards", "currency"):
        if router.is_available(domain):
            print(f"  (skip) {domain} unexpectedly has a trained model - nothing to test here")
            continue
        detector = router.get_detector(domain)
        check(
            f"get_detector('{domain}') returns None when no trained model exists",
            detector is None,
        )
    check(
        "status_report() only reports AVAILABLE/NOT AVAILABLE, never invents a third state",
        set(router.status_report().values()) <= {"AVAILABLE", "NOT AVAILABLE"},
        detail=f"{router.status_report()}",
    )


# ---------------------------------------------------------------------------
# Scenario 7: priority selection must never pick "person" over an
# actively path-blocking obstacle, and must never silently drop a
# CRITICAL candidate in favor of a larger-but-lower-tier one.
# ---------------------------------------------------------------------------
def scenario_priority_prefers_critical_over_large_low_tier():
    print("\n[Scenario 7] CRITICAL obstacle outranks a merely large low-tier object")
    candidates = [
        {
            "label": "couch",
            "tier": classify_priority("couch", is_very_close=False, blocks_path=False),
            "area_ratio": 0.6,
        },
        {
            "label": "stairs",
            "tier": classify_priority("stairs", is_very_close=True, blocks_path=True),
            "area_ratio": 0.05,
        },
    ]
    winner = select_most_relevant(candidates)
    check(
        "small-but-CRITICAL 'stairs' beats large-but-low-tier 'couch'",
        winner is not None and winner["label"] == "stairs",
        detail=f"winner={winner}",
    )


def main():
    scenarios = [
        scenario_single_frame_misclassification,
        scenario_consistent_detection_confirms,
        scenario_below_floor_never_tracked,
        scenario_announcement_cooldown_and_interrupt,
        scenario_unknown_object_is_honest,
        scenario_router_never_silently_falls_back,
        scenario_priority_prefers_critical_over_large_low_tier,
    ]
    print("=" * 78)
    print("FALSE-POSITIVE / SAFETY-BEHAVIOR TEST")
    print("=" * 78)
    for scenario in scenarios:
        scenario()

    print("\n" + "=" * 78)
    print(f"RESULT: {len(_PASS)} passed, {len(_FAIL)} failed")
    print("=" * 78)
    if _FAIL:
        print("FAILED:")
        for name in _FAIL:
            print(f"  - {name}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
