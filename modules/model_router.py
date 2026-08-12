"""
Model Router
==============
Central registry for every domain-specific detection model (household,
indoor, outdoor, road hazards, currency, footpath). Loads models
LAZILY (on first actual use, not at import time) and NEVER silently
substitutes a missing custom model with the generic COCO model in
production - if a domain's trained model isn't available, that domain
is simply disabled and reported as such (Section 43: "Partial model
availability").

A COCO-pretrained YOLO model may only be used behind an explicit
DEVELOPMENT_MODE=1 environment variable, for development/manual
testing - and even then it's a clearly separate code path from any
real trained domain model, never a silent fallback (Section 9).
"""
import os

from modules.object_detector import ObjectDetector
from utils.logger import get_logger

log = get_logger()

DEVELOPMENT_MODE = os.environ.get("DEVELOPMENT_MODE", "0") == "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# domain -> expected weights file, relative to models/
DOMAIN_WEIGHTS = {
    "household": "household/best.pt",
    "indoor": "indoor/best.pt",
    "outdoor": "outdoor/best.pt",
    "road_hazards": "road_hazards/best.pt",
    "currency": "currency/best.pt",
    "footpath": "footpath/best.pt",
}


class ModelRouter:
    def __init__(self, confidence_by_domain=None, device="cpu", models_dir=MODELS_DIR):
        self.confidence_by_domain = confidence_by_domain or {}
        self.device = device
        self.models_dir = models_dir
        self._loaded = {}        # domain -> ObjectDetector | None (None = load attempted and failed)
        self._availability = {}  # domain -> bool, computed at construction
        self._scan_availability()

    def _scan_availability(self):
        for domain, rel_path in DOMAIN_WEIGHTS.items():
            full_path = os.path.join(self.models_dir, rel_path)
            available = os.path.exists(full_path)
            self._availability[domain] = available
            if available:
                log.info("Model available: %s -> %s", domain, full_path)
            else:
                log.warning(
                    "%s model unavailable. Place/train models/%s to enable %s detection.",
                    domain.capitalize(), rel_path, domain,
                )

    def is_available(self, domain):
        return self._availability.get(domain, False)

    def available_domains(self):
        return [d for d, ok in self._availability.items() if ok]

    def get_detector(self, domain):
        """Return a loaded ObjectDetector for `domain`, or None if that
        domain's custom model isn't available. Never falls back to
        COCO here - see get_development_coco_detector() for that,
        which is explicitly opt-in only."""
        if not self._availability.get(domain, False):
            return None
        if domain not in self._loaded:
            full_path = os.path.join(self.models_dir, DOMAIN_WEIGHTS[domain])
            try:
                self._loaded[domain] = ObjectDetector(
                    model_path=full_path,
                    conf_threshold=self.confidence_by_domain.get(domain, 0.45),
                    device=self.device,
                )
            except Exception as exc:
                log.error("Failed to load %s model: %s", domain, exc)
                self._loaded[domain] = None
                self._availability[domain] = False
        return self._loaded[domain]

    def get_development_coco_detector(self, weights_path, confidence=0.45):
        """A generic COCO-pretrained model, for development/manual
        testing ONLY - never used as a silent production fallback.
        Raises unless DEVELOPMENT_MODE=1 is set, so it can't be reached
        accidentally by production code paths."""
        if not DEVELOPMENT_MODE:
            raise RuntimeError(
                "get_development_coco_detector() called without DEVELOPMENT_MODE=1 - "
                "the generic COCO model must never be used as a silent production fallback."
            )
        if "_dev_coco" not in self._loaded:
            self._loaded["_dev_coco"] = ObjectDetector(
                model_path=weights_path, conf_threshold=confidence, device=self.device
            )
        return self._loaded["_dev_coco"]

    def status_report(self):
        """{"household": "AVAILABLE", "outdoor": "NOT AVAILABLE", ...} -
        exactly the format Section 43 asks the app to be able to report."""
        return {domain: ("AVAILABLE" if ok else "NOT AVAILABLE")
                 for domain, ok in self._availability.items()}

    def set_domain_confidence(self, confidence_by_domain):
        """Update per-domain confidence thresholds after construction
        (app.py calls this at startup with config/config.yaml's
        `confidence:` section). Only affects models loaded AFTER this
        call - if a domain's detector was already lazily loaded with
        the old threshold, drop its cache so the next get_detector()
        call re-loads it with the new one instead of silently keeping
        the stale value."""
        self.confidence_by_domain.update(confidence_by_domain)
        for domain in list(self._loaded.keys()):
            if domain in confidence_by_domain:
                del self._loaded[domain]


# Single shared instance - app.py wires real per-domain confidence
# thresholds into this at startup via set_domain_confidence() (see
# config/config.yaml's `confidence:` section).
router = ModelRouter()
