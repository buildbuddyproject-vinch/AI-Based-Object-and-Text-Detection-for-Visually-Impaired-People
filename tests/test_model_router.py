"""Unit tests for the model router's availability logic - uses a
temporary models/ directory with fake (empty) weight files so this
never needs a real trained model or GPU."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.model_router import ModelRouter, DOMAIN_WEIGHTS


class TestModelRouterAvailability(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _touch(self, rel_path):
        full = os.path.join(self.tmp_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(b"not a real model")

    def test_no_models_present_means_nothing_available(self):
        router = ModelRouter(models_dir=self.tmp_dir)
        self.assertEqual(router.available_domains(), [])
        for domain in DOMAIN_WEIGHTS:
            self.assertFalse(router.is_available(domain))

    def test_partial_availability_is_reported_correctly(self):
        self._touch(DOMAIN_WEIGHTS["indoor"])
        self._touch(DOMAIN_WEIGHTS["currency"])
        router = ModelRouter(models_dir=self.tmp_dir)
        self.assertTrue(router.is_available("indoor"))
        self.assertTrue(router.is_available("currency"))
        self.assertFalse(router.is_available("outdoor"))
        self.assertFalse(router.is_available("household"))
        self.assertEqual(sorted(router.available_domains()), ["currency", "indoor"])

    def test_status_report_format(self):
        self._touch(DOMAIN_WEIGHTS["indoor"])
        router = ModelRouter(models_dir=self.tmp_dir)
        report = router.status_report()
        self.assertEqual(report["indoor"], "AVAILABLE")
        self.assertEqual(report["outdoor"], "NOT AVAILABLE")

    def test_get_detector_for_unavailable_domain_returns_none(self):
        router = ModelRouter(models_dir=self.tmp_dir)
        self.assertIsNone(router.get_detector("household"))

    def test_get_detector_for_present_but_invalid_weights_fails_gracefully(self):
        # A file exists (so it looks "available") but isn't a real model -
        # loading it must fail gracefully (None), not crash the caller.
        self._touch(DOMAIN_WEIGHTS["indoor"])
        router = ModelRouter(models_dir=self.tmp_dir)
        self.assertTrue(router.is_available("indoor"))
        detector = router.get_detector("indoor")
        self.assertIsNone(detector)
        # And availability should now correctly reflect the load failure.
        self.assertFalse(router.is_available("indoor"))

    def test_development_coco_detector_requires_development_mode(self):
        import modules.model_router as model_router_module
        router = ModelRouter(models_dir=self.tmp_dir)
        original = model_router_module.DEVELOPMENT_MODE
        model_router_module.DEVELOPMENT_MODE = False
        try:
            with self.assertRaises(RuntimeError):
                router.get_development_coco_detector("weights/yolov8n.pt")
        finally:
            model_router_module.DEVELOPMENT_MODE = original


if __name__ == "__main__":
    unittest.main()
