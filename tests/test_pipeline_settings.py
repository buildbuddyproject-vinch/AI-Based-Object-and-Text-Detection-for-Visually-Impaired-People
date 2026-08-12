"""Unit test confirming config/config.yaml loads correctly and has the
sections the rest of the pipeline depends on."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline_settings import load_pipeline_settings


class TestPipelineSettings(unittest.TestCase):
    def test_loads_expected_top_level_sections(self):
        settings = load_pipeline_settings()
        for key in ("camera", "models", "confidence", "tracking", "speech", "navigation"):
            self.assertIn(key, settings)

    def test_all_six_domains_have_a_model_path_entry(self):
        settings = load_pipeline_settings()
        for domain in ("household", "indoor", "outdoor", "road_hazards", "currency", "footpath"):
            self.assertIn(domain, settings["models"])

    def test_missing_file_raises_filenotfounderror(self):
        with self.assertRaises(FileNotFoundError):
            load_pipeline_settings("/nonexistent/path/config.yaml")


if __name__ == "__main__":
    unittest.main()
