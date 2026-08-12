"""
Multi-Domain Pipeline Settings Loader
========================================
Loads config/config.yaml (camera/models/confidence/tracking/speech/
navigation settings for the household/indoor/outdoor/road-hazard/
currency/footpath auto-assistance pipeline).

IMPORTANT - why this file lives in utils/, not inside config/: the
project already has a top-level config.py module (used by the original
single-domain web app in app.py). Python resolves `import config` to
that regular module, which means a `config/` *package* of the same
name is unreachable via `from config.whatever import ...` - a genuine
Python name collision, confirmed by testing, not a style preference.
Rather than delete/rename the existing working config.py (and every
file that already imports it), the YAML file stays at the path Section
34 asks for (config/config.yaml), but is loaded here by plain file
path instead of via a Python package import.
"""
import os

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_YAML_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_pipeline_settings(path=CONFIG_YAML_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.yaml not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# Loaded once at import time - shared, read-only view of config.yaml.
# Call load_pipeline_settings() directly (not this cached copy) if you
# need per-test isolation.
settings = load_pipeline_settings()
