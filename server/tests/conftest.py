# backend/tests/conftest.py
from __future__ import annotations

import os

# Global test defaults.
# These must be set BEFORE importing llm_server modules, because settings are created at import time.
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DEBUG", "0")

# Never require a model for readiness in tests unless a test opts in.
os.environ.setdefault("REQUIRE_MODEL_READY", "0")

# Prevent accidental external dependencies in tests.
os.environ.setdefault("REDIS_ENABLED", "0")

# Default to "off" so the app never auto-builds/loads a real model during unit tests.
# Integration tests typically patch build_llm_from_settings anyway.
os.environ.setdefault("MODEL_LOAD_MODE", "off")