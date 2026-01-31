# backend/tests/unit/conftest.py
from __future__ import annotations

import os

# Unit tests should not depend on DB/Redis.
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DEBUG", "0")