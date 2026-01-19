from __future__ import annotations

import os
from typing import Optional

import pytest

from llm_eval.client.http_client import HttpEvalClient


def _get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


@pytest.fixture(scope="session")
def integration_base_url() -> str:
    """
    Base URL for live llm-server instance.

    Set:
      - INTEGRATION_BASE_URL="http://localhost:8000"
        (or nginx: "http://localhost:8080/api")
    """
    v = _get_env("INTEGRATION_BASE_URL") or _get_env("LLM_SERVER_BASE_URL")
    if not v:
        pytest.skip("Missing INTEGRATION_BASE_URL (or LLM_SERVER_BASE_URL). Skipping integration tests.")
    return v.rstrip("/")


@pytest.fixture(scope="session")
def integration_api_key() -> str:
    """
    API key for live llm-server instance.

    Set:
      - API_KEY="..."
    """
    v = _get_env("API_KEY")
    if not v:
        pytest.skip("Missing API_KEY. Skipping integration tests.")
    return v


@pytest.fixture
def live_client(integration_base_url: str, integration_api_key: str) -> HttpEvalClient:
    return HttpEvalClient(base_url=integration_base_url, api_key=integration_api_key, timeout=60.0)