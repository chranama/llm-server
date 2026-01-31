# integrations/test_generate_only/test_persistence_inference_logs.py
from __future__ import annotations

import pytest

from integrations.lib.fixtures import load_prompt
from integrations.lib.db import fetch_inference_log_count

pytestmark = [pytest.mark.asyncio, pytest.mark.generate_only, pytest.mark.requires_api_key, pytest.mark.requires_db]


async def test_inference_logs_increase_after_generate(client, assert_mode):
    """
    Requires a DB-backed deployment AND an introspection method.
    `fetch_inference_log_count` should use whatever you expose:
      - admin endpoint (preferred)
      - stats endpoint
      - or a dedicated test-only endpoint
    """
    assert assert_mode is True

    before = await fetch_inference_log_count(client)

    prompt = load_prompt("generate_ping.txt")
    r = await client.post("/v1/generate", json={"prompt": prompt, "max_new_tokens": 8, "temperature": 0.2})
    r.raise_for_status()

    after = await fetch_inference_log_count(client)
    assert after >= before + 1, f"expected inference logs to increase (before={before}, after={after})"