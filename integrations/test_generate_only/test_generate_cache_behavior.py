# integrations/test_generate_only/test_generate_cache_behavior.py
from __future__ import annotations

import pytest

from integrations.lib.fixtures import load_prompt

pytestmark = [pytest.mark.asyncio, pytest.mark.generate_only, pytest.mark.requires_api_key, pytest.mark.requires_redis]


async def test_generate_cache_key_is_stable(client, assert_mode):
    """
    This is intentionally mild + deterministic:
    - sends the same prompt twice
    - does NOT assume you expose hit/miss in JSON
    - only asserts both calls succeed
    Higher-signal cache assertions should be made via metrics (separate test).
    """
    assert assert_mode is True
    prompt = load_prompt("generate_cache_key_stable.txt")

    for _ in range(2):
        r = await client.post(
            "/v1/generate",
            json={"prompt": prompt, "max_new_tokens": 16, "temperature": 0.0},
        )
        r.raise_for_status()
        data = r.json()
        assert isinstance(data, dict)
        assert "text" in data or "choices" in data