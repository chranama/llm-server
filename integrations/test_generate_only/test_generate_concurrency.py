# integrations/test_generate_only/test_generate_concurrency.py
from __future__ import annotations

import pytest

from integrations.lib.concurrency import run_concurrent
from integrations.lib.fixtures import load_prompt

pytestmark = [pytest.mark.asyncio, pytest.mark.generate_only, pytest.mark.requires_api_key]


async def test_generate_handles_concurrent_requests(client, assert_mode):
    """
    Light concurrency probe: enough to catch obvious connection pooling / deadlock issues,
    but not a load test.
    """
    assert assert_mode is True

    prompt = load_prompt("generate_ping.txt")

    async def one() -> None:
        r = await client.post("/v1/generate", json={"prompt": prompt, "max_new_tokens": 8, "temperature": 0.2})
        r.raise_for_status()

    # defaults inside run_concurrent should be conservative
    await run_concurrent(one, n=8)