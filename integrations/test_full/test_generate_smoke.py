# integrations/test_full/test_generate_smoke.py
from __future__ import annotations

import pytest

from integrations.lib.assertions import assert_generate_smoke
from integrations.lib.fixtures import load_prompt

pytestmark = [pytest.mark.asyncio, pytest.mark.full, pytest.mark.requires_api_key]


async def test_generate_returns_output(client, assert_mode):
    assert assert_mode is True
    prompt = load_prompt("generate_ping.txt")
    await assert_generate_smoke(client=client, prompt=prompt, max_tokens=16)