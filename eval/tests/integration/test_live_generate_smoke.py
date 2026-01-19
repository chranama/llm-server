from __future__ import annotations

import pytest

from llm_eval.client.http_client import GenerateErr, GenerateOk, HttpEvalClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_generate_smoke(live_client: HttpEvalClient):
    resp = await live_client.generate(
        prompt="Reply with exactly the word: OK",
        max_new_tokens=8,
        temperature=0.0,
        cache=False,
    )

    assert resp.latency_ms >= 0.0
    assert isinstance(resp, (GenerateOk, GenerateErr))

    if isinstance(resp, GenerateErr):
        # typed error, not exception
        assert resp.status_code >= 0
        assert resp.error_code
        return

    assert isinstance(resp, GenerateOk)
    assert isinstance(resp.model, str) and resp.model
    assert isinstance(resp.output_text, str)