from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_outputs():
    # First batch call consumes two outputs for p1, p2
    # Second call should be fully cached => should NOT consume more outputs.
    return ["A1", "B1"]


@pytest.mark.anyio
async def test_generate_batch_caches_per_item(client, auth_headers):
    payload = {
        "prompts": ["p1", "p2"],
        "max_new_tokens": 16,
        "temperature": 0.0,
        "cache": True,
    }

    r1 = await client.post("/v1/generate/batch", headers=auth_headers, json=payload)
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert [x["output"] for x in b1["results"]] == ["A1", "B1"]
    assert [x["cached"] for x in b1["results"]] == [False, False]

    # second pass should come from DB cache (redis disabled)
    r2 = await client.post("/v1/generate/batch", headers=auth_headers, json=payload)
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert [x["output"] for x in b2["results"]] == ["A1", "B1"]
    assert [x["cached"] for x in b2["results"]] == [True, True]