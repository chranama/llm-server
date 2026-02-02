from __future__ import annotations

import pytest
from sqlalchemy import select, func

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_outputs():
    # First request should call the model -> "FIRST"
    # Second request should hit DB cache -> still "FIRST"
    return ["FIRST", "SECOND"]


@pytest.mark.anyio
async def test_generate_db_cache_hit(client, auth_headers, test_sessionmaker):
    prompt = "hello cache"

    # 1) First call => not cached
    r1 = await client.post(
        "/v1/generate",
        headers=auth_headers,
        json={"prompt": prompt, "cache": True, "temperature": 0.0},
    )
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["cached"] is False
    assert b1["output"] == "FIRST"

    # 2) Second call => cached (should NOT consume "SECOND")
    r2 = await client.post(
        "/v1/generate",
        headers=auth_headers,
        json={"prompt": prompt, "cache": True, "temperature": 0.0},
    )
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["cached"] is True
    assert b2["output"] == "FIRST"

    # 3) Ensure cache rows exist
    from llm_server.db.models import CompletionCache
    async with test_sessionmaker() as session:
        n = (await session.execute(select(func.count()).select_from(CompletionCache))).scalar_one()
        assert n >= 1