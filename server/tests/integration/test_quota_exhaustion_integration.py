# tests/integration/test_quota_exhaustion_integration.py
from __future__ import annotations

import uuid
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_quota_exhaustion_blocks_subsequent_requests(test_sessionmaker, client):
    from llm_server.db.models import ApiKey

    key = f"quota_{uuid.uuid4().hex}"
    async with test_sessionmaker() as session:
        session.add(ApiKey(key=key, active=True, quota_monthly=1, quota_used=0))
        await session.commit()

    headers = {"X-API-Key": key}

    r1 = await client.post("/v1/generate", headers=headers, json={"prompt": "hi", "cache": False})
    assert r1.status_code == 200, r1.text

    r2 = await client.post("/v1/generate", headers=headers, json={"prompt": "hi again", "cache": False})
    assert r2.status_code == 402, r2.text
    body = r2.json()
    assert body["code"] == "quota_exhausted"