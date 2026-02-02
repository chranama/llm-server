import pytest
from sqlalchemy import select

pytestmark = pytest.mark.integration

@pytest.mark.anyio
async def test_quota_consumed_on_failure(client, test_sessionmaker):
    from llm_server.db.models import ApiKey

    key="qtest"
    async with test_sessionmaker() as s:
        s.add(ApiKey(key=key, active=True, quota_monthly=1, quota_used=0))
        await s.commit()

    r = await client.post("/v1/extract", headers={"X-API-Key":key},
        json={"schema_id":"missing","text":"x","cache":False,"repair":False})

    assert r.status_code == 404

    async with test_sessionmaker() as s:
        row = (await s.execute(select(ApiKey).where(ApiKey.key==key))).scalar_one()
        assert row.quota_used == 1