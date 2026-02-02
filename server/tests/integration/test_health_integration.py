import pytest

pytestmark = pytest.mark.integration

@pytest.mark.anyio
async def test_healthz(client):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

@pytest.mark.anyio
async def test_readyz(client):
    r = await client.get("/readyz")
    assert r.status_code in (200, 503)
    body = r.json()
    assert "db" in body
    assert "redis" in body