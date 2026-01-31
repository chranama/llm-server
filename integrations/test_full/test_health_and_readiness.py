# integrations/test_full/test_health_and_readiness.py
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.full]


async def test_healthz_is_reachable(client):
    r = await client.get("/healthz")
    r.raise_for_status()
    assert r.status_code == 200


@pytest.mark.requires_api_key
async def test_readyz_if_present(client):
    r = await client.get("/readyz")
    if r.status_code == 404:
        pytest.skip("/readyz not exposed in this deployment")
    r.raise_for_status()
    assert r.status_code == 200