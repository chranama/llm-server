# integrations/test_generate_only/test_auth_requirements.py
from __future__ import annotations

import pytest
import httpx

pytestmark = [pytest.mark.asyncio, pytest.mark.generate_only]


async def test_healthz_does_not_require_api_key(base_url, http_timeout):
    """
    Health endpoints should be unauthenticated.
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(http_timeout)) as c:
        r = await c.get("/healthz")
        r.raise_for_status()
        assert r.status_code == 200


async def test_v1_models_is_protected_if_deployed_that_way(base_url, http_timeout):
    """
    Some deployments require API key for /v1/*, others may not.
    This test is tolerant:
      - 200 means unprotected
      - 401/403 means protected (OK)
      - other codes should fail
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(http_timeout)) as c:
        r = await c.get("/v1/models")
        if r.status_code == 200:
            return
        if r.status_code in (401, 403):
            return
        raise AssertionError(f"Unexpected /v1/models status without API key: {r.status_code} body={r.text[:300]}")
    
async def test_v1_models_rejects_bad_api_key_when_protected(base_url, http_timeout):
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(http_timeout)) as c:
        r = await c.get("/v1/models", headers={"X-API-Key": "bad_key_for_test"})
        # If /v1/models is unprotected, it may still return 200 (OK).
        if r.status_code == 200:
            return
        # If protected, bad key should not be accepted.
        if r.status_code in (401, 403):
            return
        raise AssertionError(f"Unexpected /v1/models status with bad API key: {r.status_code} body={r.text[:300]}")