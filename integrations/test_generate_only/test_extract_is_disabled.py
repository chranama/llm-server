# integrations/test_generate_only/test_extract_is_disabled.py
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.generate_only, pytest.mark.requires_api_key]


async def test_extract_endpoint_is_disabled(client, assert_mode):
    """
    In generate-only mode, /v1/extract must not succeed.
    Acceptable outcomes: 404 (route absent), 403 (auth), 409/422 (capability gate), etc.
    """
    assert assert_mode is True

    payload = {"schema_id": "sroie_receipt_v1", "text": "probe"}
    r = await client.post("/v1/extract", json=payload)

    assert r.status_code != 200, f"expected /v1/extract disabled, got 200: {r.text[:400]}"