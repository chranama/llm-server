# integrations/test_full/test_extract_repair_behavior.py
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.full, pytest.mark.requires_api_key]


async def test_extract_rejects_missing_fields(client, assert_mode):
    """
    Full mode should enforce request validation.
    We don't enforce a specific status code (FastAPI commonly uses 422),
    but it must not succeed.
    """
    assert assert_mode is True

    # Missing schema_id and/or text should fail validation
    r = await client.post("/v1/extract", json={"text": "hello"})
    assert r.status_code != 200


async def test_extract_repair_flag_if_supported(client, assert_mode):
    """
    If your /v1/extract supports repair toggles, this is where you assert it.
    If not supported (422/400 for unknown field), skip.
    """
    assert assert_mode is True

    payload = {"schema_id": "sroie_receipt_v1", "text": "TOTAL $7.60", "repair": True}
    r = await client.post("/v1/extract", json=payload)

    if r.status_code in (400, 422) and ("repair" in r.text.lower() or "extra fields" in r.text.lower()):
        pytest.skip("repair flag not supported by this deployment")
    r.raise_for_status()
    data = r.json()
    assert isinstance(data, dict)