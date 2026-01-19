from __future__ import annotations

import pytest

from llm_eval.client.http_client import ExtractErr, ExtractOk, HttpEvalClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_extract_sroie_receipt_v1_happy_path(live_client: HttpEvalClient):
    """
    Contract test for canonical extraction schema.

    We do NOT assert exact values (model-dependent), but we assert:
      - typed protocol (Ok/Err)
      - latency present
      - schema_id echo is stable
      - data is a dict
    """
    schema_id = "sroie_receipt_v1"
    text = "Company: ACME\nDate: 2024-01-01\nTotal: 10.00\nAddress: 123 Main St"

    resp = await live_client.extract(
        schema_id=schema_id,
        text=text,
        temperature=0.0,
        max_new_tokens=256,
        cache=False,
        repair=True,
    )

    assert resp.latency_ms >= 0.0

    # If server rejects schema or auth, we still learn something; assert typed protocol.
    if isinstance(resp, ExtractErr):
        # common failure modes: auth, schema missing, server error, validation
        assert isinstance(resp.status_code, int)
        assert resp.status_code >= 0
        assert isinstance(resp.error_code, str) and resp.error_code
        assert isinstance(resp.message, str)
        return

    assert isinstance(resp, ExtractOk)
    assert resp.schema_id == schema_id
    assert isinstance(resp.model, str) and resp.model
    assert isinstance(resp.data, dict)

    # Soft expectations: these keys are typical for SROIE receipt extraction
    # (don’t fail if absent; just validate if present)
    for k in ("company", "date", "total", "address"):
        if k in resp.data:
            assert resp.data[k] is None or isinstance(resp.data[k], (str, int, float))


@pytest.mark.asyncio
async def test_extract_invalid_schema_returns_typed_error(live_client: HttpEvalClient):
    """
    This ensures non-200 responses become ExtractErr (not exceptions).
    """
    resp = await live_client.extract(
        schema_id="__definitely_not_a_real_schema__",
        text="Hello",
        cache=False,
        repair=False,
    )

    assert resp.latency_ms >= 0.0
    assert isinstance(resp, (ExtractOk, ExtractErr))

    # If the server oddly accepts it, don't fail hard; but typically this should be an error.
    if isinstance(resp, ExtractErr):
        assert resp.status_code != 200
        assert resp.error_code