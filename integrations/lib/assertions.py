from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from integrations.lib.fixtures import GoldenFixture, evaluate_contract
from integrations.markers import ModelsSnapshot, assert_full, assert_generate_only


# -------------------------
# Capability assertions
# -------------------------


def assert_models_generate_only(snapshot: ModelsSnapshot) -> None:
    """Enforce generate-only deployment contract (based on /v1/models)."""
    assert_generate_only(snapshot)


def assert_models_full(snapshot: ModelsSnapshot) -> None:
    """Enforce full-capabilities deployment contract (based on /v1/models)."""
    assert_full(snapshot)


# -------------------------
# Extraction assertions
# -------------------------


def _coerce_extracted_object(resp_json: Any) -> Dict[str, Any]:
    """
    Normalize extract responses into the extracted object.

    Supported shapes:
      1) {"output": {...}, ...}
      2) {"extracted": {...}, ...}
      3) {...} (already the object)
    """
    if not isinstance(resp_json, dict):
        raise AssertionError("extract response must be a JSON object")

    for key in ("output", "extracted", "data", "result"):
        v = resp_json.get(key)
        if isinstance(v, dict):
            return v

    # If it "looks like" the extracted object already, accept it.
    return resp_json


async def assert_extract_matches_golden(
    *,
    client: httpx.AsyncClient,
    fixture: GoldenFixture,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call /v1/extract and assert output satisfies the golden contract.

    Returns the extracted object (normalized).
    """
    payload: Dict[str, Any] = {
        "schema_id": fixture.schema_id,
        "text": fixture.text,
    }
    if extra_payload:
        payload.update(extra_payload)

    r = await client.post("/v1/extract", json=payload)
    r.raise_for_status()

    raw = r.json()
    extracted = _coerce_extracted_object(raw)

    evaluate_contract(
        extracted=extracted,
        contract=fixture.contract,
        ctx=f"{fixture.kind}/{fixture.name}",
    )

    return extracted


# -------------------------
# Generate assertions
# -------------------------


async def assert_generate_smoke(
    *,
    client: httpx.AsyncClient,
    prompt: str,
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    """
    Minimal sanity check for /v1/generate using current backend contract:
      {"model": "...", "output": "...", "cached": bool}
    Returns the full JSON response.
    """
    r = await client.post(
        "/v1/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
        },
    )
    r.raise_for_status()

    data = r.json()
    assert isinstance(data, dict), "generate response must be JSON object"

    assert "model" in data, "generate response missing 'model'"
    assert "output" in data, "generate response missing 'output'"
    assert isinstance(data["output"], str), "generate output must be a string"
    assert "cached" in data and isinstance(data["cached"], bool), "generate response missing boolean 'cached'"

    return data