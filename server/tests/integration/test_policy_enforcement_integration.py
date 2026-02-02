# tests/integration/test_policy_enforcement_integration.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write(p: Path, obj) -> None:
    p.write_text(json.dumps(obj), encoding="utf-8")


@pytest.mark.anyio
async def test_policy_disables_extract_blocks_endpoint(client, auth_headers, monkeypatch, tmp_path: Path):
    p = tmp_path / "policy.json"
    _write(p, {"enable_extract": False})
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    # Capability is enforced before schema load, so schema_id can be anything.
    payload = {"schema_id": "does_not_matter", "text": "hello"}
    r = await client.post("/v1/extract", json=payload, headers=auth_headers)
    assert r.status_code == 400

    body = r.json()
    assert body.get("code") == "capability_not_supported"
    # sanity check the merged caps include policy denial
    extra = body.get("extra") or {}
    caps = extra.get("model_capabilities") or {}
    assert caps.get("extract") is False


@pytest.mark.anyio
async def test_policy_disables_extract_reflected_in_models_endpoint(client, auth_headers, monkeypatch, tmp_path: Path):
    p = tmp_path / "policy.json"
    _write(p, {"enable_extract": False})
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    r = await client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data and len(data["models"]) >= 1

    # Default model should reflect extract=False
    default_id = data["default_model"]
    m = next(x for x in data["models"] if x["id"] == default_id)
    assert m["capabilities"]["extract"] is False


@pytest.mark.anyio
async def test_policy_invalid_file_fail_closed_blocks_extract(client, auth_headers, monkeypatch, tmp_path: Path):
    p = tmp_path / "policy.json"
    p.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    payload = {"schema_id": "whatever", "text": "hello"}
    r = await client.post("/v1/extract", json=payload, headers=auth_headers)
    assert r.status_code == 400
    assert r.json().get("code") == "capability_not_supported"