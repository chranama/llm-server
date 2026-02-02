from __future__ import annotations

import json
import pytest

pytestmark = pytest.mark.integration


def _write_schema(tmp_path, schema_id: str, schema: dict) -> None:
    (tmp_path / f"{schema_id}.json").write_text(json.dumps(schema), encoding="utf-8")


@pytest.fixture
def llm_outputs():
    # Default outputs for this module; tests can override per-test by redefining
    # llm_outputs inside a specific test via monkeypatch, but simplest is to keep
    # module-level and craft tests around it.
    return [
        "not json",  # for test_extract_repair_success: fail parse -> triggers repair
        '<<<JSON>>>{"id":"repaired"}<<<END>>>',  # repair succeeds
        "not json",  # for test_extract_repair_disabled_returns_422: fail parse -> no repair
    ]


@pytest.mark.anyio
async def test_extract_repair_success(monkeypatch, tmp_path, client, auth_headers):
    """
    First model output invalid => repair triggered => second output valid.
    """
    schema_id = "ticket_v1"
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    _write_schema(tmp_path, schema_id, schema)
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    # Ensure SCHEMAS_DIR change takes effect
    import llm_server.core.schema_registry as reg
    reg._SCHEMA_CACHE.clear()

    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": schema_id, "text": "ticket id repaired", "cache": False, "repair": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_id"] == schema_id
    assert body["repair_attempted"] is True
    assert body["data"] == {"id": "repaired"}


@pytest.mark.anyio
async def test_extract_repair_disabled_returns_422(monkeypatch, tmp_path, client, auth_headers):
    """
    Invalid model output + repair=False => 422 AppError envelope.
    """
    schema_id = "ticket_v1"
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    _write_schema(tmp_path, schema_id, schema)
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    import llm_server.core.schema_registry as reg
    reg._SCHEMA_CACHE.clear()

    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": schema_id, "text": "x", "cache": False, "repair": False},
    )
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["code"] in ("invalid_json", "schema_validation_failed")
    assert "message" in body