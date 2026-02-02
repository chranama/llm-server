# tests/integration/test_schemas_detail_integration.py
from __future__ import annotations

import json
import pytest

pytestmark = pytest.mark.integration


def _write_schema(tmp_path, schema_id: str, schema: dict) -> None:
    (tmp_path / f"{schema_id}.json").write_text(json.dumps(schema), encoding="utf-8")


@pytest.mark.anyio
async def test_schema_detail_returns_full_schema(monkeypatch, tmp_path, auth_headers, client):
    schema_id = "ticket_v1"
    schema = {
        "title": "Ticket v1",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    _write_schema(tmp_path, schema_id, schema)
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    import llm_server.core.schema_registry as reg
    reg._SCHEMA_CACHE.clear()

    r = await client.get(f"/v1/schemas/{schema_id}", headers=auth_headers)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["type"] == "object"
    assert payload["properties"]["id"]["type"] == "string"


@pytest.mark.anyio
async def test_schema_detail_missing_is_404(monkeypatch, tmp_path, auth_headers, app_client):
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    import llm_server.core.schema_registry as reg
    reg._SCHEMA_CACHE.clear()

    async with (await app_client()) as client:
        r = await client.get("/v1/schemas/does_not_exist", headers=auth_headers)
        assert r.status_code == 404, r.text