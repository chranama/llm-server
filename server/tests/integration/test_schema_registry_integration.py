import json
import pytest

pytestmark = pytest.mark.integration

@pytest.mark.anyio
async def test_schema_index(monkeypatch, tmp_path, client, auth_headers):
    schema = {
        "title": "T",
        "description": "D",
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }
    (tmp_path / "s.json").write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    # IMPORTANT: schema_registry caches across process
    import llm_server.core.schema_registry as reg
    reg._SCHEMA_CACHE.clear()

    r = await client.get("/v1/schemas", headers=auth_headers)
    assert r.status_code == 200, r.text

    body = r.json()
    assert isinstance(body, list)

    assert {"schema_id": "s", "title": "T", "description": "D"} in body