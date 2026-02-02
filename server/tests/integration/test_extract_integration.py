from __future__ import annotations

import json
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_outputs():
    # One-shot valid output (no repair needed)
    return ['<<<JSON>>>{"id":"1"}<<<END>>>']


@pytest.mark.anyio
async def test_extract_success(monkeypatch, tmp_path, client, auth_headers):
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": False,
    }
    (tmp_path / "a.json").write_text(json.dumps(schema), encoding="utf-8")
    monkeypatch.setenv("SCHEMAS_DIR", str(tmp_path))

    # IMPORTANT: schema_registry caches across process
    import llm_server.core.schema_registry as reg
    reg._SCHEMA_CACHE.clear()

    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "a", "text": "id 1", "cache": False, "repair": True},
    )

    assert r.status_code == 200, r.text
    assert r.json()["data"]["id"] == "1"