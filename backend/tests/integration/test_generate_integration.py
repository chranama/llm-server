# backend/tests/integration/test_generate_integration.py
from __future__ import annotations

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.integration


class _DummyLLM:
    def __init__(self, text: str = "ok", model_id: str = "test-model"):
        self._text = text
        self.model_id = model_id

    def generate(self, prompt: str, **kwargs) -> str:
        return self._text


@pytest.fixture(autouse=True)
def _integration_env(monkeypatch: pytest.MonkeyPatch):
    # Never allow real model load
    monkeypatch.setenv("MODEL_LOAD_MODE", "off")
    monkeypatch.setenv("MODEL_WARMUP", "0")


@pytest.fixture(autouse=True)
def _override_llm(app):  # <-- IMPORTANT: use the app fixture instance
    from llm_server.api.deps import get_llm  # <-- correct dependency

    app.dependency_overrides[get_llm] = lambda: _DummyLLM("ok", "test-model")
    yield
    app.dependency_overrides.pop(get_llm, None)

@pytest.fixture(autouse=True)
def _override_settings(monkeypatch: pytest.MonkeyPatch):
    """
    Default integration mode = generate-only.
    Individual tests can override if needed.
    """
    from llm_server.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "enable_generate", True, raising=False)
    monkeypatch.setattr(s, "enable_extract", False, raising=False)
    yield


@pytest.mark.anyio
async def test_models_reflect_generate_only(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()

    # deployment capability switches now live here
    assert body["deployment_capabilities"]["generate"] is True
    assert body["deployment_capabilities"]["extract"] is False

    # sanity: should have at least the default model
    assert "default_model" in body and body["default_model"]

    # In generate-only integration mode, the default model should have extract=False effectively
    default_id = body["default_model"]
    models = {m["id"]: m for m in body["models"]}
    assert default_id in models

    caps = models[default_id].get("capabilities") or {}
    assert caps.get("generate") is True
    assert caps.get("extract") is False


@pytest.mark.anyio
async def test_generate_works(client, auth_headers):
    r = await client.post("/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False})
    assert r.status_code == 200
    assert str(r.json()["output"]).strip().lower() == "ok"


@pytest.mark.anyio
async def test_extract_is_disabled(client, auth_headers):
    r = await client.post(
        "/v1/extract",
        headers=auth_headers,
        json={"schema_id": "ticket_v1", "text": "hello"},
    )

    assert r.status_code == 501
    body = r.json()
    assert body["code"] == "capability_disabled"
    assert body["extra"]["capability"] == "extract"


@pytest.mark.anyio
async def test_generate_log_written(client, auth_headers, test_sessionmaker):
    await client.post("/v1/generate", headers=auth_headers, json={"prompt": "hi", "cache": False})

    from llm_server.db.models import InferenceLog

    async with test_sessionmaker() as session:
        rows = (await session.execute(select(InferenceLog))).scalars().all()
        assert len(rows) >= 1