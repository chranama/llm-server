# backend/tests/integration/test_extract_model_autoselection_integration.py
import json
import types

import pytest
from fastapi import FastAPI
import httpx

from llm_server.api import generate as gen_api
from llm_server.api import extract as ext_api
from llm_server.api import deps
from llm_server.core import errors
from llm_server.services.llm_registry import MultiModelManager  # ✅ add


class FakeApiKey:
    def __init__(self, key="k"):
        self.key = key


class FakeModel:
    def __init__(self, output: str):
        self._out = output

    def generate(self, **kwargs):
        return self._out


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None


@pytest.mark.anyio
async def test_extract_uses_default_for_capability_generate_uses_default(monkeypatch):
    app = FastAPI()
    errors.setup(app)

    app.state.settings = types.SimpleNamespace(
        env="test",
        model_load_mode="lazy",
        enable_generate=True,
        enable_extract=True,
        model_id="default_gen",
        allowed_models=["default_gen", "extractor"],
        all_model_ids=["default_gen", "extractor"],
    )
    app.state.model_load_mode = "lazy"
    app.state.model_error = None

    app.include_router(gen_api.router)
    app.include_router(ext_api.router)

    app.dependency_overrides[deps.get_api_key] = lambda: FakeApiKey()

    mm = MultiModelManager(
        models={
            "default_gen": FakeModel("GEN_OK"),
            "extractor": FakeModel('{"ok": true}'),
        },
        default_id="default_gen",
        model_meta={
            "default_gen": {"capabilities": ["generate"]},
            "extractor": {"capabilities": ["extract"]},
        },
    )
    app.dependency_overrides[deps.get_llm] = lambda: mm  # ✅

    monkeypatch.setenv("TOKEN_COUNTING", "0")

    # Patch generate DB/cache/log calls
    monkeypatch.setattr(gen_api.db_session, "get_sessionmaker", lambda: (lambda: DummySession()))

    async def _gen_get_cached(*a, **k):
        return None, False, None

    async def _gen_write_cache(*a, **k):
        return None

    async def _gen_log(*a, **k):
        return None

    monkeypatch.setattr(gen_api, "get_cached_output", _gen_get_cached)
    monkeypatch.setattr(gen_api, "write_cache", _gen_write_cache)
    monkeypatch.setattr(gen_api, "write_inference_log", _gen_log)
    monkeypatch.setattr(gen_api, "record_token_metrics", lambda *a, **k: None)

    # Patch extract DB/cache/log + schema/validation
    monkeypatch.setattr(ext_api.db_session, "get_sessionmaker", lambda: (lambda: DummySession()))

    async def _ext_get_cached(*a, **k):
        return None, False, None

    async def _ext_write_cache(*a, **k):
        return None

    async def _ext_log(*a, **k):
        return None

    monkeypatch.setattr(ext_api, "get_cached_output", _ext_get_cached)
    monkeypatch.setattr(ext_api, "write_cache", _ext_write_cache)
    monkeypatch.setattr(ext_api, "write_inference_log", _ext_log)

    monkeypatch.setattr(ext_api, "load_schema", lambda sid: {"type": "object"})
    monkeypatch.setattr(ext_api, "validate_jsonschema", lambda schema, obj: None)
    monkeypatch.setattr(ext_api, "_validate_first_matching", lambda schema, raw: json.loads(raw))

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r1 = await ac.post("/v1/generate", json={"prompt": "hi", "cache": False})
        r2 = await ac.post(
            "/v1/extract",
            json={"schema_id": "ticket_v1", "text": "x", "cache": False, "repair": False},
        )

    assert r1.status_code == 200
    assert r1.json()["model"] == "default_gen"

    assert r2.status_code == 200
    assert r2.json()["model"] == "extractor"