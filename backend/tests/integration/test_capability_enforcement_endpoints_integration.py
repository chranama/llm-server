# backend/tests/integration/test_capability_enforcement_endpoints_integration.py
import types
import pytest
from fastapi import FastAPI
import httpx

from llm_server.api import extract as ext_api
from llm_server.api import deps
from llm_server.core import errors


class FakeApiKey:
    def __init__(self, key="k"):
        self.key = key


class FakeModel:
    def generate(self, **kwargs):
        return "OK"


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None


@pytest.mark.anyio
async def test_deployment_disable_extract_returns_501(monkeypatch):
    app = FastAPI()
    errors.setup(app)  # ✅ install canonical error envelope

    app.state.settings = types.SimpleNamespace(
        env="test",
        model_load_mode="lazy",
        enable_generate=True,
        enable_extract=False,  # deployment disabled
        model_id="m1",
        allowed_models=["m1"],
        all_model_ids=["m1"],
    )
    app.state.model_load_mode = "lazy"
    app.state.model_error = None

    app.include_router(ext_api.router)

    app.dependency_overrides[deps.get_api_key] = lambda: FakeApiKey()
    app.dependency_overrides[deps.get_llm] = lambda: {"m1": FakeModel()}  # ✅ no-arg

    monkeypatch.setattr(ext_api.db_session, "get_sessionmaker", lambda: (lambda: DummySession()))

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/extract", json={"schema_id": "ticket_v1", "text": "x"})

    assert r.status_code == 501
    body = r.json()
    assert body["code"] == "capability_disabled"


@pytest.mark.anyio
async def test_model_lacks_extract_returns_400(monkeypatch):
    app = FastAPI()
    errors.setup(app)  # ✅ install canonical error envelope

    app.state.settings = types.SimpleNamespace(
        env="test",
        model_load_mode="lazy",
        enable_generate=True,
        enable_extract=True,  # deployment on
        model_id="m1",
        allowed_models=["m1"],
        all_model_ids=["m1"],
    )
    app.state.model_load_mode = "lazy"
    app.state.model_error = None

    app.include_router(ext_api.router)

    app.dependency_overrides[deps.get_api_key] = lambda: FakeApiKey()
    app.dependency_overrides[deps.get_llm] = lambda: {"m1": FakeModel()}  # ✅ no-arg

    monkeypatch.setattr(deps, "model_capabilities", lambda model_id, request=None: {"extract": False, "generate": True})
    monkeypatch.setattr(ext_api.db_session, "get_sessionmaker", lambda: (lambda: DummySession()))

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/extract", json={"schema_id": "ticket_v1", "text": "x"})

    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "capability_not_supported"