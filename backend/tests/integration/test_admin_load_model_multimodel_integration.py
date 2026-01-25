# backend/tests/integration/test_admin_load_model_multimodel_integration.py
from __future__ import annotations

import types

import httpx
import pytest
from fastapi import FastAPI

from llm_server.api import admin as admin_api
from llm_server.api import deps as deps_api


class FakeRole:
    def __init__(self, name: str = "admin"):
        self.name = name


class FakeApiKey:
    def __init__(self, key: str = "k", role_id=None):
        self.key = key
        self.role_id = role_id
        self.id = 1
        self.role = FakeRole("admin")


class DummySession:
    async def execute(self, stmt):
        class _Res:
            def scalar_one_or_none(self_inner):
                # emulate ApiKey with role loaded
                return FakeApiKey()

        return _Res()


class DummyBackend:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.ensure_loaded_calls = 0

    def ensure_loaded(self):
        self.ensure_loaded_calls += 1


class FakeMultiModelManager:
    """
    Intentionally NOT a MultiModelManager instance; admin.py should duck-type.
    """
    def __init__(self):
        self.models = {"m1": DummyBackend("m1"), "m2": DummyBackend("m2")}
        self.default_id = "m1"

    def ensure_loaded(self):
        # registry semantics: ensure_loaded should load default only
        self.models[self.default_id].ensure_loaded()


@pytest.mark.anyio
async def test_admin_models_load_multimodel_from_off(monkeypatch):
    app = FastAPI()
    app.include_router(admin_api.router)

    # app state simulating MODEL_LOAD_MODE=off boot
    app.state.settings = types.SimpleNamespace(
        env="test",
        model_load_mode="off",
        model_id="m1",
        all_model_ids=["m1", "m2"],
    )
    app.state.model_load_mode = "off"
    app.state.llm = None
    app.state.model_loaded = False
    app.state.model_error = None

    # Override auth + session deps
    app.dependency_overrides[deps_api.get_api_key] = lambda: FakeApiKey()
    app.dependency_overrides[admin_api.get_session] = lambda: DummySession()

    # Ensure _ensure_admin passes without real DB role loads
    async def _ensure_admin(api_key, session):
        return None

    monkeypatch.setattr(admin_api, "_ensure_admin", _ensure_admin)

    # IMPORTANT: patch the symbol used inside admin.py (NOT llm_server.services.llm)
    monkeypatch.setattr(admin_api, "build_llm_from_settings", lambda: FakeMultiModelManager(), raising=True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/admin/models/load", json={})

    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert payload["default_model"] == "m1"
    assert set(payload["models"]) == {"m1", "m2"}

    # model_loaded should reflect ensure_loaded was called successfully
    assert app.state.model_loaded is True
    assert isinstance(app.state.llm, FakeMultiModelManager)
    assert app.state.llm.models["m1"].ensure_loaded_calls == 1
    assert app.state.llm.models["m2"].ensure_loaded_calls == 0