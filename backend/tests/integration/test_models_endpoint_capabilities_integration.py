# backend/tests/integration/test_models_endpoint_capabilities_integration.py
import types
import pytest
from fastapi import FastAPI
import httpx

from llm_server.api import models as models_api
from llm_server.api import deps as deps_api


@pytest.mark.anyio
async def test_models_endpoint_deployment_and_model_caps_reflected(monkeypatch):
    app = FastAPI()
    app.include_router(models_api.router)

    # settings snapshot
    app.state.settings = types.SimpleNamespace(
        env="test",
        model_load_mode="off",
        model_id="modelA",
        all_model_ids=["modelA", "modelB"],
        enable_generate=True,
        enable_extract=False,  # deployment gate off
    )
    app.state.model_load_mode = "off"

    # Patch models.yaml caps:
    # modelA extract False, modelB unspecified => defaults True but then gated by deployment
    monkeypatch.setattr(
        deps_api,
        "_model_capabilities_from_models_yaml",
        lambda mid: {"extract": False} if mid == "modelA" else None,
        raising=True,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/v1/models")

    assert r.status_code == 200
    payload = r.json()

    assert payload["deployment_capabilities"]["extract"] is False

    by_id = {m["id"]: m for m in payload["models"]}
    assert by_id["modelA"]["capabilities"]["extract"] is False
    # modelB would otherwise allow extract, but deployment gate forces False
    assert by_id["modelB"]["capabilities"]["extract"] is False