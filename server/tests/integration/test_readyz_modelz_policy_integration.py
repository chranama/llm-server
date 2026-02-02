# tests/integration/test_readyz_modelz_policy_integration.py
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_readyz_off_mode_does_not_require_model_when_policy_disabled(monkeypatch, app_client):
    monkeypatch.setenv("MODEL_LOAD_MODE", "off")
    monkeypatch.setenv("REQUIRE_MODEL_READY", "0")

    async with (await app_client()) as c:
        r = await c.get("/readyz")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ready"
        assert body["model_load_mode"] == "off"
        assert body["require_model_ready"] is False


@pytest.mark.anyio
async def test_readyz_off_mode_requires_model_when_policy_enabled(monkeypatch, app_client):
    monkeypatch.setenv("MODEL_LOAD_MODE", "off")
    monkeypatch.setenv("REQUIRE_MODEL_READY", "1")

    async with (await app_client()) as c:
        r = await c.get("/readyz")
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["status"] == "not ready"
        assert body["require_model_ready"] is True
        assert body["model_loaded"] is False


@pytest.mark.anyio
async def test_modelz_reports_not_ready_when_off_and_not_loaded(monkeypatch, app_client):
    monkeypatch.setenv("MODEL_LOAD_MODE", "off")

    async with (await app_client()) as c:
        r = await c.get("/modelz")
        assert r.status_code == 503, r.text
        body = r.json()
        assert body["status"] == "not ready"
        assert body["model_load_mode"] == "off"
        assert body["model_loaded"] is False