# tests/integration/test_model_load_mode_integration.py
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_model_load_mode_off(monkeypatch, app_client, auth_headers):
    monkeypatch.setenv("MODEL_LOAD_MODE", "off")

    async with (await app_client()) as c:
        # Make sure server believes model is not loaded
        c.app.state.llm = None  # httpx exposes the underlying ASGI app on the client

        r = await c.post("/v1/generate", headers=auth_headers, json={"prompt": "hi"})
        assert r.status_code == 503, r.text