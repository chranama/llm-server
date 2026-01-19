# tests/unit/client/test_http_client.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

import llm_eval.client.http_client as hc


class _FakeAsyncClient:
    """
    Drop-in stand-in for httpx.AsyncClient used inside HttpEvalClient.

    - Acts as an async context manager
    - Serves pre-baked responses from a queue
    - Records the most recent request for assertions
    """

    def __init__(self, responses: List[httpx.Response], timeout: float):
        self._responses = list(responses)
        self.timeout = timeout

        # capture last request
        self.last_url: Optional[str] = None
        self.last_json: Optional[Dict[str, Any]] = None
        self.last_headers: Optional[Dict[str, str]] = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        if not self._responses:
            raise AssertionError("No more fake responses configured for _FakeAsyncClient")
        return self._responses.pop(0)


def _make_json_response(status_code: int, payload: Any) -> httpx.Response:
    # httpx.Response needs bytes content; json=... will set content + headers.
    return httpx.Response(status_code=status_code, json=payload)


def _make_text_response(status_code: int, text: str) -> httpx.Response:
    return httpx.Response(status_code=status_code, content=text.encode("utf-8"))


def _patch_httpx_async_client(monkeypatch, responses: List[httpx.Response]) -> _FakeAsyncClient:
    created: dict[str, Any] = {"client": None}

    def _factory(*, timeout: float) -> _FakeAsyncClient:
        c = _FakeAsyncClient(responses=responses, timeout=timeout)
        created["client"] = c
        return c

    # Patch the module-local httpx symbol used by HttpEvalClient
    monkeypatch.setattr(hc.httpx, "AsyncClient", _factory)
    assert created["client"] is None
    # return the created client after first use (test will access via closure)
    # We'll expose it by returning a placeholder; tests should read it after call.
    return created  # type: ignore[return-value]


class _TimeSeq:
    def __init__(self, seq: List[float]):
        self.seq = list(seq)

    def __call__(self) -> float:
        if not self.seq:
            raise AssertionError("time.time() called more times than expected in test")
        return self.seq.pop(0)


@pytest.mark.asyncio
async def test_generate_ok_extracts_output_and_model_and_cached(monkeypatch):
    responses = [
        _make_json_response(
            200,
            {
                "output": " hello ",
                "model": "m-test",
                "cached": True,
            },
        )
    ]
    created = _patch_httpx_async_client(monkeypatch, responses)

    # deterministic latency: (0.2 - 0.1) * 1000 = 100ms
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.1, 0.2]))

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY", timeout=12.0)
    out = await client.generate(prompt="p", max_new_tokens=7, temperature=0.0, model="m-override", cache=True)

    assert isinstance(out, hc.GenerateOk)
    assert out.output_text == "hello"
    assert out.model == "m-test"  # server wins
    assert out.cached is True
    assert out.latency_ms == pytest.approx(100.0)

    fake_client: _FakeAsyncClient = created["client"]
    assert fake_client is not None
    assert fake_client.timeout == 12.0
    assert fake_client.last_url == "http://svc/v1/generate"
    assert fake_client.last_headers == {"X-API-Key": "KEY"}
    assert fake_client.last_json == {
        "prompt": "p",
        "max_new_tokens": 7,
        "temperature": 0.0,
        "model": "m-override",
        "cache": True,
    }


@pytest.mark.asyncio
async def test_generate_ok_tolerates_nested_text_shape(monkeypatch):
    responses = [
        _make_json_response(
            200,
            {
                "data": {"text": "hi"},
                "model_id": "m0",
            },
        )
    ]
    _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([1.0, 1.0]))  # 0ms

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await client.generate(prompt="p")

    assert isinstance(out, hc.GenerateOk)
    assert out.output_text == "hi"
    assert out.model == "m0"


@pytest.mark.asyncio
async def test_generate_err_standard_json_shape(monkeypatch):
    responses = [
        _make_json_response(
            503,
            {"code": "upstream_unavailable", "message": "nope", "extra": {"stage": "proxy"}},
        )
    ]
    _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.0, 0.01]))  # 10ms

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await client.generate(prompt="p")

    assert isinstance(out, hc.GenerateErr)
    assert out.status_code == 503
    assert out.error_code == "upstream_unavailable"
    assert out.message == "nope"
    assert out.extra == {"stage": "proxy"}
    assert out.latency_ms == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_generate_err_non_json_body_falls_back_to_http_error(monkeypatch):
    responses = [_make_text_response(500, "boom")]
    _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.0, 0.0]))  # 0ms

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await client.generate(prompt="p")

    assert isinstance(out, hc.GenerateErr)
    assert out.status_code == 500
    assert out.error_code == "http_error"
    assert out.message == "boom"
    assert out.extra is None


@pytest.mark.asyncio
async def test_extract_ok_parses_fields(monkeypatch):
    responses = [
        _make_json_response(
            200,
            {
                "schema_id": "s1",
                "model": "m1",
                "data": {"a": 1},
                "cached": False,
                "repair_attempted": True,
            },
        )
    ]
    created = _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.3, 0.35]))  # 50ms

    client = hc.HttpEvalClient(base_url="http://svc/", api_key="KEY")
    out = await client.extract(schema_id="s1", text="doc", model="m-override", cache=True, repair=False)

    assert isinstance(out, hc.ExtractOk)
    assert out.schema_id == "s1"
    assert out.model == "m1"  # server wins
    assert out.data == {"a": 1}
    assert out.cached is False
    assert out.repair_attempted is True
    assert out.latency_ms == pytest.approx(50.0)

    fake_client: _FakeAsyncClient = created["client"]
    assert fake_client.last_url == "http://svc/v1/extract"  # base_url rstrip("/") honored
    assert fake_client.last_headers == {"X-API-Key": "KEY"}
    assert fake_client.last_json == {
        "schema_id": "s1",
        "text": "doc",
        "max_new_tokens": 512,
        "temperature": 0.0,
        "cache": True,
        "repair": False,
        "model": "m-override",
    }


@pytest.mark.asyncio
async def test_extract_err_standard_json_shape(monkeypatch):
    responses = [
        _make_json_response(
            422,
            {"code": "schema_validation_failed", "message": "bad", "extra": {"stage": "validate"}},
        )
    ]
    _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.0, 0.02]))  # 20ms

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await client.extract(schema_id="s1", text="doc")

    assert isinstance(out, hc.ExtractErr)
    assert out.status_code == 422
    assert out.error_code == "schema_validation_failed"
    assert out.message == "bad"
    assert out.extra == {"stage": "validate"}
    assert out.latency_ms == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_extract_err_non_json_body_falls_back_to_http_error(monkeypatch):
    responses = [_make_text_response(401, "no auth")]
    _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([5.0, 5.0]))  # 0ms

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await client.extract(schema_id="s1", text="doc")

    assert isinstance(out, hc.ExtractErr)
    assert out.status_code == 401
    assert out.error_code == "http_error"
    assert out.message == "no auth"
    assert out.extra is None

@pytest.mark.asyncio
async def test_generate_ok_when_payload_is_raw_string(monkeypatch):
    responses = [
        httpx.Response(status_code=200, content=b"\"hello world\""),
    ]

    _patch_httpx_async_client(monkeypatch, responses)
    monkeypatch.setattr(hc.time, "time", _TimeSeq([0.0, 0.01]))  # 10ms

    client = hc.HttpEvalClient(base_url="http://svc", api_key="KEY")
    out = await client.generate(prompt="p")

    assert isinstance(out, hc.GenerateOk)
    assert out.output_text == "hello world"
    assert out.model == "unknown"
    assert out.cached is False