# backend/tests/unit/test_errors_unit.py
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


class DummyState:
    def __init__(self, request_id: str):
        self.request_id = request_id


class DummyRequest:
    def __init__(self, request_id: str = "rid123"):
        self.state = DummyState(request_id)


def test_to_json_error_shape_and_request_id_header():
    from llm_server.core.errors import _to_json_error

    req = DummyRequest("abc")
    resp = _to_json_error(  # type: ignore[arg-type]
        req,
        status_code=418,
        code="teapot",
        message="nope",
        extra={"foo": "bar"},
    )

    assert resp.status_code == 418
    assert resp.headers.get("X-Request-ID") == "abc"

    payload = json.loads(resp.body.decode("utf-8"))
    assert payload["code"] == "teapot"
    assert payload["message"] == "nope"
    assert payload["request_id"] == "abc"
    assert payload["extra"]["foo"] == "bar"