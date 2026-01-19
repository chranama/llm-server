# tests/unit/client/test_http_client_protocol_types.py
from __future__ import annotations

from typing import Any, Optional

import pytest

from llm_eval.client.http_client import ExtractErr, ExtractOk, GenerateErr, GenerateOk


def _assert_has_latency_ms(obj: Any) -> None:
    assert hasattr(obj, "latency_ms"), f"{type(obj).__name__} missing latency_ms"
    v = getattr(obj, "latency_ms")
    assert isinstance(v, (int, float)), f"{type(obj).__name__}.latency_ms must be numeric, got {type(v).__name__}"
    # normalize to float expectation (your code treats it as float-like)
    float(v)


def test_generate_ok_shape_and_types():
    r = GenerateOk(model="m", output_text="hi", cached=False, latency_ms=12.5)
    assert isinstance(r.model, str)
    assert isinstance(r.output_text, str)
    assert isinstance(r.cached, bool)
    _assert_has_latency_ms(r)


def test_generate_ok_allows_none_fields_where_expected():
    # Some callers may return None output_text or model; runners handle this.
    r = GenerateOk(model=None, output_text=None, cached=False, latency_ms=0.0)
    assert r.model is None
    assert r.output_text is None
    assert isinstance(r.cached, bool)
    _assert_has_latency_ms(r)


def test_generate_err_shape_and_types():
    r = GenerateErr(
        status_code=502,
        error_code="bad_gateway",
        message="oops",
        extra={"stage": "upstream"},
        latency_ms=33.3,
    )
    assert isinstance(r.status_code, int)
    assert isinstance(r.error_code, str)
    assert isinstance(r.message, str)
    assert isinstance(r.extra, dict)
    _assert_has_latency_ms(r)


def test_generate_err_allows_none_message_and_extra():
    r = GenerateErr(
        status_code=500,
        error_code="internal_error",
        message=None,
        extra=None,
        latency_ms=0.0,
    )
    assert isinstance(r.status_code, int)
    assert isinstance(r.error_code, str)
    assert r.message is None
    assert r.extra is None
    _assert_has_latency_ms(r)


def test_extract_ok_shape_and_types():
    r = ExtractOk(
        schema_id="sroie_receipt_v1",
        model="m",
        data={"company": "ACME"},
        cached=False,
        repair_attempted=False,
        latency_ms=10.0,
    )
    assert isinstance(r.schema_id, str)
    assert isinstance(r.data, dict)
    assert isinstance(r.cached, bool)
    assert isinstance(r.repair_attempted, bool)
    _assert_has_latency_ms(r)


def test_extract_ok_allows_none_model():
    r = ExtractOk(
        schema_id="sroie_receipt_v1",
        model=None,
        data={"company": "ACME"},
        cached=False,
        repair_attempted=False,
        latency_ms=1.0,
    )
    assert r.model is None
    _assert_has_latency_ms(r)


def test_extract_err_shape_and_types():
    r = ExtractErr(
        status_code=422,
        error_code="schema_validation_failed",
        message="bad json",
        extra={"stage": "validate"},
        latency_ms=9.0,
    )
    assert isinstance(r.status_code, int)
    assert isinstance(r.error_code, str)
    assert isinstance(r.message, str)
    assert isinstance(r.extra, dict)
    _assert_has_latency_ms(r)


def test_extract_err_allows_none_message_and_extra():
    r = ExtractErr(
        status_code=500,
        error_code="internal_error",
        message=None,
        extra=None,
        latency_ms=0.0,
    )
    assert r.message is None
    assert r.extra is None
    _assert_has_latency_ms(r)


@pytest.mark.parametrize(
    "obj",
    [
        GenerateOk(model="m", output_text="x", cached=False, latency_ms=0.0),
        GenerateErr(status_code=500, error_code="e", message=None, extra=None, latency_ms=0.0),
        ExtractOk(
            schema_id="s",
            model="m",
            data={},
            cached=False,
            repair_attempted=False,
            latency_ms=0.0,
        ),
        ExtractErr(status_code=500, error_code="e", message=None, extra=None, latency_ms=0.0),
    ],
)
def test_all_protocol_results_have_latency_ms(obj: Any):
    _assert_has_latency_ms(obj)