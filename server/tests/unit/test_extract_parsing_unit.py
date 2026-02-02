# backend/tests/unit/test_extract_parsing_unit.py
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_iter_json_objects_finds_multiple_dicts():
    from llm_server.api.extract import _iter_json_objects

    raw = 'noise {"a": 1} mid {"b": 2} tail'
    objs = list(_iter_json_objects(raw))

    assert {"a": 1} in objs
    assert {"b": 2} in objs


def test_iter_json_objects_ignores_arrays_and_scalars():
    from llm_server.api.extract import _iter_json_objects

    raw = '["x"] {"a": 1} 123 {"b": 2}'
    objs = list(_iter_json_objects(raw))

    assert {"a": 1} in objs
    assert {"b": 2} in objs
    assert all(isinstance(o, dict) for o in objs)


def test_validate_first_matching_prefers_delimited_json(monkeypatch):
    import llm_server.api.extract as ex

    # accept only {"ok": True}
    def fake_validate(schema, data):
        if data != {"ok": True}:
            raise ex.JSONSchemaValidationError(code="schema_validation_failed", message="nope", errors=[])

    monkeypatch.setattr(ex, "validate_jsonschema", fake_validate, raising=True)

    schema = {"type": "object"}  # irrelevant due to fake_validate
    raw = 'noise {"ok": false} ' + ex._JSON_BEGIN + '\n{"ok": true}\n' + ex._JSON_END + " tail"
    out = ex._validate_first_matching(schema, raw)

    assert out == {"ok": True}


def test_validate_first_matching_raises_invalid_json_when_no_objects(monkeypatch):
    import llm_server.api.extract as ex

    monkeypatch.setattr(ex, "validate_jsonschema", lambda s, d: None, raising=True)

    schema = {"type": "object"}
    with pytest.raises(ex.AppError) as e:
        ex._validate_first_matching(schema, "no json here")

    assert e.value.code == "invalid_json"
    assert e.value.status_code == 422


def test_validate_first_matching_raises_schema_validation_failed_when_no_candidate_valid(monkeypatch):
    import llm_server.api.extract as ex

    def always_fail(schema, data):
        raise ex.JSONSchemaValidationError(code="schema_validation_failed", message="bad", errors=[{"loc": "$"}])

    monkeypatch.setattr(ex, "validate_jsonschema", always_fail, raising=True)

    schema = {"type": "object"}
    with pytest.raises(ex.AppError) as e:
        ex._validate_first_matching(schema, 'noise {"a": 1} {"b": 2}')

    assert e.value.code == "schema_validation_failed"
    assert e.value.status_code == 422


def test_failure_stage_mapping():
    from fastapi import status
    from llm_server.api.extract import _failure_stage_for_app_error
    from llm_server.core.errors import AppError

    e_parse = AppError(code="invalid_json", message="x", status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)
    e_val = AppError(code="schema_validation_failed", message="x", status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    assert _failure_stage_for_app_error(e_parse, is_repair=False) == "parse"
    assert _failure_stage_for_app_error(e_val, is_repair=False) == "validate"
    assert _failure_stage_for_app_error(e_parse, is_repair=True) == "repair_parse"
    assert _failure_stage_for_app_error(e_val, is_repair=True) == "repair_validate"