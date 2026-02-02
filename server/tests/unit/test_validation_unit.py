# backend/tests/unit/test_validation_unit.py
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_parse_json_strict_accepts_object():
    from llm_server.core.validation import parse_json_strict

    assert parse_json_strict('{"a": 1}') == {"a": 1}


def test_parse_json_strict_rejects_empty():
    from llm_server.core.validation import StrictJSONError, parse_json_strict

    with pytest.raises(StrictJSONError) as e:
        parse_json_strict("   ")

    assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_code_fence():
    from llm_server.core.validation import StrictJSONError, parse_json_strict

    with pytest.raises(StrictJSONError) as e:
        parse_json_strict("```json\n{\"a\":1}\n```")

    assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_trailing_garbage():
    from llm_server.core.validation import StrictJSONError, parse_json_strict

    with pytest.raises(StrictJSONError) as e:
        parse_json_strict('{"a": 1} trailing')

    assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_nan_infinity():
    from llm_server.core.validation import StrictJSONError, parse_json_strict

    for bad in ['{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}']:
        with pytest.raises(StrictJSONError) as e:
            parse_json_strict(bad)
        assert e.value.code == "invalid_json"


def test_parse_json_strict_rejects_non_object_json():
    """
    Only keep this if your contract is "must be a JSON object".
    If arrays are allowed, delete this test.
    """
    from llm_server.core.validation import StrictJSONError, parse_json_strict

    with pytest.raises(StrictJSONError) as e:
        parse_json_strict("[1,2,3]")
    assert e.value.code == "invalid_json"