# tests/unit/cli/test_cli_contract.py
from __future__ import annotations

from typing import Any

import pytest

import llm_eval.cli as cli


def _coerce(x: Any):
    return cli._coerce_nested_payload(x)


def test_cli_rejects_non_dict_payload() -> None:
    with pytest.raises(TypeError, match=r"expected dict"):
        _coerce("not a dict")


def test_cli_rejects_missing_summary() -> None:
    with pytest.raises(TypeError, match=r"must include key 'summary'"):
        _coerce({"results": []})


def test_cli_rejects_summary_not_dict() -> None:
    with pytest.raises(TypeError, match=r"key 'summary' with a dict"):
        _coerce({"summary": "nope", "results": []})


def test_cli_rejects_results_not_list() -> None:
    with pytest.raises(TypeError, match=r"results.*must be a list"):
        _coerce({"summary": {"task": "t"}, "results": {"id": "1"}})


def test_cli_results_none_is_treated_as_empty_list() -> None:
    summary, results, report_text, returned_config = _coerce(
        {"summary": {"task": "t"}, "results": None}
    )
    assert summary["task"] == "t"
    assert results == []
    assert report_text is None
    assert returned_config is None


def test_cli_filters_non_dict_results_rows() -> None:
    summary, results, report_text, returned_config = _coerce(
        {
            "summary": {"task": "t"},
            "results": [{"id": "1"}, 123, "x", None, {"id": "2"}],
        }
    )
    assert summary["task"] == "t"
    assert results == [{"id": "1"}, {"id": "2"}]
    assert report_text is None
    assert returned_config is None


def test_cli_report_text_is_none_when_missing_or_blank() -> None:
    _, _, report_text1, _ = _coerce({"summary": {"task": "t"}, "results": []})
    assert report_text1 is None

    _, _, report_text2, _ = _coerce(
        {"summary": {"task": "t"}, "results": [], "report_text": "   "}
    )
    assert report_text2 is None

    _, _, report_text3, _ = _coerce(
        {"summary": {"task": "t"}, "results": [], "report_text": "ok\n"}
    )
    assert report_text3 == "ok\n"


def test_cli_config_is_none_when_missing_or_wrong_type() -> None:
    _, _, _, cfg1 = _coerce({"summary": {"task": "t"}, "results": []})
    assert cfg1 is None

    _, _, _, cfg2 = _coerce({"summary": {"task": "t"}, "results": [], "config": "x"})
    assert cfg2 is None

    _, _, _, cfg3 = _coerce(
        {"summary": {"task": "t"}, "results": [], "config": {"a": 1}}
    )
    assert cfg3 == {"a": 1}

def test_cli_summary_is_copied_not_aliased() -> None:
    payload = {"summary": {"task": "t"}, "results": []}
    summary, _, _, _ = _coerce(payload)

    # mutate original
    payload["summary"]["task"] = "mutated"
    assert summary["task"] == "t"

    # mutate returned
    summary["task"] = "mutated2"
    assert payload["summary"]["task"] == "mutated"