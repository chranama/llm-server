from __future__ import annotations

import json

from llm_policy.reports.writer import render_decision_json, render_decision_md, render_decision_text
from llm_policy.types.decision import (
    Decision,
    DecisionReason,
    DecisionStatus,
    DecisionWarning,
)


def test_render_decision_text_includes_core_fields():
    d = Decision(
        policy="extract_enablement",
        status=DecisionStatus.allow,
        enable_extract=True,
        thresholds_profile="extract/sroie",
        reasons=[],
        warnings=[],
        metrics={"n_total": 10, "schema_validity_rate": 0.99},
    )

    out = render_decision_text(d)

    assert "policy=extract_enablement" in out
    assert "thresholds_profile=extract/sroie" in out
    assert "enable_extract=True" in out
    assert "status=DecisionStatus.allow" in out or "status=allow" in out
    assert "ok=True" in out


def test_render_decision_text_coerces_reasons_and_warnings():
    # mix pydantic models and dicts (including old-style "extra")
    d = Decision(
        policy="extract_enablement",
        status=DecisionStatus.deny,
        enable_extract=False,
        reasons=[
            DecisionReason(code="schema_validity_too_low", message="too low", context={"cur": 0.8}),
            {"code": "missing_metric", "message": "required_present_rate missing", "extra": {"k": "v"}},
        ],
        warnings=[
            DecisionWarning(code="insufficient_sample_size", message="low N", context={"n": 3}),
        ],
        metrics={"n_total": 3},
    )

    out = render_decision_text(d)
    assert "REASONS:" in out
    assert "- schema_validity_too_low: too low" in out
    assert "- missing_metric: required_present_rate missing" in out

    assert "WARNINGS:" in out
    assert "- insufficient_sample_size: low N" in out

    md = render_decision_md(d)
    assert "## Reasons" in md
    assert "**schema_validity_too_low**" in md
    assert "**missing_metric**" in md
    assert "## Warnings" in md
    assert "**insufficient_sample_size**" in md


def test_render_decision_json_roundtrips():
    d = Decision(
        policy="extract_enablement",
        status=DecisionStatus.allow,
        enable_extract=True,
        reasons=[],
        warnings=[],
        metrics={"a": 1},
    )

    s = render_decision_json(d)
    payload = json.loads(s)

    # Should be exactly model_dump output
    assert payload == d.model_dump()
    assert payload["policy"] == "extract_enablement"
    assert payload["enable_extract"] is True