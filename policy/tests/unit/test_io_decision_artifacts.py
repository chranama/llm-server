from __future__ import annotations

import json
from pathlib import Path

from llm_policy.io.decision_artifacts import render_decision_artifact_json, write_decision_artifact
from llm_policy.types.decision import Decision, DecisionStatus


def test_render_decision_artifact_json_is_v1_contract() -> None:
    d = Decision(policy="extract_enablement", status=DecisionStatus.allow, enable_extract=True)
    s = render_decision_artifact_json(d)
    obj = json.loads(s)

    assert obj["schema_version"] == "policy_decision_v1"
    assert obj["ok"] is True
    assert obj["enable_extract"] is True
    assert "generated_at" in obj


def test_write_decision_artifact_writes_atomically(tmp_path: Path) -> None:
    out = tmp_path / "policy_out" / "latest.json"
    d = Decision(policy="extract_enablement", status=DecisionStatus.allow, enable_extract=True)

    p = write_decision_artifact(d, out)
    assert p == out
    assert out.exists()

    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["schema_version"] == "policy_decision_v1"