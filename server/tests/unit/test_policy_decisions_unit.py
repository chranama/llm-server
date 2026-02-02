# tests/unit/test_policy_decisions_unit.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_server.services.policy_decisions import (
    load_policy_decision_from_env,
    policy_capability_overrides,
)


def _write(p: Path, obj) -> None:
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_policy_no_env_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POLICY_DECISION_PATH", raising=False)
    snap = load_policy_decision_from_env()
    assert snap.ok is True
    assert snap.model_id is None
    assert snap.enable_extract is None
    assert snap.source_path is None
    assert snap.error is None
    assert snap.raw == {}


def test_policy_missing_file_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "missing.json"
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.error == "policy_decision_missing"
    assert snap.source_path == str(p)


def test_policy_invalid_json_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.source_path == str(p)
    assert snap.error is not None
    assert snap.error.startswith("policy_decision_parse_error:")


def test_policy_enable_extract_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "ok.json"
    _write(p, {"enable_extract": True})
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is True
    assert snap.enable_extract is True
    assert snap.error is None


def test_policy_contract_errors_nonzero_denies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "deny.json"
    _write(p, {"contract_errors": 2})
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.error == "policy_decision_not_ok"


def test_policy_status_deny_denies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "deny.json"
    _write(p, {"status": "deny"})
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))
    snap = load_policy_decision_from_env()
    assert snap.ok is False
    assert snap.enable_extract is False
    assert snap.error == "policy_decision_not_ok"


class _Req:
    """Tiny request stub for unit tests."""
    def __init__(self):
        class _State: ...
        class _App: ...
        self.app = _App()
        self.app.state = _State()


def test_policy_override_scoped_to_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "scoped.json"
    _write(p, {"model_id": "m1", "enable_extract": False})
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    req = _Req()
    assert policy_capability_overrides("m1", request=req) == {"extract": False}
    assert policy_capability_overrides("m2", request=req) is None


def test_policy_invalid_file_fail_closed_for_all_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("POLICY_DECISION_PATH", str(p))

    req = _Req()
    assert policy_capability_overrides("any", request=req) == {"extract": False}