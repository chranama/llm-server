from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_policy.cli import main


@pytest.fixture(autouse=True)
def _patch_cli_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Patch:
      - load_eval_artifact
      - load_extract_thresholds
      - decide_extract_enablement
      - renderers
    So CLI tests are stable and don't depend on filesystem thresholds.
    """

    class FakeDecision:
        def __init__(self, enable: bool):
            self.enable_extract = enable
            self._ok = enable

        def ok(self) -> bool:
            return self._ok

    class FakeArtifact:
        pass

    monkeypatch.setattr("llm_policy.cli.load_eval_artifact", lambda run_dir: FakeArtifact(), raising=True)
    monkeypatch.setattr(
        "llm_policy.cli.load_extract_thresholds",
        lambda cfg, profile=None: (profile or "extract/default", {"fake": True}),
        raising=True,
    )

    # default decision: allow
    monkeypatch.setattr(
        "llm_policy.cli.decide_extract_enablement",
        lambda artifact, thresholds, thresholds_profile=None: FakeDecision(True),
        raising=True,
    )

    monkeypatch.setattr("llm_policy.cli.render_decision_text", lambda d: "TEXT\n", raising=True)
    monkeypatch.setattr("llm_policy.cli.render_decision_md", lambda d: "MD\n", raising=True)
    monkeypatch.setattr("llm_policy.cli.render_decision_json", lambda d: json.dumps({"ok": d.ok()}) + "\n", raising=True)


def test_decide_extract_text_to_stdout(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    rc = main(["decide-extract", "--run-dir", str(tmp_path), "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "TEXT\n"


def test_decide_extract_md_to_file(tmp_path: Path):
    out_file = tmp_path / "out.md"
    rc = main(["decide-extract", "--run-dir", str(tmp_path), "--format", "md", "--out", str(out_file)])
    assert rc == 0
    assert out_file.read_text(encoding="utf-8") == "MD\n"


def test_decide_extract_json(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    rc = main(["decide-extract", "--run-dir", str(tmp_path), "--format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["ok"] is True


def test_decide_extract_exit_code_is_2_when_not_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeDecision:
        enable_extract = False
        def ok(self) -> bool:
            return False

    monkeypatch.setattr(
        "llm_policy.cli.decide_extract_enablement",
        lambda artifact, thresholds, thresholds_profile=None: FakeDecision(),
        raising=True,
    )

    rc = main(["decide-extract", "--run-dir", str(tmp_path), "--format", "text"])
    assert rc == 2