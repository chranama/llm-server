from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_policy.cli import main


def test_patch_models_rejects_conflicting_flags(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    rc = main([
        "patch-models",
        "--models-yaml", str(tmp_path / "models.yaml"),
        "--model-id", "m1",
        "--enable-extract",
        "--disable-extract",
    ])
    assert rc == 2
    out = capsys.readouterr().out
    assert "choose only one" in out.lower()


def test_patch_models_calls_patch_models_yaml(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    calls = {}

    class Res:
        changed = True
        warnings = ["w1"]

    def fake_patch(path, model_id, capability, enable, write):
        calls["path"] = path
        calls["model_id"] = model_id
        calls["capability"] = capability
        calls["enable"] = enable
        calls["write"] = write
        return Res()

    monkeypatch.setattr("llm_policy.cli.patch_models_yaml", fake_patch, raising=True)

    rc = main([
        "patch-models",
        "--models-yaml", str(tmp_path / "models.yaml"),
        "--model-id", "m1",
        "--disable-extract",
        "--dry-run",
    ])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["changed"] is True
    assert payload["warnings"] == ["w1"]

    assert calls["model_id"] == "m1"
    assert calls["capability"] == "extract"
    assert calls["enable"] is False
    assert calls["write"] is False