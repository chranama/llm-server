# tests/unit/cli/test_cli_persistence.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import llm_eval.cli as cli
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


class _FakeRunner(BaseEvalRunner):
    task_name = "fake_task"

    def __init__(self, payload: dict[str, Any]):
        # BaseEvalRunner requires base_url/api_key, but they won't be used by _run_impl.
        super().__init__(base_url="http://fake", api_key="fake", config=EvalConfig())
        self._payload = payload

    async def _run_impl(self) -> Any:
        return self._payload


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


@pytest.mark.asyncio
async def test_cli_saves_all_artifacts_with_runner_report(monkeypatch, tmp_path: Path, capsys):
    # --- monkeypatch config loading + api key resolution ---
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "run": {"outdir_root": str(tmp_path)},
            "datasets": {"fake_task": {"enabled": True, "max_items": 2}},
            "defaults": {"model_id": "m0"},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID123", "base_url": "http://svc"},
        "results": [{"id": "1"}, {"id": "2"}],
        "report_text": "hello report\n",
        "config": {"max_examples": 2, "model_override": "m0"},
    }

    # --- monkeypatch task factory to return our fake runner ---
    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )

    await cli.amain(
        [
            "--config",
            "ignored.yaml",
            "--task",
            "fake_task",
            "--no-print-summary",  # keep stdout clean
            "--save",  # saving is default ON, but explicit is fine
        ]
    )

    # (optional) drain stdout so it doesn't clutter test output if it changes later
    capsys.readouterr()

    outdir = tmp_path / "fake_task" / "RID123"
    assert outdir.exists()

    summary_p = outdir / "summary.json"
    results_p = outdir / "results.jsonl"
    report_p = outdir / "report.txt"
    config_p = outdir / "config.json"

    assert summary_p.exists()
    assert results_p.exists()
    assert report_p.exists()
    assert config_p.exists()

    summary = _read_json(summary_p)
    rows = _read_jsonl(results_p)
    report = report_p.read_text(encoding="utf-8")
    cfg = _read_json(config_p)

    assert summary["task"] == "fake_task"
    assert summary["run_id"] == "RID123"
    assert summary["run_dir"] == str(outdir)

    assert rows == [{"id": "1"}, {"id": "2"}]
    assert report == "hello report\n"
    assert cfg == {"max_examples": 2, "model_override": "m0"}


@pytest.mark.asyncio
async def test_cli_fallback_report_when_runner_report_missing(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "run": {"outdir_root": str(tmp_path)},
            "datasets": {"fake_task": {"enabled": True}},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    payload = {
        "summary": {
            "task": "fake_task",
            "run_id": "RID999",
            "base_url": "http://svc",
            "dataset": "D",
            "split": "S",
            "max_examples": 1,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
        },
        "results": [],
        "report_text": None,  # force fallback
        "config": {"max_examples": 1},
    }

    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )

    await cli.amain(
        [
            "--config",
            "ignored.yaml",
            "--task",
            "fake_task",
            "--no-print-summary",
            "--save",
        ]
    )
    capsys.readouterr()

    outdir = tmp_path / "fake_task" / "RID999"
    report = (outdir / "report.txt").read_text(encoding="utf-8")

    # minimal invariants of fallback report
    assert "task=fake_task" in report
    assert "run_id=RID999" in report
    assert "base_url=http://svc" in report
    assert "dataset=D" in report
    assert "split=S" in report
    assert "precision=0.9" in report
    assert "f1=0.85" in report


@pytest.mark.asyncio
async def test_cli_no_save_writes_nothing(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "run": {"outdir_root": str(tmp_path)},
            "datasets": {"fake_task": {"enabled": True}},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID0", "base_url": "http://svc"},
        "results": [{"id": "x"}],
        "report_text": "r\n",
        "config": {"max_examples": 1},
    }

    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )

    await cli.amain(
        [
            "--config",
            "ignored.yaml",
            "--task",
            "fake_task",
            "--no-print-summary",
            "--no-save",
        ]
    )
    capsys.readouterr()

    # nothing should be created under tmp_path because no-save
    assert list(tmp_path.glob("**/*")) == []