# tests/unit/cli/test_cli_persistence_edges.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

import llm_eval.cli as cli
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


class _FakeRunner(BaseEvalRunner):
    task_name = "fake_task"

    def __init__(self, payload: dict[str, Any]):
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


def _monkeypatch_core_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "run": {"outdir_root": str(tmp_path)},
            "datasets": {"fake_task": {"enabled": True}},
            "defaults": {"model_id": "m0"},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")


def _monkeypatch_task(monkeypatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )


def _argv(
    *,
    outdir: Optional[str] = None,
    no_save: bool = False,
    save: bool = False,
    no_print_summary: bool = True,
    print_summary: bool = False,
) -> cli.argparse.Namespace:
    return cli.argparse.Namespace(
        config="ignored.yaml",
        task="fake_task",
        list_tasks=False,
        base_url=None,
        api_key=None,
        max_examples=None,
        model=None,
        print_summary=print_summary,
        no_print_summary=no_print_summary,
        save=save,
        no_save=no_save,
        outdir=outdir,
        debug_n=0,
        debug_fields=None,
    )


@pytest.mark.asyncio
async def test_cli_outdir_override_writes_to_exact_path(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    outdir = tmp_path / "custom" / "place"
    payload = {
        "summary": {"task": "fake_task", "run_id": "RID1", "base_url": "http://svc"},
        "results": [{"id": "1"}],
        "report_text": "r\n",
        "config": {"max_examples": 1},
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(outdir=str(outdir)),
    )

    await cli.amain()

    assert outdir.exists()
    assert (outdir / "summary.json").exists()
    assert (outdir / "results.jsonl").exists()
    assert (outdir / "report.txt").exists()
    assert (outdir / "config.json").exists()

    summary = _read_json(outdir / "summary.json")
    assert summary["run_dir"] == str(outdir)


@pytest.mark.asyncio
async def test_cli_does_not_write_results_jsonl_when_results_empty(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID2", "base_url": "http://svc"},
        "results": [],
        "report_text": "r\n",
        "config": {"max_examples": 0},
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(),
    )

    await cli.amain()

    outdir = tmp_path / "fake_task" / "RID2"
    assert (outdir / "summary.json").exists()
    assert (outdir / "report.txt").exists()
    assert (outdir / "config.json").exists()
    assert not (outdir / "results.jsonl").exists()


@pytest.mark.asyncio
async def test_cli_does_not_write_config_json_when_config_missing(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID3", "base_url": "http://svc"},
        "results": [{"id": "x"}],
        "report_text": "r\n",
        # no config
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(),
    )

    await cli.amain()

    outdir = tmp_path / "fake_task" / "RID3"
    assert (outdir / "summary.json").exists()
    assert (outdir / "results.jsonl").exists()
    assert (outdir / "report.txt").exists()
    assert not (outdir / "config.json").exists()


@pytest.mark.asyncio
async def test_cli_overwrites_existing_artifacts_on_rerun(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    outdir = tmp_path / "fake_task" / "RID4"
    outdir.mkdir(parents=True, exist_ok=True)

    # seed existing files with old content
    (outdir / "summary.json").write_text('{"old": true}', encoding="utf-8")
    (outdir / "report.txt").write_text("old report\n", encoding="utf-8")
    (outdir / "results.jsonl").write_text('{"id":"old"}\n', encoding="utf-8")

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID4", "base_url": "http://svc"},
        "results": [{"id": "new"}],
        "report_text": "new report\n",
        "config": {"max_examples": 1},
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(),
    )

    await cli.amain()

    summary = _read_json(outdir / "summary.json")
    assert summary["task"] == "fake_task"
    assert summary["run_id"] == "RID4"
    assert summary["run_dir"] == str(outdir)

    rows = _read_jsonl(outdir / "results.jsonl")
    assert rows == [{"id": "new"}]

    report = (outdir / "report.txt").read_text(encoding="utf-8")
    assert report == "new report\n"


@pytest.mark.asyncio
async def test_cli_task_and_run_id_are_filled_if_missing(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    # runner "forgets" task/run_id; CLI must fill them in
    payload = {
        "summary": {"base_url": "http://svc"},
        "results": [],
        "report_text": None,  # fallback report
        "config": {},
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(),
    )

    await cli.amain()

    # We don't know the generated run_id; find the created directory.
    task_root = tmp_path / "fake_task"
    assert task_root.exists()

    run_dirs = [p for p in task_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    outdir = run_dirs[0]

    summary = _read_json(outdir / "summary.json")
    assert summary["task"] == "fake_task"
    assert isinstance(summary["run_id"], str) and summary["run_id"]
    assert summary["run_dir"] == str(outdir)

    report = (outdir / "report.txt").read_text(encoding="utf-8")
    assert "task=fake_task" in report
    assert "run_id=" in report
    assert "base_url=http://svc" in report


@pytest.mark.asyncio
async def test_cli_report_text_fallback_creates_report_file(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    payload = {
        "summary": {
            "task": "fake_task",
            "run_id": "RID5",
            "base_url": "http://svc",
            "precision": 0.5,
            "f1": 0.25,
        },
        "results": [{"id": "1"}],
        "report_text": "   ",  # blank -> fallback
        "config": {"max_examples": 1},
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(),
    )

    await cli.amain()

    outdir = tmp_path / "fake_task" / "RID5"
    report = (outdir / "report.txt").read_text(encoding="utf-8")
    assert "task=fake_task" in report
    assert "run_id=RID5" in report
    assert "precision=0.5" in report
    assert "f1=0.25" in report


@pytest.mark.asyncio
async def test_cli_no_save_does_not_create_outdir(monkeypatch, tmp_path: Path):
    _monkeypatch_core_config(monkeypatch, tmp_path)

    payload = {
        "summary": {"task": "fake_task", "run_id": "RID6", "base_url": "http://svc"},
        "results": [{"id": "x"}],
        "report_text": "r\n",
        "config": {"max_examples": 1},
    }
    _monkeypatch_task(monkeypatch, payload)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: _argv(no_save=True),
    )

    await cli.amain()

    assert list(tmp_path.glob("**/*")) == []