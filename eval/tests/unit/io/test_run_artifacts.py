# eval/tests/unit/io/test_run_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

# This is the new io module you introduced in the CLI refactor.
# It should own filesystem semantics for eval artifacts.
from llm_eval.io.run_artifacts import (
    default_eval_out_pointer_path,
    default_outdir,
    write_eval_latest_pointer,
    write_eval_run_artifacts,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_default_outdir_is_stable_and_expected() -> None:
    out = default_outdir("results", "extraction_sroie", "20260101T000000Z")
    # Default should be: <root>/<task>/<run_id>/
    assert out.replace("\\", "/") == "results/extraction_sroie/20260101T000000Z"


@pytest.mark.unit
def test_write_eval_run_artifacts_writes_expected_files(tmp_path: Path) -> None:
    outdir = tmp_path / "results" / "extraction_sroie" / "RUN123"

    summary = {
        "task": "extraction_sroie",
        "run_id": "RUN123",
        "base_url": "http://localhost:8000",
        "run_dir": str(outdir),
        "schema_validity_rate": 0.9,
        "latency_p95_ms": 123.4,
    }
    results = [
        {"id": "1", "ok": True, "status_code": 200, "latency_ms": 12.3},
        {"id": "2", "ok": False, "status_code": 500, "error_code": "server_error", "latency_ms": 50.0},
    ]
    report_txt = "task=extraction_sroie\nrun_id=RUN123\n"
    report_md = "# Eval Report\n"

    returned_config = {"foo": "bar"}

    paths = write_eval_run_artifacts(
        outdir=str(outdir),
        summary=summary,
        results=results,
        report_txt=report_txt,
        report_md=report_md,
        returned_config=returned_config,
    )

    # Expected paths exist
    assert paths.outdir.exists()
    assert paths.summary_json.exists()
    assert paths.results_jsonl.exists()
    assert paths.report_txt.exists()
    assert paths.report_md.exists()
    assert paths.config_json is not None
    assert paths.config_json.exists()

    # summary.json round-trips
    loaded_summary = _read_json(paths.summary_json)
    assert loaded_summary["task"] == "extraction_sroie"
    assert loaded_summary["run_id"] == "RUN123"
    assert loaded_summary["run_dir"] == str(outdir)

    # results.jsonl is jsonl with same number of dict rows
    lines = _read_text(paths.results_jsonl).strip().splitlines()
    assert len(lines) == len(results)
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["id"] == "1"
    assert row1["id"] == "2"

    # reports persisted as exact text
    assert _read_text(paths.report_txt) == report_txt
    assert _read_text(paths.report_md) == report_md

    # config.json persisted
    loaded_cfg = _read_json(paths.config_json)
    assert loaded_cfg == returned_config


@pytest.mark.unit
def test_write_eval_run_artifacts_omits_optional_files_when_empty(tmp_path: Path) -> None:
    outdir = tmp_path / "results" / "task" / "RUN0"

    summary = {
        "task": "task",
        "run_id": "RUN0",
        "base_url": "http://localhost:8000",
        "run_dir": str(outdir),
    }

    # results empty -> should not write results.jsonl (expected contract)
    paths = write_eval_run_artifacts(
        outdir=str(outdir),
        summary=summary,
        results=[],
        report_txt="ok\n",
        report_md="# ok\n",
        returned_config=None,
    )

    assert paths.summary_json.exists()
    assert paths.report_txt.exists()
    assert paths.report_md.exists()

    # Optional outputs should be absent
    assert paths.results_jsonl is None
    assert paths.config_json is None


@pytest.mark.unit
def test_default_eval_out_pointer_path_is_eval_out_latest_json() -> None:
    p = default_eval_out_pointer_path()
    # allow cwd-relative path; just assert the convention
    assert p.as_posix().endswith("eval_out/latest.json")


@pytest.mark.unit
def test_write_eval_latest_pointer_writes_expected_shape(tmp_path: Path) -> None:
    pointer = tmp_path / "eval_out" / "latest.json"
    run_dir = tmp_path / "results" / "task" / "RUN1"
    summary_path = run_dir / "summary.json"

    # Create fake run_dir + summary.json to simulate a real run
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"task": "task", "run_id": "RUN1"}, indent=2), encoding="utf-8")

    out_path = write_eval_latest_pointer(
        pointer_path=str(pointer),
        task="task",
        run_id="RUN1",
        run_dir=str(run_dir),
        summary_path=str(summary_path),
        extra={"base_url": "http://localhost:8000"},
    )

    assert out_path.exists()
    payload = _read_json(out_path)

    # Required stable fields
    assert payload["schema_version"] == "eval_latest_v1"
    assert payload["task"] == "task"
    assert payload["run_id"] == "RUN1"
    assert payload["run_dir"] == str(run_dir)
    assert payload["summary_path"] == str(summary_path)

    # Non-authoritative extra is present
    assert payload["extra"]["base_url"] == "http://localhost:8000"

    # generated_at should exist and be non-empty string
    assert isinstance(payload.get("generated_at"), str)
    assert payload["generated_at"].strip() != ""


@pytest.mark.unit
def test_write_eval_latest_pointer_is_atomic_and_does_not_leave_tmp(tmp_path: Path) -> None:
    pointer = tmp_path / "eval_out" / "latest.json"
    run_dir = tmp_path / "results" / "task" / "RUN2"
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    out_path = write_eval_latest_pointer(
        pointer_path=pointer,
        task="task",
        run_id="RUN2",
        run_dir=run_dir,
        summary_path=summary_path,
        extra=None,
    )
    assert out_path.exists()

    # Ensure temp file is not left behind
    tmp_candidate = pointer.with_suffix(pointer.suffix + ".tmp")
    assert not tmp_candidate.exists()