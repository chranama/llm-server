from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """
    A fake eval run directory that looks like eval/results/<task>/<run_id>/.
    load_eval_artifact(run_dir) should read summary.json (and optionally results.jsonl).
    """
    run_dir = tmp_path / "results" / "extraction_sroie" / "20260131T000000Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_results_jsonl(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    p = run_dir / "results.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def make_run_dir(tmp_run_dir):
    """
    Factory that writes a minimal summary.json (and optional results.jsonl) into tmp_run_dir.
    Returns the run_dir Path.
    """
    def _make(*, summary: dict[str, Any], results: list[dict[str, Any]] | None = None):
        write_summary(tmp_run_dir, summary)
        if results is not None:
            write_results_jsonl(tmp_run_dir, results)
        return tmp_run_dir

    return _make