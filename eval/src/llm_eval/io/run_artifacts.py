# eval/src/llm_eval/io/run_artifacts.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union


Pathish = Union[str, Path]


def _utc_now_iso() -> str:
    # RFC3339-ish, seconds precision, Z suffix
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    # Intentionally do not use default=str; fail loudly if not JSON-safe.
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, dict):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def default_outdir(root: str, task: str, run_id: str) -> str:
    return str(Path(root) / task / run_id)


@dataclass(frozen=True)
class EvalRunPaths:
    outdir: Path
    summary_json: Path
    results_jsonl: Path
    report_txt: Path
    report_md: Path
    config_json: Path


def make_run_paths(outdir: Pathish) -> EvalRunPaths:
    d = Path(outdir)
    return EvalRunPaths(
        outdir=d,
        summary_json=d / "summary.json",
        results_jsonl=d / "results.jsonl",
        report_txt=d / "report.txt",
        report_md=d / "report.md",
        config_json=d / "config.json",
    )


def write_eval_run_artifacts(
    *,
    outdir: Pathish,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    report_txt: str,
    report_md: str,
    returned_config: Optional[dict[str, Any]] = None,
) -> EvalRunPaths:
    """
    Canonical persistence for an eval run directory.

    Contract:
      - summary MUST include 'task', 'run_id', and 'run_dir' before writing.
      - Atomic writes for summary/report/config so downstream consumers never see partial JSON.
      - results.jsonl is written as a single file; not strictly atomic, but written to tmp then replaced.
    """
    paths = make_run_paths(outdir)

    # Ensure run_dir present before writing summary.json (your tests depend on this)
    summary = dict(summary)
    summary["run_dir"] = str(paths.outdir)

    _atomic_write_json(paths.summary_json, summary)

    if results:
        _write_jsonl(paths.results_jsonl, results)

    _atomic_write_text(paths.report_txt, report_txt)
    _atomic_write_text(paths.report_md, report_md)

    if isinstance(returned_config, dict):
        _atomic_write_json(paths.config_json, returned_config)

    return paths


def default_eval_out_pointer_path() -> Path:
    """
    Conventional host path for a pointer artifact that indicates the latest run.
    Mirrors policy_out/latest.json style, but for eval.
    """
    return Path("eval_out") / "latest.json"


def write_eval_latest_pointer(
    *,
    pointer_path: Pathish,
    task: str,
    run_id: str,
    run_dir: Pathish,
    summary_path: Optional[Pathish] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Write a tiny pointer artifact for "latest eval run".

    This is NOT a duplicate of summary.json. It just points at it.
    """
    p = Path(pointer_path)
    payload: dict[str, Any] = {
        "schema_version": "eval_pointer_v1",
        "generated_at": _utc_now_iso(),
        "task": task,
        "run_id": run_id,
        "run_dir": str(Path(run_dir)),
    }
    if summary_path is not None:
        payload["summary_path"] = str(Path(summary_path))
    if extra:
        # non-authoritative convenience data
        payload["extra"] = dict(extra)

    _atomic_write_json(p, payload)
    return p