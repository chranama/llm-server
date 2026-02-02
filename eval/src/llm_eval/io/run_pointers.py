from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from llm_contracts.runtime.eval_run_pointer import (
    EvalRunPointerSnapshot,
    build_eval_run_pointer_payload_v1,
    read_eval_run_pointer,
    write_eval_run_pointer,
)

Pathish = Union[str, Path]


def _env_flag(name: str, default: bool) -> bool:
    """
    Parse env var as a boolean-ish flag.
    Accepts: 1/0, true/false, yes/no, on/off (case-insensitive).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def pointer_out_path_for_task(task: str) -> Path:
    """
    Determine where to write the eval_out "latest" pointer.

    Canonical contract:
      - One pointer for "the latest eval run" for the currently executed task:
          eval_out/latest.json

    Priority:
      1) EVAL_LATEST_PATH  (explicit single file path override)
      2) ./eval_out/latest.json (repo-relative; containers typically run in /work)

    Notes:
      - `task` is intentionally unused for the default path.
        It's retained in the function signature for backwards compatibility and
        to allow call sites to keep passing task without refactors.
    """
    env = os.getenv("EVAL_LATEST_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path("eval_out") / "latest.json").resolve()


def should_write_eval_latest_pointer(*, default: bool = True) -> bool:
    """
    Centralized switch to enable/disable writing eval_out pointers.
    - If saving run artifacts is on, pointer writing is usually on as well.
    - Can be disabled via EVAL_WRITE_LATEST=0.
    """
    return _env_flag("EVAL_WRITE_LATEST", default=default)


def build_eval_run_pointer(
    *,
    task: str,
    run_id: str,
    run_dir: str,
    summary_path: str,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    schema_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    notes: Optional[Dict[str, Any]] = None,
    store: str = "fs",
) -> Dict[str, Any]:
    """
    Build a validated pointer payload (dict) using llm_contracts as the contract source of truth.
    """
    return build_eval_run_pointer_payload_v1(
        task=task,
        run_id=run_id,
        store=store,
        run_dir=run_dir,
        summary_path=summary_path,
        base_url=base_url,
        model_override=model_override,
        schema_id=schema_id,
        max_examples=max_examples,
        notes=notes,
    )


def write_eval_latest_pointer(
    *,
    task: str,
    run_id: str,
    run_dir: str,
    summary_path: str,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    schema_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    notes: Optional[Dict[str, Any]] = None,
    out_path: Optional[Pathish] = None,
    store: str = "fs",
) -> Path:
    """
    Write the canonical eval_out/latest.json pointer atomically.

    This is a convenience pointer only, NOT the authoritative data boundary.
    The authoritative run artifacts remain in results/<task>/<run_id>/.

    - Default path: eval_out/latest.json
    - Override: pass out_path=... OR set EVAL_LATEST_PATH
    """
    payload = build_eval_run_pointer(
        task=task,
        run_id=run_id,
        run_dir=run_dir,
        summary_path=summary_path,
        base_url=base_url,
        model_override=model_override,
        schema_id=schema_id,
        max_examples=max_examples,
        notes=notes,
        store=store,
    )

    p = Path(out_path).expanduser().resolve() if out_path is not None else pointer_out_path_for_task(task)
    return write_eval_run_pointer(p, payload)


def read_eval_latest_pointer(*, task: str, path: Optional[Pathish] = None) -> EvalRunPointerSnapshot:
    """
    Read and validate the canonical eval_out/latest.json pointer.

    Returns a fail-closed snapshot (ok=False) on parse/validation errors.
    """
    p = Path(path).expanduser().resolve() if path is not None else pointer_out_path_for_task(task)
    return read_eval_run_pointer(p)