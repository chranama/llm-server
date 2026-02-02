# contracts/src/llm_contracts/runtime/eval_run_pointer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from llm_contracts.schema import (
    atomic_write_json_internal,
    read_json_internal,
    validate_internal,
)

Pathish = Union[str, Path]

EVAL_RUN_POINTER_SCHEMA = "eval_run_pointer_v1.schema.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EvalRunPointerSnapshot:
    """
    Stable, minimal API for the rest of the repo.
    Treat this as the "public contract" inside Python.

    - raw: validated dict payload (for debugging / forward-compat)
    """
    ok: bool
    schema_version: str
    generated_at: str

    task: str
    run_id: str
    store: str  # "fs" | "db"

    run_dir: Optional[str]
    summary_path: Optional[str]

    # non-authoritative convenience fields (safe to ignore)
    base_url: Optional[str]
    model_override: Optional[str]
    schema_id: Optional[str]
    max_examples: Optional[int]
    notes: Optional[Dict[str, Any]]

    raw: Dict[str, Any]
    source_path: Optional[str] = None
    error: Optional[str] = None


def build_eval_run_pointer_payload_v1(
    *,
    task: str,
    run_id: str,
    store: str = "fs",
    run_dir: Optional[str] = None,
    summary_path: Optional[str] = None,
    base_url: Optional[str] = None,
    model_override: Optional[str] = None,
    schema_id: Optional[str] = None,
    max_examples: Optional[int] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "eval_run_pointer_v1",
        "generated_at": _utc_now_iso(),
        "task": task,
        "run_id": run_id,
        "store": store,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "base_url": base_url,
        "model_override": model_override,
        "schema_id": schema_id,
        "max_examples": max_examples,
        "notes": notes,
    }

    # keep compact
    payload = {k: v for k, v in payload.items() if v is not None}

    validate_internal(EVAL_RUN_POINTER_SCHEMA, payload)
    return payload


def parse_eval_run_pointer(payload: Dict[str, Any], *, source_path: Optional[str] = None) -> EvalRunPointerSnapshot:
    """
    Validate + parse payload into stable snapshot.
    """
    validate_internal(EVAL_RUN_POINTER_SCHEMA, payload)

    # Required fields per schema
    schema_version = str(payload["schema_version"])
    generated_at = str(payload["generated_at"])
    task = str(payload["task"])
    run_id = str(payload["run_id"])
    store = str(payload["store"])

    # Optional fields
    run_dir = payload.get("run_dir")
    summary_path = payload.get("summary_path")

    snap = EvalRunPointerSnapshot(
        ok=True,
        schema_version=schema_version,
        generated_at=generated_at,
        task=task,
        run_id=run_id,
        store=store,
        run_dir=str(run_dir) if isinstance(run_dir, str) else None,
        summary_path=str(summary_path) if isinstance(summary_path, str) else None,
        base_url=str(payload.get("base_url")) if isinstance(payload.get("base_url"), str) else None,
        model_override=payload.get("model_override") if isinstance(payload.get("model_override"), str) else None,
        schema_id=payload.get("schema_id") if isinstance(payload.get("schema_id"), str) else None,
        max_examples=payload.get("max_examples") if isinstance(payload.get("max_examples"), int) else None,
        notes=payload.get("notes") if isinstance(payload.get("notes"), dict) else None,
        raw=dict(payload),
        source_path=source_path,
        error=None,
    )
    return snap


def read_eval_run_pointer(path: Pathish) -> EvalRunPointerSnapshot:
    p = Path(path).resolve()
    try:
        payload = read_json_internal(EVAL_RUN_POINTER_SCHEMA, p)
        return parse_eval_run_pointer(payload, source_path=str(p))
    except Exception as e:
        # fail-closed for pointer usage (callers decide semantics)
        return EvalRunPointerSnapshot(
            ok=False,
            schema_version="",
            generated_at="",
            task="",
            run_id="",
            store="",
            run_dir=None,
            summary_path=None,
            base_url=None,
            model_override=None,
            schema_id=None,
            max_examples=None,
            notes=None,
            raw={},
            source_path=str(p),
            error=f"eval_run_pointer_parse_error: {type(e).__name__}: {e}",
        )


def write_eval_run_pointer(path: Pathish, payload: Dict[str, Any]) -> Path:
    """
    Validate + atomic write.
    """
    return atomic_write_json_internal(EVAL_RUN_POINTER_SCHEMA, path, payload)


def default_eval_out_path(task: str) -> Path:
    """
    Conventional location: eval_out/<task>/latest.json
    """
    return (Path("eval_out") / task / "latest.json").resolve()