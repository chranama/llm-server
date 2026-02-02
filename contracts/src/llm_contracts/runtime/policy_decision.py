# contracts/src/llm_contracts/runtime/policy_decision.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from llm_contracts.schema import atomic_write_json_internal, read_json_internal, validate_internal

Pathish = Union[str, Path]

POLICY_DECISION_SCHEMA = "policy_decision_v1.schema.json"
POLICY_DECISION_SCHEMA_VERSION = "policy_decision_v1"


@dataclass(frozen=True)
class PolicyDecisionSnapshot:
    """
    Stable minimal backend-facing snapshot.

    `raw` retains the full validated payload for forward compatibility.
    """
    ok: bool
    schema_version: str
    generated_at: str

    policy: str
    status: str  # allow|deny|unknown

    enable_extract: bool
    contract_errors: int

    # optional traceability
    model_id: Optional[str]
    thresholds_profile: Optional[str]
    thresholds_version: Optional[str]
    eval_run_dir: Optional[str]
    eval_task: Optional[str]
    eval_run_id: Optional[str]

    raw: Dict[str, Any]
    source_path: Optional[str] = None
    error: Optional[str] = None


def _opt_str(payload: Dict[str, Any], key: str) -> Optional[str]:
    v = payload.get(key)
    return v if isinstance(v, str) and v.strip() else None


def parse_policy_decision(payload: Dict[str, Any], *, source_path: Optional[str] = None) -> PolicyDecisionSnapshot:
    """
    Parse + validate the policy decision artifact.

    Contract responsibilities live here (llm_contracts):
      - schema validation
      - schema_version acceptance
      - semantic fail-closed invariants
    """
    validate_internal(POLICY_DECISION_SCHEMA, payload)

    schema_version = str(payload["schema_version"]).strip()
    if schema_version != POLICY_DECISION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported policy decision schema_version: {schema_version}")

    generated_at = str(payload["generated_at"]).strip()
    policy = str(payload["policy"]).strip()
    status = str(payload["status"]).strip()

    ok = bool(payload["ok"])
    enable_extract = bool(payload["enable_extract"])
    contract_errors = int(payload["contract_errors"])

    # Fail-closed invariants you rely on elsewhere:
    # - non-ok => extract disabled
    # - any contract_errors => ok=false + extract disabled
    # - deny/unknown => ok=false + extract disabled
    if contract_errors > 0:
        ok = False
        enable_extract = False
    if status in ("deny", "unknown"):
        ok = False
        enable_extract = False
    if not ok:
        enable_extract = False

    return PolicyDecisionSnapshot(
        ok=ok,
        schema_version=schema_version,
        generated_at=generated_at,
        policy=policy,
        status=status,
        enable_extract=enable_extract,
        contract_errors=contract_errors,
        model_id=_opt_str(payload, "model_id"),
        thresholds_profile=_opt_str(payload, "thresholds_profile"),
        thresholds_version=_opt_str(payload, "thresholds_version"),
        eval_run_dir=_opt_str(payload, "eval_run_dir"),
        eval_task=_opt_str(payload, "eval_task"),
        eval_run_id=_opt_str(payload, "eval_run_id"),
        raw=dict(payload),
        source_path=source_path,
        error=None,
    )


def read_policy_decision(path: Pathish) -> PolicyDecisionSnapshot:
    """
    Read + validate + parse a policy decision artifact.

    On any error: return ok=False with a populated error string (fail closed).
    """
    p = Path(path).resolve()
    try:
        payload = read_json_internal(POLICY_DECISION_SCHEMA, p)
        return parse_policy_decision(payload, source_path=str(p))
    except Exception as e:
        # Mark as a contract failure signal for downstream policy gating / visibility.
        return PolicyDecisionSnapshot(
            ok=False,
            schema_version="",
            generated_at="",
            policy="",
            status="unknown",
            enable_extract=False,
            contract_errors=1,
            model_id=None,
            thresholds_profile=None,
            thresholds_version=None,
            eval_run_dir=None,
            eval_task=None,
            eval_run_id=None,
            raw={},
            source_path=str(p),
            error=f"policy_decision_parse_error: {type(e).__name__}: {e}",
        )


def write_policy_decision(path: Pathish, payload: Dict[str, Any]) -> Path:
    """
    Validate + atomic write.
    """
    return atomic_write_json_internal(POLICY_DECISION_SCHEMA, path, payload)


def default_policy_out_path() -> Path:
    """
    Conventional location: policy_out/latest.json
    (Callers can override via env in their own layer)
    """
    return (Path("policy_out") / "latest.json").resolve()