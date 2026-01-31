# backend/src/llm_server/services/policy_decisions.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PolicyDecisionSnapshot:
    """
    Minimal runtime representation.
    Keep this intentionally tiny: backend ingests decisions, doesn't compute them.
    """
    ok: bool
    model_id: Optional[str]
    enable_extract: Optional[bool]
    raw: Dict[str, Any]
    source_path: Optional[str]
    error: Optional[str]


# ----------------------------
# Internal helpers
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_optional_bool(x: Any) -> Optional[bool]:
    """
    Strict-ish boolean parsing to avoid footguns like bool("false") == True.

    Accepts:
      - bool
      - int/float 0 or 1
      - strings: true/false, 1/0, yes/no, on/off (case-insensitive)
    Returns:
      - True/False
      - None if unparseable
    """
    if x is None:
        return None

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)):
        # only accept exact 0/1 to avoid surprises
        if x == 0:
            return False
        if x == 1:
            return True
        return None

    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
        return None

    return None


def _normalize_model_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _status_denies(status: Any) -> bool:
    if not isinstance(status, str):
        return False
    s = status.strip().lower()
    return s in ("deny", "unknown")


# ----------------------------
# Public API
# ----------------------------

def load_policy_decision_from_env() -> PolicyDecisionSnapshot:
    """
    Load a policy decision JSON from POLICY_DECISION_PATH.

    Semantics:
      - If POLICY_DECISION_PATH is unset/empty => no override (ok=True, enable_extract=None)
      - If set but file missing/invalid => fail-closed (ok=False, enable_extract=False)
      - If set and JSON loads but indicates non-ok (status deny/unknown OR contract_errors>0) => fail-closed
      - If enable_extract is present but unparseable (e.g. "maybe") => treat as invalid => fail-closed
    """
    path_s = os.getenv("POLICY_DECISION_PATH", "").strip()
    if not path_s:
        # No policy configured: treat as "no override"
        return PolicyDecisionSnapshot(
            ok=True,
            model_id=None,
            enable_extract=None,
            raw={},
            source_path=None,
            error=None,
        )

    p = Path(path_s)
    if not p.exists():
        return PolicyDecisionSnapshot(
            ok=False,
            model_id=None,
            enable_extract=False,  # fail-closed
            raw={},
            source_path=str(p),
            error="policy_decision_missing",
        )

    try:
        raw = _read_json(p)
        if not isinstance(raw, dict):
            raise ValueError("policy decision JSON root must be an object")

        model_id = _normalize_model_id(raw.get("model_id"))

        # Parse enable_extract safely.
        enable_extract_raw = raw.get("enable_extract", None)
        enable_extract = _parse_optional_bool(enable_extract_raw)

        # If the key exists but can't be parsed, treat file as invalid (fail-closed).
        if enable_extract_raw is not None and enable_extract is None:
            return PolicyDecisionSnapshot(
                ok=False,
                model_id=model_id,
                enable_extract=False,  # fail-closed
                raw=raw if isinstance(raw, dict) else {},
                source_path=str(p),
                error="policy_decision_invalid_enable_extract",
            )

        status = raw.get("status", None)
        contract_errors = raw.get("contract_errors", 0) or 0

        ok = True
        try:
            if int(contract_errors) > 0:
                ok = False
        except Exception:
            # If contract_errors is malformed, treat as invalid file => fail-closed
            ok = False

        if _status_denies(status):
            ok = False

        if not ok:
            # fail-closed if decision not ok
            enable_extract = False if enable_extract is None else bool(enable_extract)

        return PolicyDecisionSnapshot(
            ok=ok,
            model_id=model_id,
            enable_extract=enable_extract,
            raw=raw,
            source_path=str(p),
            error=None if ok else "policy_decision_not_ok",
        )

    except Exception as e:
        return PolicyDecisionSnapshot(
            ok=False,
            model_id=None,
            enable_extract=False,  # fail-closed
            raw={},
            source_path=str(p),
            error=f"policy_decision_parse_error: {type(e).__name__}: {e}",
        )


def get_policy_snapshot(request) -> PolicyDecisionSnapshot:
    """
    Get cached snapshot from app.state if present; else load and cache.
    """
    snap = getattr(request.app.state, "policy_snapshot", None)
    if isinstance(snap, PolicyDecisionSnapshot):
        return snap
    snap = load_policy_decision_from_env()
    request.app.state.policy_snapshot = snap
    return snap


def reload_policy_snapshot(request) -> PolicyDecisionSnapshot:
    """
    Force reload from disk and overwrite app.state cache.
    """
    snap = load_policy_decision_from_env()
    request.app.state.policy_snapshot = snap
    return snap


def policy_capability_overrides(
    model_id: str,
    *,
    request,
) -> Optional[Dict[str, bool]]:
    """
    Returns an override mapping for known capability keys, or None if no override applies.

    Semantics:
      - If policy file is configured but invalid => fail-closed extract for ALL models.
      - If decision is for a specific model_id and doesn't match => no override.
      - If enable_extract is provided => override extract accordingly.
    """
    snap = get_policy_snapshot(request)

    # If policy file is configured but invalid => fail-closed extract for all models.
    if snap.source_path and not snap.ok:
        return {"extract": False}

    # If decision is for a specific model and doesn't match, do not apply.
    if snap.model_id and snap.model_id != model_id:
        return None

    out: Dict[str, bool] = {}
    if snap.enable_extract is not None:
        out["extract"] = bool(snap.enable_extract)

    return out or None