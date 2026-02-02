# server/src/llm_server/io/policy_decisions.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from llm_contracts.runtime.policy_decision import (
    PolicyDecisionSnapshot as ContractsPolicyDecisionSnapshot,
    read_policy_decision,
)


@dataclass(frozen=True)
class PolicyDecisionSnapshot:
    """
    Backend-local minimal runtime representation.

    NOTE:
    - This intentionally stays tiny and stable for backend usage.
    - Raw is retained for forward compatibility/debugging.
    """
    ok: bool
    model_id: Optional[str]
    enable_extract: Optional[bool]
    raw: Dict[str, Any]
    source_path: Optional[str]
    error: Optional[str]


def _to_backend_snapshot(s: ContractsPolicyDecisionSnapshot) -> PolicyDecisionSnapshot:
    """
    Convert contracts snapshot -> backend tiny snapshot.
    """
    # Backend semantics:
    # - If the file exists but is non-ok, enable_extract should be False (fail-closed).
    # - If no policy configured, we represent as ok=True, enable_extract=None (no override).
    return PolicyDecisionSnapshot(
        ok=bool(s.ok),
        model_id=s.model_id,
        enable_extract=bool(s.enable_extract) if s.ok else False,
        raw=dict(s.raw or {}),
        source_path=s.source_path,
        error=s.error,
    )


def load_policy_decision_from_env() -> PolicyDecisionSnapshot:
    """
    Load a policy decision JSON from POLICY_DECISION_PATH.

    Semantics (unchanged, but now enforced by llm_contracts schema + parser):
      - If POLICY_DECISION_PATH is unset/empty => no override (ok=True, enable_extract=None)
      - If set but file missing => fail-closed (ok=False, enable_extract=False)
      - If set and file invalid/unparseable => fail-closed (ok=False, enable_extract=False)
      - If set and decision indicates non-ok => fail-closed (ok=False, enable_extract=False)
    """
    path_s = os.getenv("POLICY_DECISION_PATH", "").strip()
    if not path_s:
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
            enable_extract=False,
            raw={},
            source_path=str(p),
            error="policy_decision_missing",
        )

    snap = read_policy_decision(p)

    # If parse failed, read_policy_decision already returns ok=False and enable_extract=False.
    # Convert to backend’s tiny snapshot shape.
    return _to_backend_snapshot(snap)


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


def policy_capability_overrides(model_id: str, *, request) -> Optional[Dict[str, bool]]:
    """
    Returns an override mapping for known capability keys, or None if no override applies.

    Semantics:
      - If policy file is configured but invalid/non-ok => fail-closed extract for ALL models.
      - If decision is for a specific model_id and doesn't match => no override.
      - If decision is ok and enable_extract is present => override extract accordingly.
    """
    snap = get_policy_snapshot(request)

    # If policy file is configured but invalid/non-ok => fail-closed extract for all models.
    if snap.source_path and not snap.ok:
        return {"extract": False}

    # If decision is for a specific model and doesn't match, do not apply.
    if snap.model_id and snap.model_id != model_id:
        return None

    out: Dict[str, bool] = {}
    if snap.enable_extract is not None:
        out["extract"] = bool(snap.enable_extract)

    return out or None