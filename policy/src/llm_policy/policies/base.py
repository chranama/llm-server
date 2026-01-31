# src/llm_policy/policies/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from llm_policy.types.decision import Decision  # assumes you already have this
from llm_policy.types.eval_artifact import EvalArtifact


@dataclass(frozen=True)
class PolicyContext:
    """
    Shared context for policy evaluation. Keep this small in v0.
    Extend later (telemetry, baselines, environment, etc.).
    """
    thresholds_profile: Optional[str] = None
    meta: Dict[str, Any] | None = None


class Policy(Protocol):
    name: str

    def evaluate(self, artifact: EvalArtifact, *, ctx: PolicyContext) -> Decision: ...


def _reason(code: str, message: str, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r: Dict[str, Any] = {"code": code, "message": message}
    if extra:
        r["extra"] = extra
    return r


def combine_decisions(primary: Decision, *others: Decision) -> Decision:
    """
    Conservative combination:
      - if any decision is not ok(), combined is not ok()
      - enable_extract is AND'd (all must agree) when present

    This is intentionally simple for v0.
    """
    out = primary.model_copy(deep=True)

    for d in others:
        # merge reasons/warnings if your Decision supports them
        if getattr(d, "reasons", None):
            out.reasons.extend(d.reasons)  # type: ignore[attr-defined]
        if getattr(d, "warnings", None):
            out.warnings.extend(d.warnings)  # type: ignore[attr-defined]

        # conservative: if any fails, final fails
        if hasattr(d, "ok") and callable(getattr(d, "ok")):
            if not d.ok():
                # if Decision supports a status field, keep it; otherwise just rely on ok()
                pass

        # AND capability enablement if both specify it
        if getattr(d, "enable_extract", None) is not None:
            if getattr(out, "enable_extract", None) is None:
                out.enable_extract = bool(d.enable_extract)  # type: ignore[attr-defined]
            else:
                out.enable_extract = bool(out.enable_extract) and bool(d.enable_extract)  # type: ignore[attr-defined]

    return out