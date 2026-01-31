# src/llm_policy/policies/regressions.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm_policy.types.decision import Decision, DecisionIssue
from llm_policy.types.eval_artifact import EvalArtifact


def _issue(code: str, message: str, context: Optional[dict[str, Any]] = None) -> DecisionIssue:
    return DecisionIssue(code=code, message=message, context=context or {})


def detect_regressions(
    current: EvalArtifact,
    baseline: EvalArtifact,
    *,
    thresholds_profile: Optional[str] = None,
) -> Decision:
    """
    v0 regression checker:
      - compares a few key summary metrics
      - emits warnings (does not hard-fail) unless something is catastrophically worse

    This is intentionally conservative for initial rollout.
    """
    c = current.summary
    b = baseline.summary

    warnings: list[DecisionIssue] = []
    reasons: list[DecisionIssue] = []

    def _cmp(name: str, cur: Optional[float], base: Optional[float], *, tol_drop: float) -> None:
        if cur is None or base is None:
            return
        drop = float(base) - float(cur)
        if drop > tol_drop:
            warnings.append(
                _issue(
                    "regression",
                    f"{name} dropped by {drop:.4f} (baseline={base}, current={cur})",
                    {"metric": name, "baseline": base, "current": cur, "drop": drop, "tol_drop": tol_drop},
                )
            )

    # Quality regressions
    _cmp("schema_validity_rate", c.schema_validity_rate, b.schema_validity_rate, tol_drop=0.02)

    # Optional metrics: only compare if present
    _cmp("required_present_rate", c.required_present_rate, b.required_present_rate, tol_drop=0.02)
    _cmp(
        "doc_required_exact_match_rate",
        c.doc_required_exact_match_rate,
        b.doc_required_exact_match_rate,
        tol_drop=0.02,
    )

    # Catastrophic: if went from mostly-ok to mostly-failing
    if (b.n_ok or 0) > 0 and (c.n_ok or 0) == 0 and (c.n_total or 0) >= 10:
        reasons.append(
            _issue(
                "catastrophic_regression",
                "Baseline had successes but current run has 0 ok results",
                {"baseline_n_ok": b.n_ok, "current_n_ok": c.n_ok, "current_n_total": c.n_total},
            )
        )

    # Regressions annotate only (do not enable extract).
    # If you later want regressions to block, wire it into decide-and-patch explicitly.
    return Decision(
        policy="regressions",
        thresholds_profile=thresholds_profile,
        enable_extract=False,
        reasons=reasons,
        warnings=warnings,
        metrics={
            "baseline_run_id": getattr(b, "run_id", None),
            "current_run_id": getattr(c, "run_id", None),
        },
        eval_task=str(getattr(c, "task", "") or ""),
        eval_run_id=str(getattr(c, "run_id", "") or ""),
        eval_run_dir=str(getattr(c, "run_dir", "") or ""),
    )