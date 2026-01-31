# src/llm_policy/policies/health_gate.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm_policy.types.decision import Decision
from llm_policy.types.eval_artifact import EvalArtifact
from llm_policy.types.thresholds import ExtractThresholds


def _get_int(d: Dict[str, Any], k: str) -> int:
    v = d.get(k)
    try:
        return int(v)
    except Exception:
        return 0


def _rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 1.0
    return float(numer) / float(denom)


def health_gate_from_eval(
    artifact: EvalArtifact,
    *,
    thresholds: ExtractThresholds,
    thresholds_profile: Optional[str] = None,
) -> Decision:
    """
    A hard safety gate based on operational health.

    v0 logic:
      - if error rate too high, block extract enablement
      - specifically track 5xx + transport errors when available

    This is intended to be used *before* quality-based enablement.
    """
    s = artifact.summary
    n_total = int(s.n_total or 0)

    status_counts = s.status_code_counts or {}
    err_counts = s.error_code_counts or {}

    n_5xx = 0
    for code_str, count in status_counts.items():
        try:
            c = int(code_str)
        except Exception:
            continue
        if 500 <= c <= 599:
            n_5xx += int(count or 0)

    n_transport = int(err_counts.get("transport_error", 0) or 0)
    n_any_error = n_total - int(s.n_ok or 0)

    error_rate = _rate(n_any_error, n_total)
    r_5xx = _rate(n_5xx, n_total)
    r_transport = _rate(n_transport, n_total)

    reasons: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # sample size sanity
    if n_total < thresholds.min_n_total:
        warnings.append(
            {
                "code": "insufficient_sample_size",
                "message": f"n_total={n_total} below min_n_total={thresholds.min_n_total}; decision is low-confidence",
                "extra": {"n_total": n_total, "min_n_total": thresholds.min_n_total, "profile": thresholds_profile},
            }
        )

    blocked = False

    if error_rate > thresholds.max_error_rate:
        blocked = True
        reasons.append(
            {
                "code": "error_rate_too_high",
                "message": f"error_rate={error_rate:.3f} exceeds max_error_rate={thresholds.max_error_rate:.3f}",
                "extra": {"n_any_error": n_any_error, "n_total": n_total},
            }
        )

    if r_5xx > thresholds.max_5xx_rate:
        blocked = True
        reasons.append(
            {
                "code": "server_error_rate_too_high",
                "message": f"5xx_rate={r_5xx:.3f} exceeds max_5xx_rate={thresholds.max_5xx_rate:.3f}",
                "extra": {"n_5xx": n_5xx, "n_total": n_total, "status_code_counts": status_counts},
            }
        )

    if r_transport > thresholds.max_transport_error_rate:
        blocked = True
        reasons.append(
            {
                "code": "transport_error_rate_too_high",
                "message": f"transport_rate={r_transport:.3f} exceeds max_transport_error_rate={thresholds.max_transport_error_rate:.3f}",
                "extra": {"n_transport": n_transport, "n_total": n_total, "error_code_counts": err_counts},
            }
        )

    d = Decision(
        policy="health_gate",
        thresholds_profile=thresholds_profile,
        enable_extract=(not blocked),
        reasons=reasons,
        warnings=warnings,
        metrics={
            "n_total": n_total,
            "n_any_error": n_any_error,
            "n_5xx": n_5xx,
            "n_transport": n_transport,
            "error_rate": error_rate,
            "rate_5xx": r_5xx,
            "rate_transport": r_transport,
        },
    )
    return d