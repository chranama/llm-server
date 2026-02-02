# policy/src/llm_policy/reports/writer.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional

from pydantic import BaseModel

from llm_policy.types.decision import Decision


def _to_mapping(x: Any) -> Mapping[str, Any]:
    """
    Normalize DecisionReason/DecisionWarning objects (pydantic) or dicts into a Mapping.

    reports/ is HUMAN OUTPUT ONLY:
      - text for terminals/logs
      - markdown for docs/PR comments
    """
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, BaseModel):
        # pydantic v2
        try:
            return x.model_dump()
        except Exception:
            pass
    # last resort: try attribute access
    out: Dict[str, Any] = {}
    for k in ("code", "message", "context", "extra"):
        if hasattr(x, k):
            out[k] = getattr(x, k)
    return out


def _iter_issues(items: Optional[Iterable[Any]]) -> Iterable[Mapping[str, Any]]:
    for it in items or []:
        yield _to_mapping(it)


def render_decision_text(decision: Decision) -> str:
    """
    Human-oriented single-decision summary for terminals.

    NOTE: This is NOT the runtime ingestion artifact.
    Runtime ingestion should use DecisionArtifactV1 (io/ or artifacts/).
    """
    lines: list[str] = []
    lines.append(f"policy={decision.policy}")

    if getattr(decision, "thresholds_profile", None):
        lines.append(f"thresholds_profile={decision.thresholds_profile}")

    # keep legacy UX
    if getattr(decision, "enable_extract", None) is not None:
        lines.append(f"enable_extract={bool(decision.enable_extract)}")

    if getattr(decision, "status", None) is not None:
        lines.append(f"status={decision.status}")

    lines.append(f"ok={decision.ok() if hasattr(decision, 'ok') else 'unknown'}")

    ce = int(getattr(decision, "contract_errors", 0) or 0)
    cw = int(getattr(decision, "contract_warnings", 0) or 0)
    if ce or cw:
        lines.append(f"contract_errors={ce}")
        lines.append(f"contract_warnings={cw}")

    reasons = list(_iter_issues(getattr(decision, "reasons", None)))
    if reasons:
        lines.append("")
        lines.append("REASONS:")
        for r in reasons:
            code = r.get("code", "reason")
            msg = r.get("message", "")
            lines.append(f"- {code}: {msg}")

    warnings = list(_iter_issues(getattr(decision, "warnings", None)))
    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in warnings:
            code = w.get("code", "warning")
            msg = w.get("message", "")
            lines.append(f"- {code}: {msg}")

    metrics = getattr(decision, "metrics", None) or {}
    if metrics:
        lines.append("")
        lines.append("METRICS:")
        for k in sorted(metrics.keys()):
            lines.append(f"- {k}: {metrics[k]}")

    return "\n".join(lines) + "\n"


def render_decision_md(decision: Decision) -> str:
    """
    Human-oriented markdown report for docs / PRs / GitHub comments.

    NOTE: This is NOT the runtime ingestion artifact.
    """
    lines: list[str] = []
    lines.append(f"# Policy Decision: `{decision.policy}`")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| ok | `{decision.ok() if hasattr(decision,'ok') else 'unknown'}` |")

    if getattr(decision, "status", None) is not None:
        lines.append(f"| status | `{decision.status}` |")

    if getattr(decision, "thresholds_profile", None):
        lines.append(f"| thresholds_profile | `{decision.thresholds_profile}` |")

    if getattr(decision, "enable_extract", None) is not None:
        lines.append(f"| enable_extract | `{bool(decision.enable_extract)}` |")

    ce = int(getattr(decision, "contract_errors", 0) or 0)
    cw = int(getattr(decision, "contract_warnings", 0) or 0)
    if ce or cw:
        lines.append(f"| contract_errors | `{ce}` |")
        lines.append(f"| contract_warnings | `{cw}` |")

    reasons = list(_iter_issues(getattr(decision, "reasons", None)))
    if reasons:
        lines.append("")
        lines.append("## Reasons")
        for r in reasons:
            code = r.get("code", "reason")
            msg = r.get("message", "")
            lines.append(f"- **{code}** — {msg}")

    warnings = list(_iter_issues(getattr(decision, "warnings", None)))
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        for w in warnings:
            code = w.get("code", "warning")
            msg = w.get("message", "")
            lines.append(f"- **{code}** — {msg}")

    metrics = getattr(decision, "metrics", None) or {}
    if metrics:
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(metrics, ensure_ascii=False, indent=2))
        lines.append("```")

    return "\n".join(lines) + "\n"