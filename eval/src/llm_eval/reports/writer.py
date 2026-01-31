# llm_eval/reports/writer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class RenderedReports:
    """
    Pure rendering outputs (no filesystem writes).

    - text: canonical CLI/log report
    - md:   PR/docs-friendly markdown report
    - json_summary: json string of summary (optional convenience)
    """
    text: str
    md: str
    json_summary: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except Exception:
        return None


def _pct(x: Any) -> Optional[str]:
    f = _safe_float(x)
    if f is None:
        return None
    return f"{f * 100.0:.2f}%"


def _ms(x: Any) -> Optional[str]:
    f = _safe_float(x)
    if f is None:
        return None
    return f"{f:.1f}ms"


def _pick_metrics(summary: dict[str, Any]) -> list[tuple[str, Any]]:
    """
    Stable, cross-task-friendly metric selection.
    We intentionally don't assume every task has the same metric keys.
    """
    keys = [
        # extraction-style
        "schema_validity_rate",
        "doc_required_exact_match_rate",
        "required_present_rate",
        "answerable_exact_match_rate",
        "unanswerable_accuracy",
        "combined_score",
        # generic classification
        "precision",
        "recall",
        "f1",
        "accuracy",
        # latency
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ]
    out: list[tuple[str, Any]] = []
    for k in keys:
        if k in summary and summary.get(k) is not None:
            out.append((k, summary.get(k)))
    return out


def _format_kv_lines(summary: dict[str, Any]) -> list[str]:
    task = summary.get("task")
    run_id = summary.get("run_id")
    base_url = summary.get("base_url")

    lines: list[str] = []
    if task is not None:
        lines.append(f"task={task}")
    if run_id is not None:
        lines.append(f"run_id={run_id}")
    if base_url is not None:
        lines.append(f"base_url={base_url}")

    for k in ("dataset", "split", "schema_id", "model_override", "max_examples"):
        if summary.get(k) is not None:
            lines.append(f"{k}={summary.get(k)}")

    # metrics
    for k, v in _pick_metrics(summary):
        if k.endswith("_rate"):
            vv = _pct(v) or v
        elif k.startswith("latency_") and k.endswith("_ms"):
            vv = _ms(v) or v
        else:
            vv = v
        lines.append(f"{k}={vv}")

    return lines


def render_report_text(
    *,
    task: str,
    run_id: str,
    base_url: str,
    summary: dict[str, Any],
    runner_report_text: Optional[str] = None,
) -> str:
    """
    Canonical report.txt text.
    Prefer runner_report_text if provided (it often includes task-specific formatting),
    else fall back to a stable summary-driven report.
    """
    if isinstance(runner_report_text, str) and runner_report_text.strip():
        # Ensure it ends with newline for nice CLI/file behavior
        s = runner_report_text.strip("\n") + "\n"
        return s

    # fallback
    merged = dict(summary)
    merged.setdefault("task", task)
    merged.setdefault("run_id", run_id)
    merged.setdefault("base_url", base_url)

    lines = _format_kv_lines(merged)
    return "\n".join(lines) + "\n"


def render_report_md(
    *,
    task: str,
    run_id: str,
    base_url: str,
    summary: dict[str, Any],
    results: Optional[Iterable[dict[str, Any]]] = None,
    runner_report_text: Optional[str] = None,
    max_example_rows: int = 10,
) -> str:
    """
    Canonical report.md markdown.
    - includes top-line metadata
    - includes a metrics table if possible
    - optionally includes a small 'worst examples' table (best-effort)
    """
    merged = dict(summary)
    merged.setdefault("task", task)
    merged.setdefault("run_id", run_id)
    merged.setdefault("base_url", base_url)

    lines: list[str] = []
    lines.append(f"# Eval Report: `{merged.get('task')}`")
    lines.append("")
    lines.append(f"- **run_id:** `{merged.get('run_id')}`")
    lines.append(f"- **base_url:** `{merged.get('base_url')}`")
    if merged.get("model_override") is not None:
        lines.append(f"- **model_override:** `{merged.get('model_override')}`")
    if merged.get("dataset") is not None:
        lines.append(f"- **dataset:** `{merged.get('dataset')}`")
    if merged.get("split") is not None:
        lines.append(f"- **split:** `{merged.get('split')}`")
    if merged.get("schema_id") is not None:
        lines.append(f"- **schema_id:** `{merged.get('schema_id')}`")
    if merged.get("max_examples") is not None:
        lines.append(f"- **max_examples:** `{merged.get('max_examples')}`")
    lines.append(f"- **generated_at:** `{_utc_now_iso()}`")
    lines.append("")

    # Metrics table
    metrics = _pick_metrics(merged)
    if metrics:
        lines.append("## Summary metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for k, v in metrics:
            if k.endswith("_rate"):
                vv = _pct(v) or v
            elif k.startswith("latency_") and k.endswith("_ms"):
                vv = _ms(v) or v
            else:
                vv = v
            lines.append(f"| `{k}` | `{vv}` |")
        lines.append("")

    # Include runner report text as a details section if it exists
    if isinstance(runner_report_text, str) and runner_report_text.strip():
        lines.append("## Runner details")
        lines.append("")
        # keep it readable without fenced code blocks; markdown readers handle this fine
        # but we still indent as a blockquote-ish style to avoid huge headings
        for ln in runner_report_text.strip().splitlines():
            lines.append(f"> {ln}")
        lines.append("")

    # Example rows (best-effort)
    rows: list[dict[str, Any]] = []
    if results is not None:
        for r in results:
            if isinstance(r, dict):
                rows.append(r)

    # show failing examples first if 'ok' is present
    def _is_fail(r: dict[str, Any]) -> bool:
        ok = r.get("ok")
        if ok is None:
            return False
        return not bool(ok)

    if rows:
        rows_sorted = sorted(rows, key=lambda r: (not _is_fail(r), str(r.get("doc_id") or r.get("id") or "")))
        preview = rows_sorted[: max_example_rows]

        lines.append(f"## Example results (first {len(preview)})")
        lines.append("")
        lines.append("| ok | status | error_code | latency_ms | id |")
        lines.append("|---:|---:|---|---:|---|")
        for r in preview:
            ok = r.get("ok")
            status = r.get("status_code")
            code = r.get("error_code")
            lat = r.get("latency_ms")
            rid = r.get("doc_id") or r.get("id") or ""
            lines.append(f"| `{ok}` | `{status}` | `{code}` | `{_ms(lat) or lat}` | `{rid}` |")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_reports_bundle(
    *,
    task: str,
    run_id: str,
    base_url: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
    runner_report_text: Optional[str] = None,
) -> RenderedReports:
    text = render_report_text(
        task=task,
        run_id=run_id,
        base_url=base_url,
        summary=summary,
        runner_report_text=runner_report_text,
    )
    md = render_report_md(
        task=task,
        run_id=run_id,
        base_url=base_url,
        summary=summary,
        results=results or [],
        runner_report_text=runner_report_text,
    )
    json_summary = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    return RenderedReports(text=text, md=md, json_summary=json_summary)