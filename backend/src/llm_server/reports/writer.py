# backend/src/llm_server/reports/writer.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Mapping


def _iso(dt: Any) -> str:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return "" if dt is None else str(dt)


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    return x


def render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_jsonable(dict(payload)), ensure_ascii=False, indent=2) + "\n"


def render_text_kv(title: str, kv: Mapping[str, Any]) -> str:
    lines: list[str] = [title]
    lines.append("-" * len(title))
    for k, v in kv.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {_iso(v) if isinstance(v, datetime) else v}")
    return "\n".join(lines) + "\n"


def render_md_kv(title: str, kv: Mapping[str, Any]) -> str:
    lines: list[str] = [f"# {title}", ""]
    lines.append("| Key | Value |")
    lines.append("|---|---|")
    for k, v in kv.items():
        vv = _iso(v) if isinstance(v, datetime) else ("" if v is None else str(v))
        lines.append(f"| `{k}` | {vv} |")
    lines.append("")
    return "\n".join(lines)


def render_md_table(title: str, *, columns: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    lines: list[str] = [f"## {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        vals: list[str] = []
        for c in columns:
            v = r.get(c)
            if isinstance(v, datetime):
                vals.append(v.isoformat())
            elif v is None:
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def render_admin_summary(*, stats_payload: Mapping[str, Any], per_model: list[Mapping[str, Any]], fmt: str) -> str:
    """
    A lightweight "presentation output" that can be served from llm_server.

    stats_payload: global fields (window_days, since, totals, avg_latency)
    per_model: list of model rows
    """
    title = "LLM Server Admin Summary"

    if fmt == "json":
        return render_json({"title": title, "stats": dict(stats_payload), "per_model": list(per_model)})

    if fmt == "md":
        out = []
        out.append(render_md_kv(title, stats_payload))
        out.append(render_md_table("Per-model", columns=list(per_model[0].keys()) if per_model else ["model_id"], rows=per_model))
        return "\n".join(out).strip() + "\n"

    # text
    buf = []
    buf.append(render_text_kv(title, stats_payload).rstrip())
    buf.append("")
    buf.append("Per-model")
    buf.append("-" * len("Per-model"))
    if not per_model:
        buf.append("(no data)")
    else:
        cols = list(per_model[0].keys())
        for row in per_model:
            parts = [f"{c}={row.get(c)}" for c in cols]
            buf.append("  " + " ".join(parts))
    return "\n".join(buf).strip() + "\n"