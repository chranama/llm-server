from __future__ import annotations

import math

from llm_eval.reports import writer


def test_render_report_md_examples_failures_sorted_first(monkeypatch):
    monkeypatch.setattr(writer, "_utc_now_iso", lambda: "2020-01-01T00:00:00Z", raising=True)

    results = [
        {"ok": True, "status_code": 200, "error_code": None, "latency_ms": 12.0, "id": "b"},
        {"ok": False, "status_code": 422, "error_code": "schema_validation_failed", "latency_ms": 34.0, "id": "a"},
        {"ok": False, "status_code": 500, "error_code": "internal_error", "latency_ms": 56.0, "id": "c"},
    ]

    md = writer.render_report_md(
        task="task",
        run_id="rid",
        base_url="http://base",
        summary={},
        results=results,
        runner_report_text=None,
        max_example_rows=10,
    )

    # Table header present
    assert "| ok | status | error_code | latency_ms | id |" in md

    # Fail rows should appear before ok rows.
    # Also: within fail/ok grouping, sort by id/doc_id string.
    fail_a = "| `False` | `422` | `schema_validation_failed` | `34.0ms` | `a` |"
    ok_b = "| `True` | `200` | `None` | `12.0ms` | `b` |"

    assert md.index(fail_a) < md.index(ok_b)


def test_render_report_md_metrics_rate_and_latency_formatting(monkeypatch):
    monkeypatch.setattr(writer, "_utc_now_iso", lambda: "2020-01-01T00:00:00Z", raising=True)

    md = writer.render_report_md(
        task="task",
        run_id="rid",
        base_url="http://base",
        summary={
            "schema_validity_rate": 0.9,
            "latency_p95_ms": 123.456,
            "accuracy": 0.1,
        },
        results=[],
        runner_report_text=None,
    )

    assert "| `schema_validity_rate` | `90.00%` |" in md
    assert "| `latency_p95_ms` | `123.5ms` |" in md
    assert "| `accuracy` | `0.1` |" in md


def test_safe_float_and_pct_handle_nan():
    assert writer._safe_float(None) is None
    assert writer._safe_float("nope") is None
    assert writer._safe_float(float("nan")) is None

    # _pct returns None on NaN
    assert writer._pct(float("nan")) is None

    # but returns formatted percent otherwise
    assert writer._pct(0.125) == "12.50%"


def test_render_report_text_handles_nan_rate_as_raw(monkeypatch):
    # In render_report_text fallback:
    # for *_rate keys, writer tries _pct(v) or v (raw) if pct can't format.
    nan = float("nan")
    txt = writer.render_report_text(
        task="t",
        run_id="r",
        base_url="u",
        summary={"schema_validity_rate": nan},
        runner_report_text=None,
    )
    # raw nan survives (stringification) because _pct returns None, fallback uses v
    assert "schema_validity_rate=nan" in txt.lower()