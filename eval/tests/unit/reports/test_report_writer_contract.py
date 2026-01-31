from __future__ import annotations

import json

import pytest

from llm_eval.reports import writer


def test_render_reports_bundle_contract_basic():
    summary = {
        "schema_validity_rate": 0.9,
        "latency_p50_ms": 12.3,
        "dataset": "dummy",
    }

    out = writer.render_reports_bundle(
        task="extraction",
        run_id="run_123",
        base_url="http://localhost:8000",
        summary=summary,
        results=[],
        runner_report_text=None,
    )

    assert isinstance(out.text, str) and out.text.strip()
    assert isinstance(out.md, str) and out.md.strip()
    assert isinstance(out.json_summary, str) and out.json_summary.strip()

    # json_summary must be valid JSON and match provided summary (writer doesn't inject extra keys there)
    parsed = json.loads(out.json_summary)
    assert parsed == summary


def test_render_report_text_prefers_runner_report_text_and_adds_newline():
    s = writer.render_report_text(
        task="t",
        run_id="r",
        base_url="u",
        summary={"accuracy": 1.0},
        runner_report_text="hello\nworld\n",
    )
    assert s == "hello\nworld\n"  # exactly one trailing newline


def test_render_report_text_fallback_contains_expected_kvs():
    s = writer.render_report_text(
        task="task_x",
        run_id="run_x",
        base_url="http://x",
        summary={"accuracy": 0.5, "dataset": "d1"},
        runner_report_text=None,
    )
    # should contain stable kvs
    assert "task=task_x" in s
    assert "run_id=run_x" in s
    assert "base_url=http://x" in s
    assert "dataset=d1" in s
    assert "accuracy=0.5" in s
    assert s.endswith("\n")