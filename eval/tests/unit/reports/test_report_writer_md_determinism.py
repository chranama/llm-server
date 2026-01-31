from __future__ import annotations

from llm_eval.reports import writer


def test_render_report_md_patches_generated_at(monkeypatch):
    monkeypatch.setattr(writer, "_utc_now_iso", lambda: "2020-01-01T00:00:00Z", raising=True)

    md = writer.render_report_md(
        task="task",
        run_id="rid",
        base_url="http://base",
        summary={"precision": 0.25},
        results=[],
        runner_report_text=None,
    )

    assert "- **generated_at:** `2020-01-01T00:00:00Z`" in md
    assert "# Eval Report: `task`" in md
    assert "| `precision` | `0.25` |" in md
    assert md.endswith("\n")