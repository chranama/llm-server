from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import llm_eval.cli as cli


class _FakeRunner:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    async def run(self, max_examples=None, model_override=None):
        return self._payload


@pytest.mark.anyio
async def test_cli_writes_report_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # --- Arrange: deterministic run_id + stable outdir root ---
    monkeypatch.setattr(cli, "_utc_run_id", lambda: "20200101T000000Z", raising=True)

    # Patch config loader + api key so CLI doesn't depend on filesystem/env
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _path: {
            "service": {"base_url": "http://example", "api_key_env": "EVAL_API_KEY"},
            "run": {"outdir_root": str(tmp_path)},
            "defaults": {"model_id": "m0"},
            "datasets": {"extraction_sroie": {"enabled": True}},
        },
        raising=True,
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "k_test", raising=True)

    # Patch task factories to return a fake runner that yields the contract payload
    payload = {
        "summary": {
            # intentionally omit task/run_id to ensure CLI injects them
            "schema_validity_rate": 1.0,
            "latency_p50_ms": 10.0,
        },
        "results": [
            {"ok": True, "status_code": 200, "latency_ms": 12.3, "id": "a"},
            {"ok": False, "status_code": 422, "error_code": "schema_validation_failed", "latency_ms": 34.0, "id": "b"},
        ],
        "report_text": None,
        "config": {"foo": "bar"},
    }

    monkeypatch.setattr(
        cli,
        "_task_factories",
        lambda: {"extraction_sroie": (lambda base_url, api_key, cfg: _FakeRunner(payload))},
        raising=True,
    )

    # Avoid noisy stdout assertions; just disable print
    argv = [
        "--config",
        "does_not_matter.yaml",
        "--task",
        "extraction_sroie",
        "--base-url",
        "http://example",
        "--api-key",
        "k_test",
        "--no-print-summary",
        "--save",
    ]

    # --- Act ---
    await cli.amain(argv)

    # --- Assert: output directory and files exist ---
    run_dir = tmp_path / "extraction_sroie" / "20200101T000000Z"
    assert run_dir.exists() and run_dir.is_dir()

    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.jsonl"
    report_txt_path = run_dir / "report.txt"
    report_md_path = run_dir / "report.md"
    config_path = run_dir / "config.json"

    for p in [summary_path, results_path, report_txt_path, report_md_path, config_path]:
        assert p.exists(), f"Missing artifact: {p}"
        assert p.stat().st_size > 0, f"Empty artifact: {p}"

    # summary.json should include injected run_dir, task, run_id
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task"] == "extraction_sroie"
    assert summary["run_id"] == "20200101T000000Z"
    assert summary["run_dir"] == str(run_dir)

    # results.jsonl should have 2 lines
    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "a"
    assert json.loads(lines[1])["id"] == "b"

    # report.txt and report.md should have recognizable headers
    report_txt = report_txt_path.read_text(encoding="utf-8")
    assert "task=extraction_sroie" in report_txt
    assert "run_id=20200101T000000Z" in report_txt

    report_md = report_md_path.read_text(encoding="utf-8")
    assert "# Eval Report: `extraction_sroie`" in report_md
    assert "- **run_id:** `20200101T000000Z`" in report_md