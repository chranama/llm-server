# tests/unit/cli/test_cli_debug_print_does_not_crash.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import llm_eval.cli as cli
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


class _FakeRunner(BaseEvalRunner):
    task_name = "fake_task"

    def __init__(self, payload: dict[str, Any]):
        super().__init__(base_url="http://fake", api_key="fake", config=EvalConfig())
        self._payload = payload

    async def _run_impl(self) -> Any:
        return self._payload


@pytest.mark.asyncio
async def test_cli_debug_print_does_not_crash(monkeypatch, tmp_path: Path, capsys):
    # Config plumbing
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "run": {"outdir_root": str(tmp_path)},
            "datasets": {"fake_task": {"enabled": True}},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    long_pred = "x" * 600
    long_err = "y" * 600

    payload = {
        "summary": {"task": "fake_task", "run_id": "RIDDBG", "base_url": "http://svc"},
        "results": [
            {
                "id": "1",
                "status_code": 200,
                "latency_ms": 12.3,
                "predicted": long_pred,  # should be shortened
                "metrics": {"ok": True},
            },
            {
                "doc_id": "2",
                "status_code": 500,
                "error_code": "boom",
                "error": long_err,  # should be shortened
                "latency_ms": 45.6,
            },
            "not_a_dict",  # will be filtered out by _coerce_nested_payload
        ],
        "report_text": "r\n",
        "config": {"max_examples": 3},
    }

    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: _FakeRunner(payload)},
    )

    # Enable debug printing; keep summary printing off to reduce noise
    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: cli.argparse.Namespace(
            config="ignored.yaml",
            task="fake_task",
            list_tasks=False,
            base_url=None,
            api_key=None,
            max_examples=None,
            model=None,
            print_summary=False,
            no_print_summary=True,
            save=False,
            no_save=True,  # avoid filesystem assertions; focus is "doesn't crash"
            outdir=None,
            debug_n=10,
            debug_fields="id,doc_id,status_code,error_code,latency_ms,predicted,error,metrics",
        ),
    )

    await cli.amain()

    out = capsys.readouterr().out

    # Header sanity
    assert "DEBUG (first 2 results)" in out

    # Row sanity
    assert '"status_code": 200' in out
    assert '"error_code": "boom"' in out

    # Prove shortening happened (and that we didn't print full 600-char strings)
    assert long_pred not in out
    assert long_err not in out
    assert "..." in out  # at least one shortened field should include ellipsis

    # We printed JSON objects (pretty-printed)
    assert "{\n" in out
    assert "}\n" in out