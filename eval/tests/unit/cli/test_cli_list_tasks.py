# tests/unit/cli/test_cli_list_tasks.py
from __future__ import annotations

import pytest

import llm_eval.cli as cli


@pytest.mark.asyncio
async def test_cli_list_tasks_prints_sorted_tasks_and_exits(monkeypatch, capsys):
    # Keep it deterministic and independent of whatever tasks exist in prod.
    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {
            "z_task": lambda base_url, api_key, cfg: None,  # never called
            "a_task": lambda base_url, api_key, cfg: None,  # never called
            "m_task": lambda base_url, api_key, cfg: None,  # never called
        },
    )

    # parse_args should request list_tasks and NOT provide --task
    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: cli.argparse.Namespace(
            config="ignored.yaml",
            task=None,
            list_tasks=True,
            base_url=None,
            api_key=None,
            max_examples=None,
            model=None,
            print_summary=False,
            no_print_summary=True,
            save=False,
            no_save=True,
            outdir=None,
            debug_n=0,
            debug_fields=None,
        ),
    )

    await cli.amain()

    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["a_task", "m_task", "z_task"]