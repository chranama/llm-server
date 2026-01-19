# tests/unit/cli/test_cli_disabled_task.py
from __future__ import annotations

import pytest

import llm_eval.cli as cli


@pytest.mark.asyncio
async def test_cli_disabled_task_errors(monkeypatch):
    # YAML says task is disabled.
    monkeypatch.setattr(
        cli,
        "load_eval_yaml",
        lambda _: {
            "service": {"base_url": "http://svc"},
            "datasets": {"fake_task": {"enabled": False}},
            "run": {"outdir_root": "results"},
        },
    )
    monkeypatch.setattr(cli, "get_api_key", lambda _cfg: "APIKEY")

    # Ensure the task exists as a CLI choice.
    monkeypatch.setattr(
        cli,
        "TASK_FACTORIES",
        {"fake_task": lambda base_url, api_key, cfg: None},  # never called
    )

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
            no_save=True,
            outdir=None,
            debug_n=0,
            debug_fields=None,
        ),
    )

    with pytest.raises(SystemExit) as e:
        await cli.amain()

    # argparse.error exits with code 2
    assert e.value.code == 2