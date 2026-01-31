# cli/commands/eval.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cli.errors import CLIError
from cli.util.proc import ensure_bins, run
from cli.types import GlobalConfig  # type: ignore[attr-defined]


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="Run llm_eval CLI (host or compose).")
    p.set_defaults(_handler=_handle)

    sp = p.add_subparsers(dest="eval_cmd", required=True)

    h = sp.add_parser("host", help="Run eval in eval_host compose container (API_BASE_URL=host.docker.internal).")
    h.add_argument("args", nargs=argparse.REMAINDER, help="Args passed to `eval` inside container")

    d = sp.add_parser("docker", help="Run eval in eval compose container (API_BASE_URL=http://api:8000).")
    d.add_argument("args", nargs=argparse.REMAINDER, help="Args passed to `eval` inside container")


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")

    cmd = args.eval_cmd
    env = {"COMPOSE_PROJECT_NAME": cfg.project_name}
    base = ["docker", "compose", "--env-file", str(cfg.env_file), "-f", str(cfg.compose_yml)]

    if cmd == "host":
        # equivalent to: just dc "eval-host" "run --rm eval_host sh -lc \"eval {{EVAL_ARGS}}\""
        extra = list(args.args or [])
        if extra and extra[0] == "--":
            extra = extra[1:]
        inner = "eval " + " ".join(_shell_quote(x) for x in extra)
        run(base + ["--profile", "eval-host", "run", "--rm", "eval_host", "sh", "-lc", inner], env=env, verbose=args.verbose)
        return 0

    if cmd == "docker":
        extra = list(args.args or [])
        if extra and extra[0] == "--":
            extra = extra[1:]
        inner = "eval " + " ".join(_shell_quote(x) for x in extra)
        run(base + ["--profile", "eval", "run", "--rm", "eval", "sh", "-lc", inner], env=env, verbose=args.verbose)
        return 0

    raise CLIError(f"Unknown eval command: {cmd}", code=2)


def _shell_quote(s: str) -> str:
    # minimal safe quoting for sh -lc
    if not s:
        return "''"
    if all(c.isalnum() or c in "-._/:=" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"