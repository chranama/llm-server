# cli/util/proc.py
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from cli.errors import CLIError


@dataclass(frozen=True)
class RunResult:
    code: int


def _fmt_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    verbose: bool = False,
    check: bool = True,
) -> RunResult:
    """
    Run a command with good ergonomics:
      - prints the command if verbose
      - raises CLIError on failure (if check=True)
    """
    merged_env = dict(os.environ)
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})

    if verbose:
        print(f"+ {_fmt_cmd(cmd)}")

    p = subprocess.run(list(cmd), cwd=cwd, env=merged_env)
    if check and p.returncode != 0:
        raise CLIError(f"Command failed (exit {p.returncode}): {_fmt_cmd(cmd)}", code=p.returncode)
    return RunResult(code=p.returncode)


def run_bash(
    script: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    verbose: bool = False,
    check: bool = True,
) -> RunResult:
    """
    Run via bash -lc to preserve parity with your prior justfile behavior.
    """
    return run(["bash", "-lc", script], cwd=cwd, env=env, verbose=verbose, check=check)


def ensure_bins(*bins: str) -> None:
    missing = []
    for b in bins:
        if not _which(b):
            missing.append(b)
    if missing:
        raise CLIError(f"Missing required tools: {', '.join(missing)}", code=2)


def _which(name: str) -> Optional[str]:
    # minimal shutil.which without importing extra
    for p in os.getenv("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None