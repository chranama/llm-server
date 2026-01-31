# cli/util/paths.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_repo_root(start: Path, compose_rel: str = "deploy/compose/docker-compose.yml") -> Path:
    """
    Find repo root by walking upward until we see one of:
      - deploy/compose/docker-compose.yml
      - backend/pyproject.toml
      - .git
    """
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / compose_rel).exists():
            return p
        if (p / "backend" / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
    return cur


def env_default_path(repo_root: Path, env_file: str = ".env") -> Path:
    return (repo_root / env_file).resolve()


def resolve_path(repo_root: Path, maybe_rel: Optional[str]) -> Optional[Path]:
    if maybe_rel is None:
        return None
    s = str(maybe_rel).strip()
    if not s:
        return None
    p = Path(os.path.expanduser(s))
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()