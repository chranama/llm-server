# src/llm_policy/utils/fs.py
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def atomic_write_text(path: str | Path, text: str) -> None:
    """
    Best-effort atomic write: write to temp file in same directory, then replace.
    """
    p = Path(path)
    ensure_dir(p.parent)
    fd: Optional[int] = None
    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".", dir=str(p.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        fd = None
        os.replace(tmp_path, str(p))
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                x = json.loads(s)
                if isinstance(x, dict):
                    out.append(x)
            except Exception:
                # tolerate partial files
                continue
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)  # type: ignore[no-any-return]
    return obj if isinstance(obj, dict) else {}


def write_yaml(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    atomic_write_text(p, text)