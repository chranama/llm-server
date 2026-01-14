# eval/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None


def load_eval_yaml(path: str) -> Dict[str, Any]:
    """
    Load config/eval.yaml. Missing file => {}.
    """
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def dig(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_api_key(cfg: Dict[str, Any]) -> Optional[str]:
    """
    If eval.yaml provides api_key_env, read that env var.
    """
    import os

    env_name = dig(cfg, "service", "api_key_env", default=None)
    if isinstance(env_name, str) and env_name.strip():
        return os.getenv(env_name.strip()) or None
    return None