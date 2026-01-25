# llm_eval/config.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover
    yaml = None


# Supports:
#   ${VAR}
#   ${VAR:-default}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_str(s: str) -> str:
    """
    Expand ${VAR} and ${VAR:-default} using current environment.
    Leaves unknown vars without defaults as "" (like many shells).
    """
    def repl(m: re.Match[str]) -> str:
        var = m.group(1)
        default = m.group(2)
        val = os.getenv(var)
        if val is not None and val != "":
            return val
        return default or ""

    # Expand our ${VAR:-default} first
    out = _ENV_PATTERN.sub(repl, s)
    # Then expand simple $VAR if present
    out = os.path.expandvars(out)
    return out


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        if "${" in obj or "$" in obj:
            return _expand_env_str(obj)
        return obj
    return obj


def load_eval_yaml(path: str) -> Dict[str, Any]:
    """
    Load config/eval.yaml. Missing file => {}.
    Expands environment variables in string values.
    """
    p = Path(path)
    if not p.exists() or yaml is None:
        return {}

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}

    expanded = _expand_env(raw)
    return expanded if isinstance(expanded, dict) else {}


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
    env_name = dig(cfg, "service", "api_key_env", default=None)
    if isinstance(env_name, str) and env_name.strip():
        return os.getenv(env_name.strip()) or None
    return None