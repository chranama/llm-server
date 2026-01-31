# cli/util/env.py
from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_dotenv_file(path: Path) -> Dict[str, str]:
    """
    Minimal .env loader:
      - ignores comments/blank lines
      - supports KEY=VALUE
      - trims surrounding quotes in VALUE ("..." or '...')
    Returns a dict but does NOT mutate os.environ (caller decides).
    """
    if not path.exists():
        return {}

    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()

        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]

        if k:
            out[k] = v
    return out