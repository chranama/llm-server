# cli/errors.py
from __future__ import annotations

import sys


class CLIError(RuntimeError):
    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def die(message: str, code: int = 2) -> None:
    msg = (message or "").strip()
    if msg:
        print(f"❌ {msg}", file=sys.stderr)
    raise SystemExit(code)