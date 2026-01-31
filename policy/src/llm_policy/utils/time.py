# src/llm_policy/utils/time.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def utc_run_id() -> str:
    # Matches your llm_eval format for easy sorting
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def parse_iso8601(s: str) -> Optional[datetime]:
    try:
        ss = s.strip()
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        return datetime.fromisoformat(ss)
    except Exception:
        return None


@dataclass(frozen=True)
class TimeWindow:
    start: Optional[datetime] = None
    end: Optional[datetime] = None