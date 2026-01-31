# backend/src/llm_server/reports/types.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class MeUsage:
    api_key: str
    role: Optional[str]
    total_requests: int
    first_request_at: Optional[datetime]
    last_request_at: Optional[datetime]
    total_prompt_tokens: int
    total_completion_tokens: int


@dataclass(frozen=True)
class AdminUsageRow:
    api_key: str
    name: Optional[str]
    role: Optional[str]
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    first_request_at: Optional[datetime]
    last_request_at: Optional[datetime]


@dataclass(frozen=True)
class ApiKeyInfo:
    key_prefix: str
    name: Optional[str]
    role: Optional[str]
    created_at: datetime
    disabled: bool


@dataclass(frozen=True)
class ApiKeyListPage:
    total: int
    limit: int
    offset: int
    items: list[ApiKeyInfo]


@dataclass(frozen=True)
class LogsPage:
    total: int
    limit: int
    offset: int
    items: list[Any]  # caller can supply ORM rows or already-shaped dicts


@dataclass(frozen=True)
class ModelStats:
    model_id: str
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None


@dataclass(frozen=True)
class AdminStats:
    window_days: int
    since: datetime
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_latency_ms: float | None
    per_model: list[ModelStats]


@dataclass(frozen=True)
class ReportDoc:
    """
    A generic "report document" container used by writer.py.
    """
    title: str
    data: Mapping[str, Any]