# src/llm_policy/io/telemetry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass(frozen=True)
class TelemetrySnapshot:
    """
    Stub for phase-2 integration.

    For v0, keep this as a thin wrapper around raw /metrics text (Prometheus format)
    or JSON payloads from an admin endpoint.
    """
    source: str
    raw: str
    fetched_at_iso: Optional[str] = None


async def fetch_metrics_text(*, base_url: str, timeout_s: float = 10.0) -> TelemetrySnapshot:
    """
    Fetch Prometheus metrics text from /metrics.

    NOTE: This is intentionally minimal; parsing should live in policies or a future
    llm_policy/io/telemetry_parsing.py module.
    """
    url = base_url.rstrip("/") + "/metrics"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url)
        r.raise_for_status()
        return TelemetrySnapshot(source=url, raw=r.text, fetched_at_iso=None)


def fetch_metrics_text_sync(*, base_url: str, timeout_s: float = 10.0) -> TelemetrySnapshot:
    url = base_url.rstrip("/") + "/metrics"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(url)
        r.raise_for_status()
        return TelemetrySnapshot(source=url, raw=r.text, fetched_at_iso=None)