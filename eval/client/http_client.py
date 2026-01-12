# src/llm_server/eval/client/http_client.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass(frozen=True)
class ExtractOk:
    schema_id: str
    model: str
    data: Dict[str, Any]
    cached: bool
    repair_attempted: bool
    latency_ms: float


@dataclass(frozen=True)
class ExtractErr:
    status_code: int
    error_code: str
    message: str
    extra: Optional[Dict[str, Any]]
    latency_ms: float


class HttpEvalClient:
    """
    Talks to the llm-server API endpoints.

    - /v1/generate (legacy eval harness)
    - /v1/extract  (schema-validated extraction)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "test_api_key_123",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()

        return data["output"]

    async def extract(
        self,
        *,
        schema_id: str,
        text: str,
        model: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        cache: bool = True,
        repair: bool = True,
    ) -> ExtractOk | ExtractErr:
        """
        Calls POST /v1/extract.

        Returns:
          - ExtractOk on HTTP 200
          - ExtractErr on non-200 (does NOT raise)
        """
        payload: Dict[str, Any] = {
            "schema_id": schema_id,
            "text": text,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "cache": cache,
            "repair": repair,
        }
        if model is not None:
            payload["model"] = model

        t0 = time.time()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/v1/extract",
                json=payload,
                headers=self._headers(),
            )

        latency_ms = (time.time() - t0) * 1000.0

        if r.status_code == 200:
            data = r.json()
            return ExtractOk(
                schema_id=str(data["schema_id"]),
                model=str(data["model"]),
                data=dict(data["data"]),
                cached=bool(data.get("cached", False)),
                repair_attempted=bool(data.get("repair_attempted", False)),
                latency_ms=latency_ms,
            )

        # Standardize error shape (your AppError returns: code/message/extra)
        try:
            j = r.json()
        except Exception:
            j = {}

        return ExtractErr(
            status_code=r.status_code,
            error_code=str(j.get("code") or "http_error"),
            message=str(j.get("message") or r.text),
            extra=j.get("extra") if isinstance(j.get("extra"), dict) else None,
            latency_ms=latency_ms,
        )