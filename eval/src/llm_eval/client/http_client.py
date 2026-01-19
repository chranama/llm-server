# llm_eval/client/http_client.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


# =========================
# Typed results
# =========================


@dataclass(frozen=True)
class GenerateOk:
    model: str
    output_text: str
    cached: bool
    latency_ms: float


@dataclass(frozen=True)
class GenerateErr:
    status_code: int
    error_code: str
    message: str
    extra: Optional[Dict[str, Any]]
    latency_ms: float


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

    - POST /v1/generate (text generation)
    - POST /v1/extract  (schema-validated extraction)

    Notes:
    - Never raises for HTTP errors: callers get GenerateErr / ExtractErr.
    - Reuses a single AsyncClient noted in self._client for connection pooling.
      Call .aclose() if you create a long-lived client (optional).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "test_api_key_123",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    def _get_client(self) -> httpx.AsyncClient:
        # Lazily create one client to avoid recreating TCP connections per request.
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _extract_text_from_generate_payload(payload: Any) -> str:
        """
        Be tolerant to a few likely /v1/generate response shapes.

        Examples:
          - {"output": "..."}
          - {"text": "..."}
          - {"completion": "..."}
          - {"data": {"output": "..."}}  (nested)
          - raw string
        """
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for k in ("output", "text", "completion", "result"):
                v = payload.get(k)
                if isinstance(v, str):
                    return v
            data = payload.get("data")
            if isinstance(data, dict):
                for k in ("output", "text", "completion", "result"):
                    v = data.get(k)
                    if isinstance(v, str):
                        return v
        return str(payload)

    @staticmethod
    def _extract_model_from_payload(payload: Any, fallback: str = "unknown") -> str:
        if isinstance(payload, dict):
            for k in ("model", "model_id"):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            data = payload.get("data")
            if isinstance(data, dict):
                v = data.get("model") or data.get("model_id")
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return fallback

    @staticmethod
    def _extract_error_fields(resp: httpx.Response) -> tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Normalize llm-server error shape.

        Expected server error payload:
          {"code": "...", "message": "...", "extra": {...}}
        If unavailable, fall back to resp.text.
        """
        j = HttpEvalClient._safe_json(resp)
        if not isinstance(j, dict):
            j = {}
        code = str(j.get("code") or "http_error")
        msg = str(j.get("message") or (resp.text or ""))
        extra = j.get("extra") if isinstance(j.get("extra"), dict) else None
        return code, msg, extra

    # =========================
    # /v1/generate
    # =========================

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> GenerateOk | GenerateErr:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }
        if model is not None:
            payload["model"] = model
        if cache is not None:
            payload["cache"] = bool(cache)

        t0 = time.time()
        try:
            r = await self._get_client().post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers=self._headers(),
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            # "status_code=0" signals transport/client failure, not an HTTP response.
            return GenerateErr(
                status_code=0,
                error_code="transport_error",
                message=f"{type(e).__name__}: {e}",
                extra=None,
                latency_ms=latency_ms,
            )

        latency_ms = (time.time() - t0) * 1000.0

        if r.status_code == 200:
            data = self._safe_json(r)
            output_text = self._extract_text_from_generate_payload(data).strip()
            model_id = self._extract_model_from_payload(data, fallback=(model or "unknown"))
            cached = bool(data.get("cached", False)) if isinstance(data, dict) else False

            return GenerateOk(
                model=str(model_id),
                output_text=output_text,
                cached=cached,
                latency_ms=latency_ms,
            )

        code, msg, extra = self._extract_error_fields(r)
        return GenerateErr(
            status_code=int(r.status_code),
            error_code=code,
            message=msg,
            extra=extra,
            latency_ms=latency_ms,
        )

    # =========================
    # /v1/extract
    # =========================

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
        payload: Dict[str, Any] = {
            "schema_id": schema_id,
            "text": text,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "cache": bool(cache),
            "repair": bool(repair),
        }
        if model is not None:
            payload["model"] = model

        t0 = time.time()
        try:
            r = await self._get_client().post(
                f"{self.base_url}/v1/extract",
                json=payload,
                headers=self._headers(),
            )
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            return ExtractErr(
                status_code=0,
                error_code="transport_error",
                message=f"{type(e).__name__}: {e}",
                extra=None,
                latency_ms=latency_ms,
            )

        latency_ms = (time.time() - t0) * 1000.0

        if r.status_code == 200:
            data = self._safe_json(r)
            if not isinstance(data, dict):
                data = {}

            # tolerate missing keys while keeping contract stable
            schema_out = data.get("schema_id", schema_id)
            model_out = data.get("model", model or "unknown")
            obj = data.get("data")
            if not isinstance(obj, dict):
                obj = {}

            return ExtractOk(
                schema_id=str(schema_out),
                model=str(model_out),
                data=dict(obj),
                cached=bool(data.get("cached", False)),
                repair_attempted=bool(data.get("repair_attempted", False)),
                latency_ms=latency_ms,
            )

        code, msg, extra = self._extract_error_fields(r)
        return ExtractErr(
            status_code=int(r.status_code),
            error_code=code,
            message=msg,
            extra=extra,
            latency_ms=latency_ms,
        )