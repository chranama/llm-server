# tests/fakes/fake_http_client.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm_eval.client.http_client import ExtractErr, ExtractOk, GenerateErr, GenerateOk


@dataclass
class FakeHttpClient:
    """
    Queue-driven fake that satisfies the HttpClient Protocol.

    - Push canned GenerateOk/GenerateErr objects into generate_queue
    - Push canned ExtractOk/ExtractErr objects into extract_queue
    - If a queue is empty, we return a sane default error.
    """
    generate_queue: List[GenerateOk | GenerateErr] = field(default_factory=list)
    extract_queue: List[ExtractOk | ExtractErr] = field(default_factory=list)

    # optional: capture calls for assertions/debugging
    generate_calls: List[Dict[str, Any]] = field(default_factory=list)
    extract_calls: List[Dict[str, Any]] = field(default_factory=list)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        model: Optional[str] = None,
        cache: Optional[bool] = None,
    ) -> GenerateOk | GenerateErr:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "model": model,
                "cache": cache,
            }
        )

        if self.generate_queue:
            return self.generate_queue.pop(0)

        return GenerateErr(
            status_code=500,
            error_code="fake_empty_queue",
            message="FakeHttpClient.generate_queue empty",
            extra=None,
            latency_ms=0.0,
        )

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
        self.extract_calls.append(
            {
                "schema_id": schema_id,
                "text": text,
                "model": model,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "cache": cache,
                "repair": repair,
            }
        )

        if self.extract_queue:
            return self.extract_queue.pop(0)

        return ExtractErr(
            status_code=500,
            error_code="fake_empty_queue",
            message="FakeHttpClient.extract_queue empty",
            extra=None,
            latency_ms=0.0,
        )