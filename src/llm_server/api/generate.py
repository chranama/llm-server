from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from transformers import AutoTokenizer

from llm_server.db.models import InferenceLog, CompletionCache
from llm_server.db.session import async_session_maker
from llm_server.core.config import settings
from llm_server.api.deps import get_api_key
from llm_server.services.llm import build_llm_from_settings, MultiModelManager
from llm_server.core.metrics import LLM_TOKENS

router = APIRouter()


# -------------------------------
# Schemas
# -------------------------------


class GenerateRequest(BaseModel):
    prompt: str

    # Optional override to pick a non-default model (when multi-model is enabled)
    model: str | None = Field(
        default=None,
        description="Optional model id override for multi-model routing",
    )

    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None


class StreamRequest(GenerateRequest):
    pass


# -------------------------------
# LLM dependency + routing
# -------------------------------


def get_llm(request: Request) -> Any:
    """
    Accessor used as a FastAPI dependency and imported by health.py.

    - If app.state.llm is already set (from lifespan startup), just return it.
    - If it's None (e.g. startup failed or was bypassed), lazily build it
      from settings so the API can still serve requests.
    """
    llm = getattr(request.app.state, "llm", None)

    if llm is None:
        # Lazy fallback initialization (no arguments; uses global settings)
        llm = build_llm_from_settings()
        request.app.state.llm = llm

    return llm


def resolve_model(request: Request, llm: Any, model_override: str | None) -> tuple[str, Any]:
    allowed = settings.all_model_ids

    if model_override is None:
        model_id = settings.model_id
    else:
        model_id = model_override
        if model_id not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' not allowed. Allowed: {allowed}",
            )

    # Multi-model: MultiModelManager
    if isinstance(llm, MultiModelManager):
        if model_id not in llm:
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model_id}' not found in LLM registry",
            )
        return model_id, llm[model_id]

    # (optional) Multi-model: dict (if you ever use that form)
    if isinstance(llm, dict):
        if model_id not in llm:
            raise HTTPException(
                status_code=500,
                detail=f"Model '{model_id}' not found in LLM registry",
            )
        return model_id, llm[model_id]

    # Single-model mode
    return model_id, llm


# -------------------------------
# Helpers
# -------------------------------


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]


def fingerprint_params(body: GenerateRequest) -> str:
    """
    Fingerprint all generation params except:
    - prompt (handled separately)
    - model (we key by model_id in the DB, so no need to double-encode it)

    Cache key is effectively:
        (model_id, prompt_hash, params_fingerprint)
    """
    params = body.model_dump(exclude={"prompt", "model"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:32]


# -------------------------------
# Token counting helpers
# -------------------------------


@lru_cache(maxsize=16)
def _get_tokenizer(model_id: str):
    """
    Lazily load and cache a tokenizer per model_id.

    This is independent from the runtime ModelManager so it also works
    when the model is remote (HttpLLMClient) as long as the HF repo
    is accessible locally.
    """
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def count_tokens(model_id: str, prompt: str, completion: str | None) -> tuple[int | None, int | None]:
    """
    Best-effort token counting.

    Returns (prompt_tokens, completion_tokens). If anything goes wrong,
    returns (None, None) so logging doesn't break inference.
    """
    try:
        tok = _get_tokenizer(model_id)

        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        prompt_tokens = len(prompt_ids)

        if completion:
            completion_ids = tok(completion, add_special_tokens=False).input_ids
            completion_tokens = len(completion_ids)
        else:
            completion_tokens = 0

        return prompt_tokens, completion_tokens
    except Exception:
        # Don’t let logging kill the request
        return None, None


# -------------------------------
# Generate endpoint
# -------------------------------


@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    # Resolve which logical model id this request should use
    model_id, model = resolve_model(request, llm, body.model)

    # Tag request for logging/metrics middleware
    request.state.route = "/v1/generate"
    request.state.model_id = model_id

    # Best-effort request id (set by RequestLoggingMiddleware)
    request_id = getattr(request.state, "request_id", None)

    prompt_hash = hash_prompt(body.prompt)
    params_fp = fingerprint_params(body)

    start = time.time()

    async with async_session_maker() as session:
        # ---- 1. Check cache ----
        cached = await session.execute(
            select(CompletionCache).where(
                CompletionCache.model_id == model_id,
                CompletionCache.prompt_hash == prompt_hash,
                CompletionCache.params_fingerprint == params_fp,
            )
        )
        cached = cached.scalar_one_or_none()

        if cached:
            output = cached.output
            latency = (time.time() - start) * 1000

            request.state.cached = True

            # Token counting from cached output
            prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)

            # Token metrics
            if prompt_tokens is not None:
                LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
            if completion_tokens is not None:
                LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

            log = InferenceLog(
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/generate",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json=body.model_dump(
                    exclude={"prompt", "model"},
                    exclude_none=True,
                ),
                prompt=body.prompt,
                output=output,
                latency_ms=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            session.add(log)
            await session.commit()

            return {
                "model": model_id,
                "output": output,
                "cached": True,
            }

        # ---- 2. Run model ----
        result = model.generate(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            stop=body.stop,
        )

        output = result if isinstance(result, str) else str(result)
        latency = (time.time() - start) * 1000

        request.state.cached = False

        # Token counting from live output
        prompt_tokens, completion_tokens = count_tokens(model_id, body.prompt, output)

        # Token metrics
        if prompt_tokens is not None:
            LLM_TOKENS.labels(direction="prompt", model_id=model_id).inc(prompt_tokens)
        if completion_tokens is not None:
            LLM_TOKENS.labels(direction="completion", model_id=model_id).inc(completion_tokens)

        # ---- 3. Save cache ----
        cache = CompletionCache(
            model_id=model_id,
            prompt=body.prompt,
            prompt_hash=prompt_hash,
            params_fingerprint=params_fp,
            output=output,
        )
        session.add(cache)

        # ---- 4. Save log ----
        log = InferenceLog(
            api_key=api_key.key,
            request_id=request_id,
            route="/v1/generate",
            client_host=request.client.host if request.client else None,
            model_id=model_id,
            params_json=body.model_dump(
                exclude={"prompt", "model"},
                exclude_none=True,
            ),
            prompt=body.prompt,
            output=output,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        session.add(log)

        await session.commit()

        return {
            "model": model_id,
            "output": output,
            "cached": False,
        }


# -------------------------------
# Stream endpoint
# -------------------------------


async def _sse_event_generator(
    model: Any,
    body: StreamRequest,
) -> AsyncGenerator[str, None]:
    """
    Wrap the model.stream(...) iterator into an SSE stream.
    """
    for chunk in model.stream(
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        stop=body.stop,
    ):
        yield f"data: {chunk}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/stream")
async def stream(
    request: Request,
    body: StreamRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    # Resolve model id and tag request for logging/metrics
    model_id, model = resolve_model(request, llm, body.model)
    request.state.route = "/v1/stream"
    request.state.model_id = model_id
    request.state.cached = False  # no caching for streams

    # (Optionally: you could later log streaming requests keyed by model_id)
    generator = _sse_event_generator(model, body)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
    )