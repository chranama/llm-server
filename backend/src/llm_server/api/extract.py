# backend/src/llm_server/api/extract.py
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from llm_server.api.deps import (
    fingerprint_pydantic,
    get_api_key,
    get_llm,
    make_extract_redis_key,
    require_capability,
    resolve_model,
    sha32,
)
from llm_server.core.errors import AppError
from llm_server.core.metrics import (
    EXTRACTION_CACHE_HITS,
    EXTRACTION_REPAIR,
    EXTRACTION_REQUESTS,
    EXTRACTION_VALIDATION_FAILURES,
)
from llm_server.core.redis import get_redis_from_request
from llm_server.core.schema_registry import (
    SchemaLoadError,
    SchemaNotFoundError,
    list_schemas,
    load_schema,
)
from llm_server.core.validation import (
    DependencyMissingError,
    JSONSchemaValidationError,
    validate_jsonschema,
)
import llm_server.db.session as db_session  # module import so tests can patch session wiring
from llm_server.services.inference import (
    CacheSpec,
    get_cached_output,
    set_request_meta,
    write_cache,
    write_inference_log,
)

router = APIRouter()

REDIS_TTL_SECONDS = 3600

_JSON_BEGIN = "<<<JSON>>>"
_JSON_END = "<<<END>>>"


class ExtractRequest(BaseModel):
    schema_id: str = Field(..., description="Schema id (e.g. ticket_v1, invoice_v1, receipt_v1)")
    text: str = Field(..., description="Raw text or OCR text to extract from")

    model: str | None = Field(default=None, description="Optional model id override for multi-model routing")

    max_new_tokens: int | None = 512
    temperature: float | None = 0.0

    cache: bool = True
    repair: bool = True


class ExtractResponse(BaseModel):
    schema_id: str
    model: str
    data: dict[str, Any]
    cached: bool
    repair_attempted: bool


def _strip_wrapping_code_fences(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n?", "", s)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()


def _schema_summary(schema: dict[str, Any]) -> str:
    required = schema.get("required") or []
    props = schema.get("properties") or {}

    lines: list[str] = []
    if required:
        lines.append(f"REQUIRED_FIELDS: {', '.join(required)}")

    lines.append("FIELDS:")
    for k, v in props.items():
        if not isinstance(v, dict):
            continue
        t = v.get("type", "any")
        enum = v.get("enum")
        pat = v.get("pattern")
        desc = v.get("description")

        pieces = [f"- {k}: {t}"]
        if enum:
            pieces.append(f"enum={enum}")
        if pat:
            pieces.append(f"pattern={pat}")
        if desc:
            pieces.append(f"desc={str(desc)[:80]}")
        lines.append("  " + " | ".join(pieces))

    ap = schema.get("additionalProperties", None)
    if ap is False:
        lines.append("CONSTRAINT: additionalProperties=false (no extra keys).")

    return "\n".join(lines)


def _build_extraction_prompt(schema_id: str, schema: dict[str, Any], text: str) -> str:
    summary = _schema_summary(schema)
    return (
        "You are a structured information extraction engine.\n"
        "Return ONLY a JSON object that matches the contract below.\n"
        "No markdown. No code fences. No commentary.\n"
        "If a value is unknown: omit the field unless it is REQUIRED.\n"
        "If a REQUIRED field is missing in the text: set it to null.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n"
    )


def _build_repair_prompt(
    schema_id: str,
    schema: dict[str, Any],
    text: str,
    bad_output: str,
    error_hint: str,
) -> str:
    summary = _schema_summary(schema)
    return (
        "Your previous output did NOT match the contract.\n"
        "Fix it. Return ONLY the corrected JSON object.\n"
        "No markdown. No code fences. No commentary.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n\n"
        f"PREVIOUS_OUTPUT:\n{bad_output}\n\n"
        f"ERROR_HINT:\n{error_hint}\n"
    )


def _iter_json_objects(raw: str) -> list[dict[str, Any]]:
    s = _strip_wrapping_code_fences(raw)
    dec = json.JSONDecoder()

    objs: list[dict[str, Any]] = []
    i = 0
    n = len(s)

    while i < n:
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(s[j:])
            if isinstance(obj, dict):
                objs.append(obj)
            i = j + max(end, 1)
        except Exception:
            i = j + 1

    return objs


def _validate_first_matching(schema: dict[str, Any], raw_output: str) -> dict[str, Any]:
    if raw_output is None:
        raise AppError(
            code="invalid_json",
            message="Model output was empty.",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    s = raw_output.strip()

    if _JSON_BEGIN in s and _JSON_END in s:
        try:
            inner = s.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0].strip()
            inner = _strip_wrapping_code_fences(inner)
            obj = json.loads(inner)
            if not isinstance(obj, dict):
                raise ValueError("Delimited JSON was not an object")
            validate_jsonschema(schema, obj)
            return obj
        except DependencyMissingError as e:
            raise AppError(code=e.code, message=e.message, status_code=500) from e
        except JSONSchemaValidationError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                extra={"errors": e.errors, "raw_preview": (raw_output or "")[:500]},
            ) from e
        except Exception:
            pass

    candidates = _iter_json_objects(s)
    if not candidates:
        raise AppError(
            code="invalid_json",
            message="Model output did not contain any JSON object.",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            extra={"raw_preview": (raw_output or "")[:500]},
        )

    last_validation_error: Optional[JSONSchemaValidationError] = None

    for obj in candidates:
        try:
            validate_jsonschema(schema, obj)
            return obj
        except DependencyMissingError as e:
            raise AppError(code=e.code, message=e.message, status_code=500) from e
        except JSONSchemaValidationError as e:
            last_validation_error = e
            continue
        except Exception:
            continue

    extra: dict[str, Any] = {"raw_preview": (raw_output or "")[:500], "candidates_found": len(candidates)}
    if last_validation_error is not None:
        extra["errors"] = last_validation_error.errors

    raise AppError(
        code="schema_validation_failed",
        message="No JSON object in the model output conformed to the schema.",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        extra=extra,
    )


def _failure_stage_for_app_error(e: AppError, *, is_repair: bool) -> str | None:
    if getattr(e, "status_code", None) != status.HTTP_422_UNPROCESSABLE_CONTENT:
        return None
    if e.code == "invalid_json":
        return "repair_parse" if is_repair else "parse"
    if e.code == "schema_validation_failed":
        return "repair_validate" if is_repair else "validate"
    return "repair_validate" if is_repair else "validate"


@router.get("/v1/schemas")
async def schemas_index(api_key=Depends(get_api_key)):
    return [{"schema_id": s.schema_id, "title": s.title, "description": s.description} for s in list_schemas()]


@router.get("/v1/schemas/{schema_id}", response_model=dict)
async def schema_detail(schema_id: str, api_key=Depends(get_api_key)):
    try:
        schema = load_schema(schema_id)
    except SchemaNotFoundError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND,
            extra={"schema_id": e.schema_id},
        ) from e
    except SchemaLoadError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra={"schema_id": e.schema_id},
        ) from e

    return JSONResponse(content=schema)


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    request: Request,
    body: ExtractRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model, capability="extract", request=request)

    # Single enforcement path: deployment gate + per-model gate
    require_capability(model_id, "extract", request=request)

    set_request_meta(request, route="/v1/extract", model_id=model_id, cached=False)

    request_id = getattr(request.state, "request_id", None)

    EXTRACTION_REQUESTS.labels(schema_id=body.schema_id, model_id=model_id).inc()

    try:
        schema = load_schema(body.schema_id)
    except SchemaNotFoundError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND,
            extra={"schema_id": e.schema_id},
        ) from e
    except SchemaLoadError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra={"schema_id": e.schema_id},
        ) from e

    prompt_hash = sha32(f"{body.schema_id}\n{body.text}")
    params_fp = fingerprint_pydantic(body, exclude={"text", "model", "cache", "repair"})
    redis_key = make_extract_redis_key(model_id, prompt_hash, params_fp)

    cache = CacheSpec(
        model_id=model_id,
        prompt=body.text,
        prompt_hash=prompt_hash,
        params_fp=params_fp,
        redis_key=redis_key,
        redis_ttl_seconds=REDIS_TTL_SECONDS,
    )

    start = time.time()
    redis = get_redis_from_request(request)

    async with db_session.get_sessionmaker()() as session:
        # ---- cache read (redis -> db), stored as JSON string in CompletionCache.output ----
        cached_out, cached_flag, layer = await get_cached_output(
            session,
            redis,
            cache=cache,
            kind="single",
            enabled=bool(body.cache),
        )

        if isinstance(cached_out, str) and cached_flag:
            data: dict[str, Any] | None
            try:
                data_obj = json.loads(cached_out)
                if not isinstance(data_obj, dict):
                    raise ValueError("Expected object")
                validate_jsonschema(schema, data_obj)
                data = data_obj
            except DependencyMissingError as e:
                raise AppError(code=e.code, message=e.message, status_code=500) from e
            except (JSONSchemaValidationError, Exception):
                data = None

            if isinstance(data, dict):
                EXTRACTION_CACHE_HITS.labels(
                    schema_id=body.schema_id,
                    model_id=model_id,
                    layer=(layer or "db"),
                ).inc()

                request.state.cached = True
                latency_ms = (time.time() - start) * 1000

                await write_inference_log(
                    session,
                    api_key=api_key.key,
                    request_id=request_id,
                    route="/v1/extract",
                    client_host=request.client.host if request.client else None,
                    model_id=model_id,
                    params_json={"schema_id": body.schema_id, "cache": True, "repair": body.repair},
                    prompt=body.text,
                    output=json.dumps(data, ensure_ascii=False),
                    latency_ms=latency_ms,
                    prompt_tokens=None,
                    completion_tokens=None,
                    commit=True,
                )

                return ExtractResponse(
                    schema_id=body.schema_id,
                    model=model_id,
                    data=data,
                    cached=True,
                    repair_attempted=False,
                )

        # ---- run model ----
        prompt = _build_extraction_prompt(body.schema_id, schema, body.text)
        result = model.generate(
            prompt=prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
        )
        output = result if isinstance(result, str) else str(result)

        repair_attempted = False

        try:
            data = _validate_first_matching(schema, output)
        except AppError as e:
            stage = _failure_stage_for_app_error(e, is_repair=False)
            if stage is not None:
                EXTRACTION_VALIDATION_FAILURES.labels(schema_id=body.schema_id, model_id=model_id, stage=stage).inc()

            if not body.repair:
                raise

            repair_attempted = True
            EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="attempted").inc()

            error_hint = json.dumps(
                {"code": e.code, "message": e.message, **(e.extra or {})},
                ensure_ascii=False,
            )

            repair_prompt = _build_repair_prompt(
                body.schema_id,
                schema,
                body.text,
                bad_output=output,
                error_hint=error_hint,
            )
            repair_result = model.generate(
                prompt=repair_prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=0.0,
            )
            repaired = repair_result if isinstance(repair_result, str) else str(repair_result)

            try:
                data = _validate_first_matching(schema, repaired)
                EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="success").inc()
            except AppError as e2:
                stage2 = _failure_stage_for_app_error(e2, is_repair=True)
                if stage2 is not None:
                    EXTRACTION_VALIDATION_FAILURES.labels(
                        schema_id=body.schema_id, model_id=model_id, stage=stage2
                    ).inc()
                EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="failure").inc()
                raise

        latency_ms = (time.time() - start) * 1000
        request.state.cached = False

        # ---- cache write: store JSON string in CompletionCache.output, and {"output": "..."} in Redis ----
        out_json = json.dumps(data, ensure_ascii=False)
        await write_cache(
            session,
            redis,
            cache=cache,
            output=out_json,
            enabled=bool(body.cache),
        )

        # ---- log ----
        await write_inference_log(
            session,
            api_key=api_key.key,
            request_id=request_id,
            route="/v1/extract",
            client_host=request.client.host if request.client else None,
            model_id=model_id,
            params_json={"schema_id": body.schema_id, "cache": body.cache, "repair": body.repair},
            prompt=body.text,
            output=out_json,
            latency_ms=latency_ms,
            prompt_tokens=None,
            completion_tokens=None,
            commit=True,
        )

        return ExtractResponse(
            schema_id=body.schema_id,
            model=model_id,
            data=data,
            cached=False,
            repair_attempted=repair_attempted,
        )