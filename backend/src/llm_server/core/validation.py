# src/llm_server/core/validation.py
from __future__ import annotations

import json
from typing import Any, Optional

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover
    Draft202012Validator = None  # type: ignore


# -------------------------
# Typed exceptions (Phase 1/2)
# -------------------------


class StrictJSONError(Exception):
    def __init__(self, *, code: str, message: str, hint: Optional[str] = None):
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(message)


class JSONSchemaValidationError(Exception):
    def __init__(self, *, code: str, message: str, errors: list[dict[str, Any]]):
        self.code = code
        self.message = message
        self.errors = errors
        super().__init__(message)


class DependencyMissingError(Exception):
    def __init__(self, *, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# -------------------------
# Helpers
# -------------------------


def parse_json_strict(text: str) -> Any:
    """
    Strict JSON parsing.

    Requirements:
      - Input must be a single JSON value (no trailing garbage)
      - No code fences, no surrounding commentary
      - Disallow NaN / Infinity / -Infinity (not valid JSON)
      - Top-level MUST be a JSON object (dict) for our extraction contract
    """
    if text is None:
        raise StrictJSONError(code="invalid_json", message="No text to parse")

    s = text.strip()
    if not s:
        raise StrictJSONError(code="invalid_json", message="Empty text; expected JSON")

    # Common LLM failure mode: ```json ... ```
    # Reject both "starts with" and "contains" to prevent fenced blocks anywhere.
    if s.startswith("```") or "```" in s or s.startswith("~~~") or "~~~" in s:
        raise StrictJSONError(
            code="invalid_json",
            message="Model output contained a code fence; expected raw JSON only",
            hint="Return ONLY a JSON object with no markdown fences.",
        )

    def _no_constants(x: str):
        # Python's json module accepts NaN/Infinity by default. Reject them.
        raise StrictJSONError(
            code="invalid_json",
            message="Non-JSON numeric constant encountered",
            hint=f"Invalid constant: {x}. Use null or a real number.",
        )

    try:
        # Use raw_decode so we can detect trailing garbage precisely.
        decoder = json.JSONDecoder(parse_constant=_no_constants)
        obj, end = decoder.raw_decode(s)

        # Reject trailing non-whitespace (e.g., '{"a":1} trailing')
        if s[end:].strip():
            raise StrictJSONError(
                code="invalid_json",
                message="Model output contained trailing characters after JSON",
                hint="Return ONLY a JSON object (no extra text before/after).",
            )

        # Enforce top-level object contract
        if not isinstance(obj, dict):
            raise StrictJSONError(
                code="invalid_json",
                message="Top-level JSON must be an object",
                hint="Return a JSON object like {\"field\": \"value\"}, not an array or scalar.",
            )

        return obj

    except StrictJSONError:
        raise
    except json.JSONDecodeError as e:
        raise StrictJSONError(
            code="invalid_json",
            message="Model output was not valid JSON",
            hint=f"{e.msg} at line {e.lineno} column {e.colno}",
        ) from e


def validate_jsonschema(schema: dict[str, Any], data: Any) -> None:
    """
    Validate `data` against JSON Schema (Draft 2020-12).

    Notes:
      - We intentionally do NOT validate the schema itself here (check_schema),
        so schema errors are handled at schema load time (schema_registry) if you enable it there.
    """
    if Draft202012Validator is None:
        raise DependencyMissingError(
            code="jsonschema_missing",
            message="jsonschema dependency is not installed",
        )

    v = Draft202012Validator(schema)
    errs = sorted(v.iter_errors(data), key=lambda e: list(e.path))

    if not errs:
        return

    detail_errors: list[dict[str, Any]] = []
    for e in errs[:25]:
        loc = ".".join(str(x) for x in e.path) if e.path else "$"

        item: dict[str, Any] = {
            "loc": loc,
            "message": e.message,
            "validator": getattr(e, "validator", None),
        }

        vv = getattr(e, "validator_value", None)
        if vv is not None:
            item["expected"] = vv

        inst = getattr(e, "instance", None)
        if isinstance(inst, (str, int, float, bool)) or inst is None:
            item["actual"] = inst
        elif isinstance(inst, list) and len(inst) <= 10:
            item["actual"] = inst
        elif isinstance(inst, dict) and len(inst) <= 10:
            item["actual"] = inst

        detail_errors.append(item)

    raise JSONSchemaValidationError(
        code="schema_validation_failed",
        message="Extracted JSON did not conform to schema",
        errors=detail_errors,
    )