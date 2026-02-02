# contracts/src/llm_contracts/schema.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import tempfile

from jsonschema import Draft202012Validator

Pathish = Union[str, Path]


@dataclass(frozen=True)
class SchemaValidationError(Exception):
    """
    Raised when a payload fails JSON Schema validation.

    Keep this as a "clean" error so callers can decide:
      - fail closed
      - log and continue
      - surface details to an operator
    """
    schema_name: str
    message: str
    errors: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.errors:
            return f"{self.schema_name}: {self.message}"
        joined = "\n  - " + "\n  - ".join(self.errors)
        return f"{self.schema_name}: {self.message}{joined}"


def schemas_root() -> Path:
    """
    Resolve repository schemas root.

    Resolution order:
      1) SCHEMAS_ROOT env var (recommended in containers)
      2) ./schemas relative to current working directory
    """
    return Path(os.getenv("SCHEMAS_ROOT", "schemas")).resolve()


def internal_schemas_dir() -> Path:
    return schemas_root() / "internal"


def load_internal_schema(schema_filename: str) -> Dict[str, Any]:
    """
    Load a JSON Schema file from schemas/internal/.

    Example:
      load_internal_schema("policy_decision_v1.schema.json")
    """
    p = internal_schemas_dir() / schema_filename
    if not p.exists():
        raise FileNotFoundError(f"schema not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"schema must be a JSON object: {p}")
    return obj


# simple in-process cache (validator compile cost is not huge, but avoid churn)
_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def validator_for_internal(schema_filename: str) -> Draft202012Validator:
    """
    Return a compiled Draft2020-12 validator for an internal schema filename.
    Cached for process lifetime.
    """
    v = _VALIDATOR_CACHE.get(schema_filename)
    if v is not None:
        return v
    schema = load_internal_schema(schema_filename)
    v = Draft202012Validator(schema)
    _VALIDATOR_CACHE[schema_filename] = v
    return v


def validate_internal(schema_filename: str, payload: Any) -> None:
    """
    Validate payload against schemas/internal/<schema_filename>.
    Raises SchemaValidationError on failure.
    """
    v = validator_for_internal(schema_filename)

    try:
        v.validate(payload)
    except Exception as e:
        # build a more actionable error list if possible
        errors: list[str] = []
        try:
            for err in sorted(v.iter_errors(payload), key=lambda x: list(x.path)):
                path = ".".join(str(p) for p in err.path) if err.path else "<root>"
                errors.append(f"{path}: {err.message}")
        except Exception:
            errors = []

        msg = f"validation failed ({type(e).__name__})"
        raise SchemaValidationError(schema_name=schema_filename, message=msg, errors=tuple(errors)) from e


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # write temp in same directory so os.replace is atomic
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

        # fsync directory so rename is durable
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    finally:
        # if anything failed before replace, clean up temp
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass


def atomic_write_json_internal(schema_filename: str, path: Pathish, payload: Dict[str, Any]) -> Path:
    """
    Validate then atomically write JSON to disk.
    (fail fast if payload does not conform)

    Returns resolved Path.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict for atomic_write_json_internal")
    validate_internal(schema_filename, payload)

    p = Path(path).resolve()
    s = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(p, s)
    return p


def read_json_internal(schema_filename: str, path: Pathish) -> Dict[str, Any]:
    """
    Read + validate JSON from disk.

    Returns dict (validated).
    """
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SchemaValidationError(schema_name=schema_filename, message="payload root must be an object")
    validate_internal(schema_filename, raw)
    return raw