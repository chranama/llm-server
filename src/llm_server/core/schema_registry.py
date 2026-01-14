# src/llm_server/core/schema_registry.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover
    Draft202012Validator = None  # type: ignore


@dataclass(frozen=True)
class SchemaInfo:
    schema_id: str
    title: str | None
    description: str | None


class SchemaNotFoundError(FileNotFoundError):
    def __init__(self, schema_id: str):
        self.schema_id = schema_id
        self.code = "schema_not_found"
        self.message = f"Schema '{schema_id}' not found"
        super().__init__(self.message)


class SchemaLoadError(RuntimeError):
    def __init__(self, schema_id: str, error: Exception):
        self.schema_id = schema_id
        self.error = error
        self.code = "schema_load_failed"
        self.message = f"Failed to load schema '{schema_id}': {error}"
        super().__init__(self.message)


def _schemas_dir() -> Path:
    """
    Resolution order:
      1. SCHEMAS_DIR env var
      2. project root /schemas
      3. package-bundled schemas
    """
    import os

    env_dir = os.getenv("SCHEMAS_DIR")
    if env_dir:
        return Path(env_dir)

    # project root /schemas
    root = Path(__file__).resolve().parents[3]
    root_schemas = root / "schemas"
    if root_schemas.exists():
        return root_schemas

    # fallback to package path
    return Path(__file__).resolve().parents[1] / "schemas"


# Simple in-memory cache: schema_id -> parsed schema dict
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def list_schemas() -> list[SchemaInfo]:
    out: list[SchemaInfo] = []
    d = _schemas_dir()
    if not d.exists():
        return out

    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict):
            out.append(
                SchemaInfo(
                    schema_id=p.stem,
                    title=payload.get("title"),
                    description=payload.get("description"),
                )
            )
        else:
            # Non-dict schema JSON is considered invalid; skip for discovery.
            continue

    return out


def load_schema(schema_id: str) -> dict[str, Any]:
    """
    Load a schema JSON dict from llm_server/schemas/{schema_id}.json.

    Adds:
      - In-memory cache
      - Optional schema sanity-check (Draft 2020-12), when jsonschema is installed

    Raises:
      - SchemaNotFoundError
      - SchemaLoadError
    """
    cached = _SCHEMA_CACHE.get(schema_id)
    if cached is not None:
        return cached

    p = _schemas_dir() / f"{schema_id}.json"
    if not p.exists():
        raise SchemaNotFoundError(schema_id)

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Schema JSON must be an object at the top level")

        # Optional: validate schema structure itself, if dependency is present.
        # If jsonschema isn't installed, we still allow loading; extraction will fail later with jsonschema_missing.
        if Draft202012Validator is not None:
            Draft202012Validator.check_schema(payload)

        _SCHEMA_CACHE[schema_id] = payload
        return payload

    except Exception as e:
        raise SchemaLoadError(schema_id, e) from e