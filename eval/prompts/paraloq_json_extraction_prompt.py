# src/llm_server/eval/prompts/paraloq_json_extraction_prompt.py
from __future__ import annotations

import json
from typing import Any, Dict

_JSON_BEGIN = "<<<JSON>>>"
_JSON_END = "<<<END>>>"

# Keep prompt stable + strict. This benchmark is measuring model behavior under a fixed contract.
SYSTEM_PREAMBLE = (
    "You are a structured information extraction engine.\n"
    "Return ONLY a JSON object that satisfies the provided JSON Schema.\n"
    "No markdown. No code fences. No commentary. No trailing text.\n"
    "If you are uncertain about a field, omit it unless it is REQUIRED.\n"
    "If a REQUIRED field is missing in the input, output it as null.\n"
)


def _compact_json(obj: Dict[str, Any]) -> str:
    # Compact but deterministic
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_paraloq_json_extraction_prompt(
    *,
    text: str,
    schema: Dict[str, Any],
) -> str:
    """
    Produces a prompt that:
      - provides the schema explicitly
      - demands ONLY a JSON object
      - provides delimiters (not required, but helpful)
    """
    schema_str = _compact_json(schema)

    return (
        f"{SYSTEM_PREAMBLE}\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"JSON_SCHEMA:\n{schema_str}\n\n"
        f"INPUT_TEXT:\n{text}\n"
    )