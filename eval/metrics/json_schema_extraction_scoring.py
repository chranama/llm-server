# src/llm_server/eval/metrics/json_schema_extraction_scoring.py

from typing import Dict, Any
from jsonschema import Draft7Validator
import json

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

def flatten_json(obj, prefix=""):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            items.update(flatten_json(v, f"{prefix}{k}."))
    else:
        items[prefix[:-1]] = obj
    return items

def score_json_extraction(
    predicted_text: str,
    expected: Dict[str, Any],
    schema: Dict[str, Any],
):
    parsed = safe_parse_json(predicted_text)

    result = {
        "json_valid": False,
        "schema_valid": False,
        "field_exact_match": {},
        "required_present": False,
        "required_all_correct": False,
    }

    if parsed is None:
        return result

    result["json_valid"] = True

    validator = Draft7Validator(schema)
    schema_errors = list(validator.iter_errors(parsed))
    result["schema_valid"] = len(schema_errors) == 0

    flat_pred = flatten_json(parsed)
    flat_exp = flatten_json(expected)

    field_matches = {}
    for k, v in flat_exp.items():
        field_matches[k] = flat_pred.get(k) == v

    result["field_exact_match"] = field_matches

    required_fields = schema.get("required", [])
    result["required_present"] = all(f in parsed for f in required_fields)
    result["required_all_correct"] = all(
        field_matches.get(f, False) for f in required_fields
    )

    return result