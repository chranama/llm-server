# src/llm_server/eval/runners/paraloq_json_extraction_runner.py
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import httpx

from eval.datasets.paraloq_json_extraction import iter_paraloq_json_extraction
from eval.metrics.json_schema_extraction_scoring import score_json_extraction
from eval.prompts.paraloq_json_extraction_prompt import (
    _JSON_BEGIN,
    _JSON_END,
    build_paraloq_json_extraction_prompt,
)
from eval.runners.base import BaseEvalRunner


def _extract_text_from_generate_response(payload: Any) -> str:
    """
    Be tolerant to a few likely /v1/generate response shapes.

    Expected-ish shapes (examples):
      - {"text": "..."}
      - {"output": "..."}
      - {"completion": "..."}
      - {"data": {"text": "..."}}
      - raw string
    """
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("text", "output", "completion", "result"):
            v = payload.get(key)
            if isinstance(v, str):
                return v

        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("text", "output", "completion", "result"):
                v = data.get(key)
                if isinstance(v, str):
                    return v

        # last resort: stringify
        return json.dumps(payload, ensure_ascii=False)

    # last resort
    return str(payload)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if not s.startswith("```"):
        return s
    # remove first fence line and trailing fence
    s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _extract_delimited_json_block(s: str) -> str:
    """
    If delimiters exist, return what's inside.
    Otherwise return original.
    """
    if not s:
        return s
    if _JSON_BEGIN in s and _JSON_END in s:
        inner = s.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0]
        return _strip_code_fences(inner.strip())
    return _strip_code_fences(s)


class GenerateParaloqJsonExtractionRunner(BaseEvalRunner):
    """
    Benchmark: Schema-Constrained Field Extraction via /v1/generate

    - dataset: paraloq/json_data_extraction
    - prompt: strict JSON schema + "JSON only" constraint
    - scoring: json_schema_extraction_scoring.score_json_extraction
    """
    task_name = "generate_paraloq_json_extraction"

    async def _run_impl(self) -> Dict[str, Any]:
        # -----------------------
        # Config
        # -----------------------
        max_examples = self.config.max_examples
        model_override = self.config.model_override

        # Generation params (keep deterministic; this is an eval)
        max_new_tokens = 512
        temperature = 0.0

        # -----------------------
        # Run loop
        # -----------------------
        results: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []

        n_total = 0
        n_http_ok = 0

        # aggregate counters
        n_json_valid = 0
        n_schema_valid = 0
        n_required_present = 0
        n_required_all_correct = 0

        field_match_num = 0
        field_match_den = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            for ex in iter_paraloq_json_extraction(split="train", max_samples=max_examples):
                n_total += 1

                prompt = build_paraloq_json_extraction_prompt(text=ex.text, schema=ex.schema)

                payload: Dict[str, Any] = {
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                # Only include model override if caller set it
                if model_override:
                    payload["model"] = model_override

                url = f"{self.base_url}/v1/generate"
                headers = {"Authorization": f"Bearer {self.api_key}"}

                t0 = time.time()
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    latency_ms = (time.time() - t0) * 1000.0
                    latencies_ms.append(latency_ms)

                    ok = resp.status_code == 200
                    if ok:
                        n_http_ok += 1

                    body = None
                    try:
                        body = resp.json()
                    except Exception:
                        body = resp.text

                    raw_text = _extract_text_from_generate_response(body)
                    predicted_text = _extract_delimited_json_block(raw_text)

                    scored = score_json_extraction(
                        predicted_text=predicted_text,
                        expected=ex.expected,
                        schema=ex.schema,
                    )

                    # Update aggregates
                    if scored.get("json_valid"):
                        n_json_valid += 1
                    if scored.get("schema_valid"):
                        n_schema_valid += 1
                    if scored.get("required_present"):
                        n_required_present += 1
                    if scored.get("required_all_correct"):
                        n_required_all_correct += 1

                    field_exact = scored.get("field_exact_match") or {}
                    # micro-average over expected flattened keys
                    for _, v in field_exact.items():
                        field_match_den += 1
                        if v:
                            field_match_num += 1

                    results.append(
                        {
                            "id": ex.id,
                            "http_ok": ok,
                            "status_code": resp.status_code,
                            "latency_ms": latency_ms,
                            "schema_id": "inline_json_schema",  # dataset provides schema per example
                            "expected": ex.expected,
                            "predicted_text": predicted_text,
                            "metrics": scored,
                        }
                    )

                except Exception as e:
                    latency_ms = (time.time() - t0) * 1000.0
                    latencies_ms.append(latency_ms)

                    results.append(
                        {
                            "id": ex.id,
                            "http_ok": False,
                            "status_code": None,
                            "latency_ms": latency_ms,
                            "schema_id": "inline_json_schema",
                            "expected": ex.expected,
                            "predicted_text": None,
                            "metrics": {
                                "json_valid": False,
                                "schema_valid": False,
                                "field_exact_match": {},
                                "required_present": False,
                                "required_all_correct": False,
                            },
                            "error": str(e),
                        }
                    )

        # -----------------------
        # Summaries
        # -----------------------
        def pct(num: int, den: int) -> float:
            return 0.0 if den == 0 else 100.0 * (num / den)

        latency_p50 = None
        latency_p95 = None
        latency_p99 = None
        if latencies_ms:
            xs = sorted(latencies_ms)

            def q(p: float) -> float:
                if len(xs) == 1:
                    return xs[0]
                pos = (len(xs) - 1) * p
                lo = int(pos)
                hi = min(lo + 1, len(xs) - 1)
                frac = pos - lo
                return xs[lo] + (xs[hi] - xs[lo]) * frac

            latency_p50 = q(0.50)
            latency_p95 = q(0.95)
            latency_p99 = q(0.99)

        summary = {
            "task": self.task_name,
            "dataset": "paraloq/json_data_extraction",
            "split": "train",
            "base_url": self.base_url,
            "model_override": model_override,
            "max_examples": max_examples,
            "generation": {"max_new_tokens": max_new_tokens, "temperature": temperature},
            "n_total": n_total,
            "http_ok_rate": pct(n_http_ok, n_total),
            "json_valid_rate": pct(n_json_valid, n_total),
            "schema_valid_rate": pct(n_schema_valid, n_total),
            "required_present_rate": pct(n_required_present, n_total),
            "required_all_correct_rate": pct(n_required_all_correct, n_total),
            "field_micro_exact_match_rate": pct(field_match_num, field_match_den),
            "latency_p50_ms": latency_p50,
            "latency_p95_ms": latency_p95,
            "latency_p99_ms": latency_p99,
        }

        # Optional: return full results for saving by CLI
        return {
            "summary": summary,
            "results": results,
            "config": asdict(self.config),
        }