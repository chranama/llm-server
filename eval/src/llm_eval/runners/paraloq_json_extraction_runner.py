# llm_eval/runners/paraloq_json_extraction_runner.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, cast

from llm_eval.client.http_client import GenerateErr, GenerateOk
from llm_eval.datasets.paraloq_json_extraction import iter_paraloq_json_extraction
from llm_eval.metrics.common import quantile
from llm_eval.metrics.json_schema_extraction_scoring import score_json_extraction
from llm_eval.prompts.paraloq_json_extraction_prompt import (
    _JSON_BEGIN,
    _JSON_END,
    build_paraloq_json_extraction_prompt,
)
from llm_eval.runners.base import BaseEvalRunner


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if not s.startswith("```"):
        return s
    s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _extract_delimited_json_block(s: str) -> str:
    if not s:
        return s
    if _JSON_BEGIN in s and _JSON_END in s:
        inner = s.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0]
        return _strip_code_fences(inner.strip())
    return _strip_code_fences(s)


def _pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else 100.0 * (num / den)


class GenerateParaloqJsonExtractionRunner(BaseEvalRunner):
    """
    Benchmark: Schema-Constrained Field Extraction via /v1/generate

    - dataset: paraloq/json_data_extraction
    - prompt: strict JSON schema + "JSON only" constraint
    - scoring: score_json_extraction

    Pure runner: no filesystem writes.
    Returns nested payload:
      - summary: aggregate metrics
      - results: per-example rows
      - report_text: lightweight human-readable report
      - config: runner config snapshot
    """

    task_name = "generate_paraloq_json_extraction"

    async def _run_impl(self) -> Dict[str, Any]:
        client = self.make_client()

        max_examples = self.config.max_examples
        model_override = self.config.model_override

        max_new_tokens = 512
        temperature = 0.0

        results: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []

        n_total = 0
        n_http_ok = 0

        n_json_valid = 0
        n_schema_valid = 0
        n_required_present = 0
        n_required_all_correct = 0

        field_match_num = 0
        field_match_den = 0

        split = "train"
        dataset_name = "paraloq/json_data_extraction"
        run_id = self.new_run_id()

        # --- dataset seam (patchable in tests) ---
        # Stable key: "paraloq_json_extraction"
        iter_fn = self.get_dataset_callable("iter_paraloq_json_extraction", iter_paraloq_json_extraction)
        iter_fn = cast(Any, iter_fn)

        for ex in iter_fn(split=split, max_samples=max_examples):
            n_total += 1

            prompt = build_paraloq_json_extraction_prompt(text=ex.text, schema=ex.schema)

            resp = await client.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                model=model_override,
            )

            # Protocol guarantees latency_ms on both ok/err
            latencies_ms.append(resp.latency_ms)

            if isinstance(resp, GenerateErr):
                predicted_text = ""  # keep consistent with scoring

                scored = score_json_extraction(
                    predicted_text=predicted_text,
                    expected=ex.expected,
                    schema=ex.schema,
                )

                results.append(
                    {
                        "id": ex.id,
                        "http_ok": False,
                        "status_code": resp.status_code,
                        "error_code": resp.error_code,
                        "error": resp.message,
                        "extra": resp.extra if isinstance(resp.extra, dict) else None,
                        "latency_ms": resp.latency_ms,
                        "model": None,
                        "schema_id": "inline_json_schema",
                        "expected": ex.expected,
                        "predicted_text": predicted_text,
                        "metrics": scored,
                    }
                )
                continue

            assert isinstance(resp, GenerateOk)

            n_http_ok += 1

            predicted_text = _extract_delimited_json_block(resp.output_text or "")

            scored = score_json_extraction(
                predicted_text=predicted_text,
                expected=ex.expected,
                schema=ex.schema,
            )

            if scored.get("json_valid"):
                n_json_valid += 1
            if scored.get("schema_valid"):
                n_schema_valid += 1
            if scored.get("required_present"):
                n_required_present += 1
            if scored.get("required_all_correct"):
                n_required_all_correct += 1

            field_exact = scored.get("field_exact_match") or {}
            for _, v in field_exact.items():
                field_match_den += 1
                if v:
                    field_match_num += 1

            results.append(
                {
                    "id": ex.id,
                    "http_ok": True,
                    "status_code": 200,
                    "latency_ms": resp.latency_ms,
                    "model": resp.model,
                    "schema_id": "inline_json_schema",
                    "expected": ex.expected,
                    "predicted_text": predicted_text,
                    "metrics": scored,
                }
            )

        summary: Dict[str, Any] = {
            "task": self.task_name,
            "run_id": run_id,
            "dataset": dataset_name,
            "split": split,
            "base_url": self.base_url,
            "model_override": model_override,
            "max_examples": max_examples,
            "generation": {"max_new_tokens": max_new_tokens, "temperature": temperature},
            "n_total": n_total,
            "http_ok_rate": _pct(n_http_ok, n_total),
            "json_valid_rate": _pct(n_json_valid, n_total),
            "schema_valid_rate": _pct(n_schema_valid, n_total),
            "required_present_rate": _pct(n_required_present, n_total),
            "required_all_correct_rate": _pct(n_required_all_correct, n_total),
            "field_micro_exact_match_rate": _pct(field_match_num, field_match_den),
            "latency_p50_ms": quantile(latencies_ms, 0.50),
            "latency_p95_ms": quantile(latencies_ms, 0.95),
            "latency_p99_ms": quantile(latencies_ms, 0.99),
        }

        report_text = "\n".join(
            [
                f"task={self.task_name}",
                f"run_id={run_id}",
                f"dataset={dataset_name}",
                f"split={split}",
                f"base_url={self.base_url}",
                f"model_override={model_override}",
                f"max_examples={max_examples}",
                "",
                f"http_ok_rate={summary['http_ok_rate']:.2f}%",
                f"json_valid_rate={summary['json_valid_rate']:.2f}%",
                f"schema_valid_rate={summary['schema_valid_rate']:.2f}%",
                f"required_present_rate={summary['required_present_rate']:.2f}%",
                f"required_all_correct_rate={summary['required_all_correct_rate']:.2f}%",
                f"field_micro_exact_match_rate={summary['field_micro_exact_match_rate']:.2f}%",
                f"latency_p50_ms={summary['latency_p50_ms']}",
                f"latency_p95_ms={summary['latency_p95_ms']}",
                f"latency_p99_ms={summary['latency_p99_ms']}",
            ]
        )

        return {
            "summary": summary,
            "results": results,
            "report_text": report_text,
            "config": asdict(self.config),
        }