# llm_eval/runners/docred_runner.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, cast

from llm_eval.client.http_client import GenerateErr, GenerateOk
from llm_eval.datasets.docred_relation_extraction import iter_docred
from llm_eval.metrics.common import quantile
from llm_eval.metrics.docred_relation_scoring import (
    aggregate_docred_scores,
    parse_predicted_maybe_json,
    score_docred_example,
)
from llm_eval.prompts.docred_prompt import build_docred_prompt
from llm_eval.runners.base import BaseEvalRunner


class GenerateDocREDRunner(BaseEvalRunner):
    """
    Benchmarks /v1/generate on DocRED using entity-conditioned relation extraction.

    Pure runner: no filesystem writes.
    Returns nested payload:
      - summary: aggregate metrics
      - results: per-example rows
      - report_text: lightweight human-readable report
      - config: runner config snapshot
    """

    task_name = "generate_docred"

    async def _run_impl(self) -> Dict[str, Any]:
        client = self.make_client()

        max_examples = self.config.max_examples
        model_override = self.config.model_override

        split = "validation"

        # --- dataset seam (patchable in tests) ---
        # Stable key: "docred" (tests can override without touching imports)
        iter_fn = self.get_dataset_callable("iter_docred", iter_docred)
        iter_fn = cast(Any, iter_fn)  # keep runner light; tests can inject any callable
        examples = list(iter_fn(split=split, max_samples=max_examples))

        per_example_scores: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []

        for ex in examples:
            expected = ex.expected if isinstance(ex.expected, dict) else {}
            title = expected.get("title")
            entities = expected.get("entities") if isinstance(expected.get("entities"), list) else []

            prompt = build_docred_prompt(
                text=ex.text,
                title=title if isinstance(title, str) else None,
                entities=entities,
            )

            resp = await client.generate(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.0,
                model=model_override,
            )

            # Protocol guarantees latency_ms on both ok/err
            latencies_ms.append(resp.latency_ms)

            if isinstance(resp, GenerateErr):
                results.append(
                    {
                        "id": ex.id,
                        "split": split,
                        "n_entities": len(entities),
                        "status_code": resp.status_code,
                        "error_code": resp.error_code,
                        "error": (resp.message or "")[:500],
                        "extra": resp.extra if isinstance(resp.extra, dict) else None,
                        "latency_ms": resp.latency_ms,
                        "model": None,
                    }
                )
                continue

            assert isinstance(resp, GenerateOk)

            pred_raw = resp.output_text or ""
            pred_obj = parse_predicted_maybe_json(pred_raw)

            score = score_docred_example(expected=expected, predicted=pred_obj)
            per_example_scores.append(score)

            results.append(
                {
                    "id": ex.id,
                    "split": split,
                    "n_entities": len(entities),
                    "n_gold_relations": score.get("n_gold"),
                    "n_pred_relations": score.get("n_pred"),
                    "predicted_preview": pred_raw[:2000],
                    "metrics": score,
                    "latency_ms": resp.latency_ms,
                    "model": resp.model,
                }
            )

        agg = aggregate_docred_scores(per_example_scores)

        run_id = self.new_run_id()
        summary: Dict[str, Any] = {
            "task": self.task_name,
            "run_id": run_id,
            "dataset": "thunlp/docred",
            "split": split,
            "base_url": self.base_url,
            "model_override": model_override,
            "max_examples": max_examples,
            **agg,
            "latency_p50_ms": quantile(latencies_ms, 0.50),
            "latency_p95_ms": quantile(latencies_ms, 0.95),
            "latency_p99_ms": quantile(latencies_ms, 0.99),
        }

        report_lines = [
            f"task={self.task_name}",
            f"run_id={run_id}",
            f"dataset={summary['dataset']}",
            f"split={summary['split']}",
            f"base_url={summary['base_url']}",
            f"model_override={summary.get('model_override')}",
            f"max_examples={summary.get('max_examples')}",
            "",
        ]
        for k in ("precision", "recall", "f1"):
            if k in summary:
                report_lines.append(f"{k}={summary.get(k)}")
        for k in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
            if summary.get(k) is not None:
                report_lines.append(f"{k}={summary.get(k)}")
        report_text = "\n".join(report_lines)

        return {
            "summary": summary,
            "results": results,
            "report_text": report_text,
            "config": asdict(self.config),
        }