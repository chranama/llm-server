# llm_eval/runners/squad_v2_runner.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, cast

from llm_eval.client.http_client import GenerateErr, GenerateOk
from llm_eval.datasets.squad_v2 import iter_squad_v2
from llm_eval.metrics.common import quantile
from llm_eval.metrics.squad_v2_scoring import score_squad_v2_example, summarize_squad_v2
from llm_eval.prompts.squad_v2_prompt import NO_ANSWER_TOKEN, build_squad_v2_prompt
from llm_eval.runners.base import BaseEvalRunner


class GenerateSquadV2Runner(BaseEvalRunner):
    """
    Benchmarks /v1/generate on SQuAD v2.

    Contract:
      - Answerable: output the short answer string only
      - Unanswerable: output exactly NO_ANSWER_TOKEN

    Pure runner: no filesystem writes.
    Returns nested payload:
      - summary: aggregate metrics
      - results: per-example rows
      - report_text: lightweight human-readable report
      - config: runner config snapshot
    """

    task_name = "generate_squad_v2"

    async def _run_impl(self) -> Dict[str, Any]:
        client = self.make_client()

        max_examples = self.config.max_examples
        model_override = self.config.model_override

        split = "validation"

        # --- dataset seam (patchable in tests) ---
        # Stable key: "squad_v2"
        iter_fn = self.get_dataset_callable("iter_squad_v2", iter_squad_v2)
        iter_fn = cast(Any, iter_fn)  # keep runner light; tests can inject any callable
        examples = list(iter_fn(split=split, max_samples=max_examples))

        latencies_ms: List[float] = []
        per_example_scores = []
        results: List[Dict[str, Any]] = []

        for ex in examples:
            prompt = build_squad_v2_prompt(
                context=ex.context,
                question=ex.question,
                title=ex.title,
            )

            resp = await client.generate(
                prompt=prompt,
                max_new_tokens=64,
                temperature=0.0,
                model=model_override,
            )

            # Protocol guarantees latency_ms on both ok/err
            latencies_ms.append(resp.latency_ms)

            if isinstance(resp, GenerateErr):
                results.append(
                    {
                        "id": ex.id,
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

            predicted_text = (resp.output_text or "").strip()

            score = score_squad_v2_example(
                predicted=predicted_text,
                answers=ex.answers,
                is_impossible=ex.is_impossible,
                no_answer_token=NO_ANSWER_TOKEN,
            )
            per_example_scores.append(score)

            results.append(
                {
                    "id": ex.id,
                    "title": ex.title,
                    "question": ex.question,
                    "is_impossible": ex.is_impossible,
                    "answers": ex.answers[:5],
                    "predicted": predicted_text,
                    "metrics": asdict(score),
                    "latency_ms": resp.latency_ms,
                    "model": resp.model,
                }
            )

        summary_metrics = summarize_squad_v2(per_example_scores)

        run_id = self.new_run_id()
        summary: Dict[str, Any] = {
            "task": self.task_name,
            "run_id": run_id,
            "dataset": "squad_v2",
            "split": split,
            "base_url": self.base_url,
            "model_override": model_override,
            "max_examples": max_examples,
            **summary_metrics,
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
        for k in ("answerable_exact_match_rate", "unanswerable_accuracy", "combined_score"):
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