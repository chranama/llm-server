# src/llm_server/eval/runners/squad_v2_runner.py
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import httpx

from eval.datasets.squad_v2 import iter_squad_v2
from eval.metrics.squad_v2_scoring import score_squad_v2_example, summarize_squad_v2
from eval.prompts.squad_v2_prompt import NO_ANSWER_TOKEN, build_squad_v2_prompt
from eval.runners.base import BaseEvalRunner


class GenerateSquadV2Runner(BaseEvalRunner):
    """
    Benchmarks /v1/generate on SQuAD v2.

    Contract:
      - Answerable: output the short answer string only
      - Unanswerable: output exactly NO_ANSWER_TOKEN (from prompts module)
    """

    task_name = "generate_squad_v2"

    async def _run_impl(self) -> Dict[str, Any]:
        max_examples = self.config.max_examples
        model_override = self.config.model_override

        examples = list(iter_squad_v2(split="validation", max_samples=max_examples))

        latencies_ms: List[float] = []
        per_example_scores = []  # List[SquadV2ExampleScore]
        results: List[Dict[str, Any]] = []

        url = f"{self.base_url}/v1/generate"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            for ex in examples:
                prompt = build_squad_v2_prompt(
                    context=ex.context,
                    question=ex.question,
                    title=ex.title,
                )

                payload: Dict[str, Any] = {
                    "prompt": prompt,
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                }
                if model_override:
                    payload["model"] = model_override

                t0 = time.time()
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    dt_ms = (time.time() - t0) * 1000.0
                    latencies_ms.append(dt_ms)

                    if resp.status_code != 200:
                        results.append(
                            {
                                "id": ex.id,
                                "status_code": resp.status_code,
                                "error": resp.text[:500],
                                "latency_ms": dt_ms,
                            }
                        )
                        continue

                    data = resp.json()

                    # Minimal assumptions about response shape
                    pred = None
                    if isinstance(data, dict):
                        pred = (
                            data.get("text")
                            or data.get("output")
                            or data.get("completion")
                            or data.get("data")
                        )

                    predicted_text = (pred or "").strip()

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
                            "latency_ms": dt_ms,
                        }
                    )

                except Exception as e:
                    dt_ms = (time.time() - t0) * 1000.0
                    latencies_ms.append(dt_ms)
                    results.append(
                        {
                            "id": ex.id,
                            "status_code": None,
                            "error": f"{type(e).__name__}: {e}",
                            "latency_ms": dt_ms,
                        }
                    )

        # lightweight percentiles (no numpy)
        def _quantile(vals: List[float], q: float) -> Optional[float]:
            if not vals:
                return None
            xs = sorted(vals)
            if q <= 0:
                return xs[0]
            if q >= 1:
                return xs[-1]
            pos = (len(xs) - 1) * q
            lo = int(pos)
            hi = min(lo + 1, len(xs) - 1)
            if lo == hi:
                return xs[lo]
            frac = pos - lo
            return xs[lo] + (xs[hi] - xs[lo]) * frac

        summary_metrics = summarize_squad_v2(per_example_scores)

        summary = {
            "task": self.task_name,
            "dataset": "squad_v2",
            "split": "validation",
            "base_url": self.base_url,
            "model_override": model_override,
            "max_examples": max_examples,
            **summary_metrics,
            "latency_p50_ms": _quantile(latencies_ms, 0.50),
            "latency_p95_ms": _quantile(latencies_ms, 0.95),
            "latency_p99_ms": _quantile(latencies_ms, 0.99),
            "results": results,
        }

        return summary