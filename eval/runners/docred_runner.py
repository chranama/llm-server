# src/llm_server/eval/runners/docred_runner.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx

from eval.datasets.docred import iter_docred
from eval.metrics.docred_scoring import (
    aggregate_docred_scores,
    parse_predicted_maybe_json,
    score_docred_example,
)
from eval.prompts.docred_prompt import build_docred_prompt
from eval.runners.base import BaseEvalRunner


class GenerateDocREDRunner(BaseEvalRunner):
    """
    Benchmarks /v1/generate on DocRED using entity-conditioned relation extraction.

    This is a good /generate proxy for /extract because it tests:
      - instruction following (strict JSON)
      - schema-like structure (relations list)
      - long-context reasoning
      - cross-sentence relation extraction
    """

    task_name = "generate_docred"

    async def _run_impl(self) -> Dict[str, Any]:
        max_examples = self.config.max_examples
        model_override = self.config.model_override

        # validation is the cleanest dev split (labels exist)
        split = "validation"
        examples = list(iter_docred(split=split, max_samples=max_examples))

        url = f"{self.base_url}/v1/generate"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        per_example_scores: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for ex in examples:
                expected = ex.expected if isinstance(ex.expected, dict) else {}
                title = expected.get("title")
                entities = expected.get("entities") if isinstance(expected.get("entities"), list) else []

                prompt = build_docred_prompt(
                    text=ex.text,
                    title=title if isinstance(title, str) else None,
                    entities=entities,
                )

                payload: Dict[str, Any] = {
                    "prompt": prompt,
                    "max_new_tokens": 512,
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

                    # minimal assumptions about response payload
                    pred_raw: Any = None
                    if isinstance(data, dict):
                        pred_raw = (
                            data.get("text")
                            or data.get("output")
                            or data.get("completion")
                            or data.get("data")
                        )

                    pred_obj = parse_predicted_maybe_json(pred_raw)

                    score = score_docred_example(expected=expected, predicted=pred_obj)
                    per_example_scores.append(score)

                    # keep a short preview for debugging
                    if isinstance(pred_raw, str):
                        pred_preview = pred_raw[:2000]
                    else:
                        pred_preview = json.dumps(pred_raw, ensure_ascii=False)[:2000]

                    results.append(
                        {
                            "id": ex.id,
                            "split": split,
                            "n_entities": len(entities),
                            "n_gold_relations": score.get("n_gold"),
                            "n_pred_relations": score.get("n_pred"),
                            "predicted_preview": pred_preview,
                            "metrics": score,
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

        agg = aggregate_docred_scores(per_example_scores)

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

        return {
            "task": self.task_name,
            "dataset": "thunlp/docred",
            "split": split,
            "base_url": self.base_url,
            "model_override": model_override,
            "max_examples": max_examples,
            **agg,
            "latency_p50_ms": _quantile(latencies_ms, 0.50),
            "latency_p95_ms": _quantile(latencies_ms, 0.95),
            "latency_p99_ms": _quantile(latencies_ms, 0.99),
            "results": results,
        }