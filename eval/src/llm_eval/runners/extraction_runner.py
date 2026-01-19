# llm_eval/runners/extraction_runner.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, cast

from llm_eval.client.http_client import ExtractErr, ExtractOk
from llm_eval.datasets.voxel51_scanned_receipts import (
    DEFAULT_SCHEMA_ID,
    iter_voxel51_scanned_receipts,
)
from llm_eval.metrics.extraction_scoring import (
    ExtractAttempt,
    format_summary,
    score_document,
    summarize_extraction,
)
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


class ExtractionEvalRunner(BaseEvalRunner):
    """
    Evaluates /v1/extract on Voxel51/scanned_receipts (SROIE-like receipts).

    Pure runner: no filesystem writes.
    Returns nested payload:
      - summary: aggregate metrics
      - results: per-doc rows
      - report_text: human-readable report
      - config: runner config snapshot
    """

    task_name = "extraction_sroie"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: Optional[EvalConfig] = None,
        *,
        schema_id: str = DEFAULT_SCHEMA_ID,
        split: str = "train",
        deps=None,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, config=config, deps=deps)
        self.schema_id = schema_id
        self.split = split

    async def _run_impl(self) -> Dict[str, Any]:
        client = self.make_client()
        run_id = self.new_run_id()

        # Contract for sroie_receipt_v1.json
        fields = ["company", "address", "date", "total"]
        required = ["company", "date", "total"]

        attempts: List[ExtractAttempt] = []
        results: List[Dict[str, Any]] = []

        # --- dataset seam (patchable in tests) ---
        # Stable key: "voxel51_scanned_receipts"
        iter_fn = self.get_dataset_callable("iter_voxel51_scanned_receipts", iter_voxel51_scanned_receipts)
        iter_fn = cast(Any, iter_fn)

        for ex in iter_fn(
            split=self.split,
            schema_id=self.schema_id,
            max_samples=self.config.max_examples,
        ):
            resp = await client.extract(
                schema_id=ex.schema_id,
                text=ex.text,
                model=self.config.model_override,
                temperature=0.0,
                max_new_tokens=512,
                cache=False,
                repair=True,
            )

            # Protocol guarantees ExtractOk | ExtractErr with latency_ms
            if isinstance(resp, ExtractOk):
                attempt = ExtractAttempt(
                    doc_id=ex.id,
                    schema_id=ex.schema_id,
                    expected=ex.expected,
                    predicted=resp.data,
                    ok=True,
                    status_code=200,
                    error_code=None,
                    error_stage=None,
                    repair_attempted=resp.repair_attempted,
                    cached=resp.cached,
                    cache_layer=None,
                    latency_ms=resp.latency_ms,
                )
                model_id: Optional[str] = resp.model
                extra: Optional[Dict[str, Any]] = None
            else:
                assert isinstance(resp, ExtractErr)

                stage: Optional[str] = None
                if isinstance(resp.extra, dict):
                    stage_val = resp.extra.get("stage") or resp.extra.get("error_stage")
                    if stage_val is not None:
                        stage = str(stage_val)

                attempt = ExtractAttempt(
                    doc_id=ex.id,
                    schema_id=ex.schema_id,
                    expected=ex.expected,
                    predicted=None,
                    ok=False,
                    status_code=resp.status_code,
                    error_code=resp.error_code,
                    error_stage=stage,
                    repair_attempted=False,
                    cached=False,
                    cache_layer=None,
                    latency_ms=resp.latency_ms,
                )
                model_id = None
                extra = resp.extra if isinstance(resp.extra, dict) else None

            doc_score = score_document(
                attempt,
                fields=fields,
                required_fields=required,
                ignore_if_expected_missing=True,
            )

            results.append(
                {
                    "doc_id": ex.id,
                    "schema_id": ex.schema_id,
                    "ok": attempt.ok,
                    "status_code": attempt.status_code,
                    "error_code": attempt.error_code,
                    "error_stage": attempt.error_stage,
                    "extra": extra,
                    "repair_attempted": attempt.repair_attempted,
                    "cached": attempt.cached,
                    "latency_ms": attempt.latency_ms,
                    "model": model_id,
                    "expected": ex.expected,
                    "predicted": attempt.predicted,
                    "field_correct": doc_score.field_correct,
                    "required_present_non_null": doc_score.required_present_non_null,
                    "required_all_correct": doc_score.required_all_correct,
                }
            )

            attempts.append(attempt)

        summary_obj = summarize_extraction(
            attempts,
            fields=fields,
            required_fields=required,
            ignore_if_expected_missing=True,
        )

        summary = asdict(summary_obj)
        summary.update(
            {
                "task": self.task_name,
                "run_id": run_id,
                "dataset": "Voxel51/scanned_receipts",
                "split": self.split,
                "schema_id": self.schema_id,
                "base_url": self.base_url,
                "model_override": self.config.model_override,
                "max_examples": self.config.max_examples,
            }
        )

        report_lines = [
            f"task={self.task_name}",
            f"run_id={run_id}",
            "dataset=Voxel51/scanned_receipts",
            f"split={self.split}",
            f"schema_id={self.schema_id}",
            f"base_url={self.base_url}",
            f"model_override={self.config.model_override}",
            f"max_examples={self.config.max_examples}",
            "",
            format_summary(summary_obj),
        ]
        report_text = "\n".join(report_lines)

        return {
            "summary": summary,
            "results": results,
            "report_text": report_text,
            "config": asdict(self.config),
        }


def make_extraction_runner(
    base_url: str,
    api_key: str,
    config: Optional[EvalConfig] = None,
) -> BaseEvalRunner:
    return ExtractionEvalRunner(
        base_url=base_url,
        api_key=api_key,
        config=config or EvalConfig(),
    )