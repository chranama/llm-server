# src/llm_server/eval/runners/extraction_runner.py
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from eval.client.http_client import ExtractErr, ExtractOk, HttpEvalClient
from eval.datasets.voxel51_scanned_receipts import (
    DEFAULT_SCHEMA_ID,
    iter_voxel51_scanned_receipts,
)
from eval.metrics.extraction_scoring import (
    ExtractAttempt,
    format_summary,
    score_document,
    summarize_extraction,
)
from eval.runners.base import BaseEvalRunner, EvalConfig


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class ExtractionEvalRunner(BaseEvalRunner):
    """
    Evaluates /v1/extract on Voxel51/scanned_receipts (SROIE-like receipts).

    Writes:
      - results.jsonl (per-doc)
      - summary.json (aggregate)
      - report.txt (human-readable)
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
        outdir: str = "results/extraction_sroie",
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, config=config)
        self.schema_id = schema_id
        self.split = split
        self.outdir = outdir

    async def _run_impl(self) -> Dict[str, Any]:
        client = HttpEvalClient(base_url=self.base_url, api_key=self.api_key)

        run_id = _utc_ts()
        run_dir = os.path.join(self.outdir, run_id)
        _ensure_dir(run_dir)

        # Contract for sroie_receipt_v1.json
        fields = ["company", "address", "date", "total"]
        required = ["company", "date", "total"]

        attempts: List[ExtractAttempt] = []

        results_path = os.path.join(run_dir, "results.jsonl")
        summary_path = os.path.join(run_dir, "summary.json")
        report_path = os.path.join(run_dir, "report.txt")

        with open(results_path, "w", encoding="utf-8") as f:
            for ex in iter_voxel51_scanned_receipts(
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
                else:
                    # Optional: if you expose stage in error.extra later, capture it here
                    stage = None
                    if resp.extra and isinstance(resp.extra, dict):
                        stage = resp.extra.get("stage") or resp.extra.get("error_stage")

                    attempt = ExtractAttempt(
                        doc_id=ex.id,
                        schema_id=ex.schema_id,
                        expected=ex.expected,
                        predicted=None,
                        ok=False,
                        status_code=resp.status_code,
                        error_code=resp.error_code,
                        error_stage=str(stage) if stage else None,
                        repair_attempted=False,
                        cached=False,
                        cache_layer=None,
                        latency_ms=resp.latency_ms,
                    )

                doc_score = score_document(
                    attempt,
                    fields=fields,
                    required_fields=required,
                    ignore_if_expected_missing=True,
                )

                row = {
                    "doc_id": ex.id,
                    "schema_id": ex.schema_id,
                    "ok": attempt.ok,
                    "status_code": attempt.status_code,
                    "error_code": attempt.error_code,
                    "error_stage": attempt.error_stage,
                    "repair_attempted": attempt.repair_attempted,
                    "cached": attempt.cached,
                    "latency_ms": attempt.latency_ms,
                    "expected": ex.expected,
                    "predicted": attempt.predicted,
                    "field_correct": doc_score.field_correct,
                    "required_present_non_null": doc_score.required_present_non_null,
                    "required_all_correct": doc_score.required_all_correct,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                attempts.append(attempt)

        summary = summarize_extraction(
            attempts,
            fields=fields,
            required_fields=required,
            ignore_if_expected_missing=True,
        )

        summary_dict = asdict(summary)
        summary_dict.update(
            {
                "task": self.task_name,
                "run_id": run_id,
                "dataset": "Voxel51/scanned_receipts",
                "split": self.split,
                "schema_id": self.schema_id,
                "base_url": self.base_url,
                "model_override": self.config.model_override,
                "max_examples": self.config.max_examples,
                "results_path": results_path,
                "run_dir": run_dir,
            }
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

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
            format_summary(summary),
        ]
        report_text = "\n".join(report_lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

        # Return concise metrics for CLI printing (optional)
        return {
            "task": self.task_name,
            "run_id": run_id,
            "n_total": summary.n_total,
            "schema_validity_rate": summary.schema_validity_rate,
            "repair_success_rate": summary.repair_success_rate,
            "cache_hit_rate": summary.cache_hit_rate,
            "field_exact_match_rate": summary.field_exact_match_rate,
            "doc_required_exact_match_rate": summary.doc_required_exact_match_rate,
            "required_present_rate": summary.required_present_rate,
            "latency_p50_ms": summary.latency_p50_ms,
            "latency_p95_ms": summary.latency_p95_ms,
            "latency_p99_ms": summary.latency_p99_ms,
            "run_dir": run_dir,
        }


def make_extraction_runner(base_url: str, api_key: str) -> BaseEvalRunner:
    return ExtractionEvalRunner(
        base_url=base_url,
        api_key=api_key,
        config=EvalConfig(),
    )