# src/llm_server/eval/cli.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from eval.runners.base import BaseEvalRunner, EvalConfig

# --- runners (current suite) ---
from eval.runners.extraction_runner import make_extraction_runner
from eval.runners.paraloq_json_extraction_runner import GenerateParaloqJsonExtractionRunner
from eval.runners.squad_v2_runner import GenerateSquadV2Runner
from eval.runners.docred_runner import GenerateDocREDRunner


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {"value": x}


def _shorten(text: str, n: int) -> str:
    s = (text or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _extract_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    r = summary.get("results")
    return r if isinstance(r, list) else []


def _default_outdir(task: str, run_id: str) -> str:
    # keep it inside the package tree (mirrors your existing layout)
    return os.path.join("src", "llm_server", "eval", "results", task, run_id)


# Map task names to runner factory functions
TASK_FACTORIES: dict[str, Callable[[str, str, Optional[EvalConfig]], BaseEvalRunner]] = {
    # /v1/extract benchmark
    "extraction_sroie": lambda base_url, api_key, cfg: make_extraction_runner(base_url=base_url, api_key=api_key, config=cfg),
    # /v1/generate benchmarks (text-only)
    "generate_paraloq_json_extraction": lambda base_url, api_key, cfg: GenerateParaloqJsonExtractionRunner(
        base_url=base_url, api_key=api_key, config=cfg
    ),
    "generate_squad_v2": lambda base_url, api_key, cfg: GenerateSquadV2Runner(
        base_url=base_url, api_key=api_key, config=cfg
    ),
    "generate_docred": lambda base_url, api_key, cfg: GenerateDocREDRunner(
        base_url=base_url, api_key=api_key, config=cfg
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evaluation tasks against an llm-server instance.",
    )

    parser.add_argument(
        "--task",
        required=False,
        choices=sorted(TASK_FACTORIES.keys()),
        help="Which evaluation task to run.",
    )

    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit.",
    )

    parser.add_argument(
        "--base-url",
        required=False,
        help="Base URL of the llm-server API (e.g. http://localhost:8000).",
    )

    parser.add_argument(
        "--api-key",
        required=False,
        help="API key used to call /v1/generate or /v1/extract.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all available).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override identifier (if your server supports it).",
    )

    # ---- debugging / output controls ----
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the summary JSON to stdout (default: on).",
    )
    parser.add_argument(
        "--no-print-summary",
        action="store_true",
        help="Disable printing the summary JSON.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save summary.json and (if present) results.jsonl under eval/results/<task>/<run_id>/ (default: on).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving summary/results files.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Override output directory (default: results/<task>/<run_id>/).",
    )

    parser.add_argument(
        "--debug-n",
        type=int,
        default=0,
        help="Print the first N per-example results (if available) for quick debugging.",
    )

    parser.add_argument(
        "--debug-fields",
        type=str,
        default=None,
        help=(
            "Comma-separated keys to display for each debug result row "
            "(e.g. 'id,status_code,latency_ms,predicted'). If omitted, uses a sensible default."
        ),
    )

    return parser


async def amain() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tasks:
        for t in sorted(TASK_FACTORIES.keys()):
            print(t)
        return

    if not args.task:
        parser.error("--task is required (or use --list-tasks).")

    if not args.base_url:
        parser.error("--base-url is required.")

    if not args.api_key:
        parser.error("--api-key is required.")

    if args.task not in TASK_FACTORIES:
        parser.error(f"Unknown task '{args.task}'. Valid options: {', '.join(sorted(TASK_FACTORIES.keys()))}")

    # defaults: print + save ON unless explicitly disabled
    do_print = True
    if args.print_summary:
        do_print = True
    if args.no_print_summary:
        do_print = False

    do_save = True
    if args.save:
        do_save = True
    if args.no_save:
        do_save = False

    cfg = EvalConfig(max_examples=args.max_examples, model_override=args.model)

    runner_factory = TASK_FACTORIES[args.task]
    runner = runner_factory(args.base_url, args.api_key, cfg)

    summary_any = await runner.run(
        max_examples=args.max_examples,
        model_override=args.model,
    )

    summary = _as_dict(summary_any)

    # Ensure task + run_id present for consistent reporting
    task_name = str(summary.get("task") or args.task)
    run_id = str(summary.get("run_id") or _utc_run_id())
    summary["task"] = task_name
    summary["run_id"] = run_id

    # Save artifacts
    outdir = args.outdir or _default_outdir(task_name, run_id)

    if do_save:
        _ensure_dir(outdir)

        # Write summary.json
        summary_path = os.path.join(outdir, "summary.json")
        _write_json(summary_path, summary)

        # Write results.jsonl if present
        results = _extract_results(summary)
        if results:
            results_path = os.path.join(outdir, "results.jsonl")
            _write_jsonl(results_path, results)

        # Write a tiny report.txt for quick grepping
        report_lines = []
        report_lines.append(f"task={task_name}")
        report_lines.append(f"run_id={run_id}")
        report_lines.append(f"base_url={summary.get('base_url', args.base_url)}")
        if summary.get("dataset"):
            report_lines.append(f"dataset={summary.get('dataset')}")
        if summary.get("split"):
            report_lines.append(f"split={summary.get('split')}")
        if summary.get("max_examples") is not None:
            report_lines.append(f"max_examples={summary.get('max_examples')}")

        # common score fields (if present)
        for k in [
            "schema_validity_rate",
            "doc_required_exact_match_rate",
            "required_present_rate",
            "answerable_exact_match_rate",
            "unanswerable_accuracy",
            "combined_score",
            "precision",
            "recall",
            "f1",
        ]:
            if k in summary:
                report_lines.append(f"{k}={summary.get(k)}")

        for k in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
            if k in summary and summary.get(k) is not None:
                report_lines.append(f"{k}={summary.get(k)}")

        with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        # surface where things went
        summary["run_dir"] = outdir

    # Optional debug print of first N results
    if args.debug_n and args.debug_n > 0:
        results = _extract_results(summary)
        fields = None
        if args.debug_fields:
            fields = [x.strip() for x in args.debug_fields.split(",") if x.strip()]

        if not fields:
            # good generic defaults across your runners
            fields = ["id", "status_code", "latency_ms", "metrics", "predicted", "predicted_preview", "error"]

        print("\n" + "=" * 80)
        print(f"DEBUG (first {min(args.debug_n, len(results))} results)")
        print("=" * 80)
        for row in results[: args.debug_n]:
            if not isinstance(row, dict):
                continue
            compact = {}
            for k in fields:
                v = row.get(k)
                # keep output readable
                if isinstance(v, str):
                    compact[k] = _shorten(v, 240)
                else:
                    compact[k] = v
            print(json.dumps(compact, ensure_ascii=False, indent=2))
            print("-" * 80)

    if do_print:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()