# eval/src/llm_eval/cli.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from llm_eval.config import dig, get_api_key, load_eval_yaml
from llm_eval.io.run_artifacts import (
    default_outdir,
    write_eval_run_artifacts,
)
from llm_eval.io.run_pointer import (
    default_eval_out_pointer_path,
    write_eval_latest_pointer,
)
from llm_eval.reports.writer import render_reports_bundle
from llm_eval.runners.base import BaseEvalRunner, EvalConfig


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _shorten(text: str, n: int) -> str:
    s = (text or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _coerce_nested_payload(
    payload_any: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], Optional[str], Optional[dict[str, Any]]]:
    """
    Required runner contract:

      {
        "summary": dict,
        "results": list[dict],
        "report_text": str | None,   # optional
        "config": dict | None,       # optional
      }

    We fail fast if this shape is not respected.
    """
    if not isinstance(payload_any, dict):
        raise TypeError(
            f"Runner returned {type(payload_any).__name__}, expected dict with keys summary/results/..."
        )

    summary_any = payload_any.get("summary")
    if not isinstance(summary_any, dict):
        raise TypeError("Runner output must include key 'summary' with a dict value")

    results_any = payload_any.get("results", [])
    if results_any is None:
        results_any = []
    if not isinstance(results_any, list):
        raise TypeError("Runner output 'results' must be a list (or omitted)")

    # enforce list[dict] (softly: drop non-dicts rather than explode)
    results: list[dict[str, Any]] = [r for r in results_any if isinstance(r, dict)]

    report_text_any = payload_any.get("report_text")
    report_text = report_text_any if isinstance(report_text_any, str) and report_text_any.strip() else None

    cfg_any = payload_any.get("config")
    returned_config = cfg_any if isinstance(cfg_any, dict) else None

    summary: dict[str, Any] = dict(summary_any)
    return summary, results, report_text, returned_config


# -----------------------
# Lazy task factories
# -----------------------
# Key point: DO NOT import runners at module import time.
# Import them only inside the factory for the selected task.

TaskFactory = Callable[[str, str, Optional[EvalConfig]], BaseEvalRunner]


def _task_factories() -> dict[str, TaskFactory]:
    def extraction_sroie(base_url: str, api_key: str, cfg: Optional[EvalConfig]) -> BaseEvalRunner:
        from llm_eval.runners.extraction_runner import make_extraction_runner

        return make_extraction_runner(base_url=base_url, api_key=api_key, config=cfg)

    def generate_paraloq_json_extraction(base_url: str, api_key: str, cfg: Optional[EvalConfig]) -> BaseEvalRunner:
        from llm_eval.runners.paraloq_json_extraction_runner import GenerateParaloqJsonExtractionRunner

        return GenerateParaloqJsonExtractionRunner(base_url=base_url, api_key=api_key, config=cfg)

    def generate_squad_v2(base_url: str, api_key: str, cfg: Optional[EvalConfig]) -> BaseEvalRunner:
        from llm_eval.runners.squad_v2_runner import GenerateSquadV2Runner

        return GenerateSquadV2Runner(base_url=base_url, api_key=api_key, config=cfg)

    def generate_docred(base_url: str, api_key: str, cfg: Optional[EvalConfig]) -> BaseEvalRunner:
        from llm_eval.runners.docred_runner import GenerateDocREDRunner

        return GenerateDocREDRunner(base_url=base_url, api_key=api_key, config=cfg)

    return {
        "extraction_sroie": extraction_sroie,
        "generate_paraloq_json_extraction": generate_paraloq_json_extraction,
        "generate_squad_v2": generate_squad_v2,
        "generate_docred": generate_docred,
    }


def build_parser() -> argparse.ArgumentParser:
    task_names = sorted(_task_factories().keys())

    parser = argparse.ArgumentParser(
        description="Run evaluation tasks against an llm-server instance.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("EVAL_CONFIG_PATH", "config/eval.yaml"),
        help="Path to eval YAML config (default: $EVAL_CONFIG_PATH or config/eval.yaml).",
    )

    parser.add_argument(
        "--task",
        required=False,
        choices=task_names,
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
        help=(
            "Base URL of the llm-server API (e.g. http://localhost:8000). "
            "If omitted, uses service.base_url from eval.yaml."
        ),
    )

    parser.add_argument(
        "--api-key",
        required=False,
        help=(
            "API key used to call /v1/generate or /v1/extract. "
            "If omitted, uses env var named by service.api_key_env in eval.yaml."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Maximum number of examples to evaluate (default: all available). "
            "If omitted, may use datasets.<task>.max_items from eval.yaml."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional model override identifier (if your server supports it). "
            "If omitted, may use defaults.model_id from eval.yaml."
        ),
    )

    # ---- debugging / output controls ----
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Force printing the summary JSON to stdout (default: on).",
    )
    parser.add_argument(
        "--no-print-summary",
        action="store_true",
        help="Disable printing the summary JSON.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Force saving summary.json and results.jsonl (default: on).",
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


def _env_flag(name: str, default: bool) -> bool:
    """
    Parse env var as a boolean-ish flag.
    Accepts: 1/0, true/false, yes/no, on/off (case-insensitive).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


async def amain(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_yaml = load_eval_yaml(args.config)
    factories = _task_factories()

    if args.list_tasks:
        for t in sorted(factories.keys()):
            print(t)
        return

    if not args.task:
        parser.error("--task is required (or use --list-tasks).")

    if args.task not in factories:
        parser.error(f"Unknown task '{args.task}'.")

    enabled = dig(cfg_yaml, "datasets", args.task, "enabled", default=True)
    if enabled is False:
        parser.error(f"Task '{args.task}' is disabled in eval.yaml (datasets.{args.task}.enabled=false).")

    base_url = args.base_url or dig(cfg_yaml, "service", "base_url", default=None)
    if not base_url:
        parser.error("--base-url is required (or set service.base_url in eval.yaml).")

    api_key = args.api_key or get_api_key(cfg_yaml)
    if not api_key:
        parser.error("--api-key is required (or set service.api_key_env in eval.yaml and export it).")

    # defaults: print + save ON unless explicitly disabled
    do_print = True
    if args.no_print_summary:
        do_print = False
    if args.print_summary:
        do_print = True

    do_save = True
    if args.no_save:
        do_save = False
    if args.save:
        do_save = True

    max_examples = args.max_examples
    if max_examples is None:
        max_examples = dig(cfg_yaml, "datasets", args.task, "max_items", default=None)

    model_override = args.model or dig(cfg_yaml, "defaults", "model_id", default=None)

    outdir_root = dig(cfg_yaml, "run", "outdir_root", default="results")
    if not isinstance(outdir_root, str) or not outdir_root.strip():
        outdir_root = "results"

    cfg = EvalConfig(max_examples=max_examples, model_override=model_override)

    runner_factory = factories[args.task]
    runner = runner_factory(base_url, api_key, cfg)

    payload_any = await runner.run(max_examples=max_examples, model_override=model_override)
    summary, results, runner_report_text, returned_config = _coerce_nested_payload(payload_any)

    # Ensure task + run_id present for consistent reporting
    task_name = str(summary.get("task") or args.task)
    run_id = str(summary.get("run_id") or _utc_run_id())
    summary["task"] = task_name
    summary["run_id"] = run_id

    # Ensure base_url is reflected in summary for downstream tooling/policy
    summary.setdefault("base_url", base_url)

    # Outdir
    outdir = args.outdir or default_outdir(outdir_root, task_name, run_id)

    # Render reports (pure strings; no writes)
    bundle = render_reports_bundle(
        task=task_name,
        run_id=run_id,
        base_url=base_url,
        summary=summary,
        results=results,
        runner_report_text=runner_report_text,
    )

    # Persist artifacts via io/
    if do_save:
        # Ensure run_dir is present and matches disk
        summary["run_dir"] = outdir

        paths = write_eval_run_artifacts(
            outdir=outdir,
            summary=summary,
            results=results,
            report_txt=bundle.text,
            report_md=bundle.md,
            returned_config=returned_config if isinstance(returned_config, dict) else None,
        )

        # Optional: write eval_out/latest.json pointer (for demo-friendly wiring)
        # Defaults to ON when saving, can disable via EVAL_WRITE_LATEST=0.
        write_latest = _env_flag("EVAL_WRITE_LATEST", default=True)
        if write_latest:
            pointer_path = os.getenv("EVAL_LATEST_PATH", "").strip()
            if not pointer_path:
                pointer_path = str(default_eval_out_pointer_path())

            write_eval_latest_pointer(
                pointer_path=pointer_path,
                task=task_name,
                run_id=run_id,
                run_dir=outdir,
                summary_path=str(paths.summary_json),
                extra={
                    # non-authoritative conveniences (safe to drop later)
                    "base_url": base_url,
                    "model_override": model_override,
                    "max_examples": max_examples,
                },
            )

    # Optional debug print of first N results
    if args.debug_n and args.debug_n > 0:
        fields = None
        if args.debug_fields:
            fields = [x.strip() for x in args.debug_fields.split(",") if x.strip()]

        if not fields:
            fields = [
                "id",
                "doc_id",
                "status_code",
                "error_code",
                "error_stage",
                "latency_ms",
                "metrics",
                "predicted",
                "predicted_text",
                "predicted_preview",
                "error",
                "extra",
            ]

        n_show = min(args.debug_n, len(results))

        print("\n" + "=" * 80)
        print(f"DEBUG (first {n_show} results)")
        print("=" * 80)

        if not results:
            print("(no per-example results returned)")
        else:
            for row in results[: args.debug_n]:
                if not isinstance(row, dict):
                    continue
                compact: dict[str, Any] = {}
                for k in fields:
                    v = row.get(k)
                    if isinstance(v, str):
                        compact[k] = _shorten(v, 240)
                    else:
                        compact[k] = v
                print(json.dumps(compact, ensure_ascii=False, indent=2))
                print("-" * 80)

    if do_print:
        # Keep existing UX: print summary JSON to stdout
        print(bundle.json_summary, end="")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()