# policy/src/llm_policy/cli.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from llm_policy.config import PolicyConfig, load_extract_thresholds
from llm_policy.io.decision_artifacts import default_policy_out_path, write_decision_artifact
from llm_policy.io.eval_artifacts import load_eval_artifact
from llm_policy.io.models_yaml import patch_models_yaml
from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.reports.writer import render_decision_md, render_decision_text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-policy", description="Policy engine for gating LLM capabilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide-extract", help="Decide whether to enable /v1/extract based on eval artifacts")
    d.add_argument(
        "--run-dir",
        type=str,
        default=os.getenv("POLICY_RUN_DIR", "latest"),
        help=(
            "Path to eval run directory (contains summary.json), or 'latest' to follow eval_out/latest.json "
            "(default: $POLICY_RUN_DIR or 'latest')."
        ),
    )
    d.add_argument("--threshold-profile", type=str, default=None, help="Threshold profile, e.g. extract/sroie")
    d.add_argument("--thresholds-root", type=str, default=None, help="Override thresholds root directory")

    # Human-only rendering (reports/)
    d.add_argument("--report", type=str, default="text", choices=["text", "md"], help="Human report format")
    d.add_argument("--report-out", type=str, default=None, help="Write human report to file (optional)")

    # Runtime ingestion artifact
    d.add_argument(
        "--artifact-out",
        type=str,
        default=None,
        help=(
            "Write runtime decision artifact JSON to this path. "
            "If omitted, defaults to POLICY_OUT_PATH or policy_out/latest.json."
        ),
    )
    d.add_argument(
        "--no-write-artifact",
        action="store_true",
        help="Do not write the runtime decision artifact (debug only).",
    )

    pm = sub.add_parser("patch-models", help="Apply a decision to models.yaml by editing capabilities")
    pm.add_argument("--models-yaml", type=str, required=True, help="Path to models.yaml")
    pm.add_argument("--model-id", type=str, required=True, help="Model id to patch")
    pm.add_argument("--enable-extract", action="store_true", help="Enable extract capability")
    pm.add_argument("--disable-extract", action="store_true", help="Disable extract capability")
    pm.add_argument("--dry-run", action="store_true", help="Do not write; just show what would change")

    dp = sub.add_parser("decide-and-patch", help="Decide enablement and patch models.yaml in one step")
    dp.add_argument(
        "--run-dir",
        type=str,
        default=os.getenv("POLICY_RUN_DIR", "latest"),
        help=(
            "Path to eval run directory (contains summary.json), or 'latest' to follow eval_out/latest.json "
            "(default: $POLICY_RUN_DIR or 'latest')."
        ),
    )
    dp.add_argument("--models-yaml", type=str, required=True)
    dp.add_argument("--model-id", type=str, required=True)
    dp.add_argument("--threshold-profile", type=str, default=None)
    dp.add_argument("--thresholds-root", type=str, default=None)
    dp.add_argument("--dry-run", action="store_true")

    # Human-only rendering (reports/)
    dp.add_argument("--report", type=str, default="text", choices=["text", "md"], help="Human report format")
    dp.add_argument("--report-out", type=str, default=None, help="Write human report to file (optional)")

    # Runtime ingestion artifact
    dp.add_argument(
        "--artifact-out",
        type=str,
        default=None,
        help=(
            "Write runtime decision artifact JSON to this path. "
            "If omitted, defaults to POLICY_OUT_PATH or policy_out/latest.json."
        ),
    )
    dp.add_argument(
        "--no-write-artifact",
        action="store_true",
        help="Do not write the runtime decision artifact (debug only).",
    )

    return p


def _emit(s: str, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s, end="")


def _render_human(decision, fmt: str) -> str:
    if fmt == "md":
        return render_decision_md(decision)
    return render_decision_text(decision)


def _artifact_path_or_default(raw: Optional[str]) -> str:
    if raw and raw.strip():
        return raw.strip()

    # Explicit env override for stability across cwd
    env = os.getenv("POLICY_OUT_PATH", "").strip()
    if env:
        return env

    # fall back to default helper
    return str(default_policy_out_path())


def _validate_outfile_path(p: str) -> None:
    """
    Guardrails:
      - require a filename (not a directory)
      - ensure parent is creatable (but don't create it here)
    """
    pp = Path(p)
    if pp.name in ("", ".", ".."):
        raise ValueError(f"artifact path must be a file path, got: {p!r}")


def _build_decision(args):
    # load_eval_artifact now supports run_dir="latest" (via eval_out/latest.json pointer)
    artifact = load_eval_artifact(getattr(args, "run_dir", "latest") or "latest")

    pcfg = PolicyConfig.default()
    if getattr(args, "thresholds_root", None):
        pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

    profile, th = load_extract_thresholds(cfg=pcfg, profile=getattr(args, "threshold_profile", None))
    return decide_extract_enablement(artifact, thresholds=th, thresholds_profile=profile)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "decide-extract":
        decision = _build_decision(args)

        # Runtime artifact (server ingestion) — canonical + atomic
        if not args.no_write_artifact:
            out_path = _artifact_path_or_default(args.artifact_out)
            _validate_outfile_path(out_path)
            write_decision_artifact(decision, out_path)

        # Human report (optional)
        rendered = _render_human(decision, args.report)
        _emit(rendered, args.report_out)

        return 0 if decision.ok() else 2

    if args.cmd == "patch-models":
        if args.enable_extract and args.disable_extract:
            print("Error: choose only one of --enable-extract or --disable-extract")
            return 2

        enable = True
        if args.disable_extract:
            enable = False
        if args.enable_extract:
            enable = True

        res = patch_models_yaml(
            path=args.models_yaml,
            model_id=args.model_id,
            capability="extract",
            enable=enable,
            write=(not args.dry_run),
        )

        print(json.dumps({"changed": res.changed, "warnings": res.warnings}, indent=2))
        return 0

    if args.cmd == "decide-and-patch":
        decision = _build_decision(args)

        res = patch_models_yaml(
            path=args.models_yaml,
            model_id=args.model_id,
            capability="extract",
            enable=bool(decision.enable_extract),
            write=(not args.dry_run),
        )

        # Runtime artifact (server ingestion) — canonical + atomic
        if not args.no_write_artifact:
            out_path = _artifact_path_or_default(args.artifact_out)
            _validate_outfile_path(out_path)
            write_decision_artifact(decision, out_path)

        # Human report (optional)
        rendered = _render_human(decision, args.report)
        _emit(rendered, args.report_out)

        # Always print patch summary JSON to stdout
        print("\n" + "-" * 80)
        print(json.dumps({"patched": res.changed, "warnings": res.warnings}, indent=2))

        return 0 if decision.ok() else 2

    print("Unknown command")
    return 2