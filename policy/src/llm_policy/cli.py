# policy/src/llm_policy/cli.py
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from llm_policy.config import PolicyConfig, load_extract_thresholds
from llm_policy.io.eval_artifacts import load_eval_artifact
from llm_policy.io.models_yaml import patch_models_yaml
from llm_policy.policies.extract_enablement import decide_extract_enablement
from llm_policy.reports.writer import render_decision_json, render_decision_md, render_decision_text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-policy", description="Policy engine for gating LLM capabilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide-extract", help="Decide whether to enable /v1/extract based on eval artifacts")
    d.add_argument("--run-dir", type=str, required=True, help="Path to eval run directory (contains summary.json)")
    d.add_argument("--threshold-profile", type=str, default=None, help="Threshold profile, e.g. extract/sroie")
    d.add_argument("--thresholds-root", type=str, default=None, help="Override thresholds root directory")
    d.add_argument("--format", type=str, default="text", choices=["text", "json", "md"], help="Output format")
    d.add_argument("--out", type=str, default=None, help="Write rendered output to file (optional)")

    pm = sub.add_parser("patch-models", help="Apply a decision to models.yaml by editing capabilities")
    pm.add_argument("--models-yaml", type=str, required=True, help="Path to models.yaml")
    pm.add_argument("--model-id", type=str, required=True, help="Model id to patch")
    pm.add_argument("--enable-extract", action="store_true", help="Enable extract capability")
    pm.add_argument("--disable-extract", action="store_true", help="Disable extract capability")
    pm.add_argument("--dry-run", action="store_true", help="Do not write; just show what would change")

    dp = sub.add_parser("decide-and-patch", help="Decide enablement and patch models.yaml in one step")
    dp.add_argument("--run-dir", type=str, required=True)
    dp.add_argument("--models-yaml", type=str, required=True)
    dp.add_argument("--model-id", type=str, required=True)
    dp.add_argument("--threshold-profile", type=str, default=None)
    dp.add_argument("--thresholds-root", type=str, default=None)
    dp.add_argument("--format", type=str, default="text", choices=["text", "json", "md"])
    dp.add_argument("--dry-run", action="store_true")

    return p


def _emit(s: str, out: Optional[str]) -> None:
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s, end="")


def _render(decision, fmt: str) -> str:
    if fmt == "json":
        return render_decision_json(decision)
    if fmt == "md":
        return render_decision_md(decision)
    return render_decision_text(decision)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "decide-extract":
        artifact = load_eval_artifact(args.run_dir)

        pcfg = PolicyConfig.default()
        if args.thresholds_root:
            pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

        profile, th = load_extract_thresholds(cfg=pcfg, profile=args.threshold_profile)
        decision = decide_extract_enablement(artifact, thresholds=th, thresholds_profile=profile)

        rendered = _render(decision, args.format)
        _emit(rendered, args.out)

        # Exit codes:
        # 0 = allow, 2 = deny/unknown/contract error (fail-closed)
        return 0 if decision.ok() else 2

    if args.cmd == "patch-models":
        if args.enable_extract and args.disable_extract:
            print("Error: choose only one of --enable-extract or --disable-extract")
            return 2

        # default if neither flag is set: enable (explicitly)
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

        payload = {
            "changed": res.changed,
            "warnings": res.warnings,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.cmd == "decide-and-patch":
        artifact = load_eval_artifact(args.run_dir)

        pcfg = PolicyConfig.default()
        if args.thresholds_root:
            pcfg = PolicyConfig(thresholds_root=args.thresholds_root)

        profile, th = load_extract_thresholds(cfg=pcfg, profile=args.threshold_profile)
        decision = decide_extract_enablement(artifact, thresholds=th, thresholds_profile=profile)

        res = patch_models_yaml(
            path=args.models_yaml,
            model_id=args.model_id,
            capability="extract",
            enable=bool(decision.enable_extract),
            write=(not args.dry_run),
        )

        rendered = _render(decision, args.format)

        print(rendered, end="")
        print("\n" + "-" * 80)
        print(json.dumps({"patched": res.changed, "warnings": res.warnings}, indent=2))

        return 0 if decision.ok() else 2

    print("Unknown command")
    return 2