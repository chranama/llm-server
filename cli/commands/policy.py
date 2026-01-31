# cli/commands/policy.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from cli.errors import CLIError
from cli.types import GlobalConfig
from cli.util.proc import ensure_bins, run


JOB_NAME_DEFAULT = "policy"
NAMESPACE_DEFAULT = "llm"

# NOTE: keep these paths relative to repo_root (cfg.repo_root)
K8S_POLICY_JOB_YAML = Path("deploy/k8s/base/policy/job.yaml")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("policy", help="Policy helpers (compose + k8s runners).")
    p.set_defaults(_handler=_handle)

    sp = p.add_subparsers(dest="policy_cmd", required=True)

    # -------------------------
    # Compose runner
    # -------------------------
    c = sp.add_parser("compose", help="Run llm-policy inside docker-compose (ephemeral).")
    csp = c.add_subparsers(dest="compose_cmd", required=True)

    _compose_add_common_policy_args(csp.add_parser("decide-extract", help="decide-extract (compose)"))
    _compose_add_patch_args(csp.add_parser("patch-models", help="patch-models (compose)"))
    _compose_add_common_policy_args(csp.add_parser("decide-and-patch", help="decide-and-patch (compose)"), patch=True)

    # -------------------------
    # K8s runner (Job-based)
    # -------------------------
    k = sp.add_parser("k8s", help="Run llm-policy as a Kubernetes Job (apply/wait/logs).")
    k.add_argument("--namespace", default=NAMESPACE_DEFAULT, help="Kubernetes namespace (default: llm)")
    k.add_argument("--job-name", default=JOB_NAME_DEFAULT, help="Policy job name (default: policy)")

    ksp = k.add_subparsers(dest="k8s_cmd", required=True)

    r = ksp.add_parser("run", help="(Re)apply policy job, wait, print logs, return exit code.")
    r.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Wait timeout in seconds for job completion (default: 600).",
    )
    r.add_argument(
        "--no-logs",
        action="store_true",
        help="Do not print job logs after completion.",
    )

    ksp.add_parser("delete", help="Delete the policy job if it exists.")
    ksp.add_parser("logs", help="Show policy job logs (best effort).")

    # Optional: quick check
    ksp.add_parser("status", help="Show job status (best effort).")


def _compose_add_common_policy_args(p: argparse.ArgumentParser, *, patch: bool = False) -> None:
    p.add_argument("--run-dir", required=True, help="Eval run directory (contains summary.json).")
    p.add_argument("--format", default="text", choices=["text", "json", "md"], help="Output format.")
    p.add_argument("--threshold-profile", default=None, help="Threshold profile (e.g. extract/sroie).")
    p.add_argument("--thresholds-root", default=None, help="Override thresholds root directory (optional).")

    if patch:
        p.add_argument("--models-yaml", required=True, help="Path to models.yaml (inside container).")
        p.add_argument("--model-id", required=True, help="Model id to patch.")
        p.add_argument("--dry-run", action="store_true", help="Do not write; just show changes.")


def _compose_add_patch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models-yaml", required=True, help="Path to models.yaml (inside container).")
    p.add_argument("--model-id", required=True, help="Model id to patch.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--enable-extract", action="store_true", help="Enable extract capability.")
    g.add_argument("--disable-extract", action="store_true", help="Disable extract capability.")
    p.add_argument("--dry-run", action="store_true", help="Do not write; just show changes.")


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker", "bash")

    if args.policy_cmd == "compose":
        return _handle_compose(cfg, args)

    if args.policy_cmd == "k8s":
        # only require kubectl for k8s mode
        ensure_bins("kubectl")
        return _handle_k8s(cfg, args)

    raise CLIError(f"Unknown policy command: {args.policy_cmd}", code=2)


# -------------------------
# Compose implementation
# -------------------------
def _compose_base(cfg: GlobalConfig, args: argparse.Namespace) -> List[str]:
    # Use docker compose with explicit file for determinism
    return [
        "docker",
        "compose",
        "-f",
        str(cfg.compose_yml),
        "--project-name",
        cfg.project_name,
        "--profile",
        "policy",
    ]


def _handle_compose(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    cmd = args.compose_cmd

    base = _compose_base(cfg, args)

    # Build llm-policy args
    policy_argv: List[str] = []
    if cmd == "decide-extract":
        policy_argv = [
            "decide-extract",
            "--run-dir",
            args.run_dir,
            "--format",
            args.format,
        ]
        if args.threshold_profile:
            policy_argv += ["--threshold-profile", args.threshold_profile]
        if args.thresholds_root:
            policy_argv += ["--thresholds-root", args.thresholds_root]

    elif cmd == "patch-models":
        enable = True if args.enable_extract else False
        policy_argv = [
            "patch-models",
            "--models-yaml",
            args.models_yaml,
            "--model-id",
            args.model_id,
            "--enable-extract" if enable else "--disable-extract",
        ]
        if args.dry_run:
            policy_argv.append("--dry-run")

    elif cmd == "decide-and-patch":
        policy_argv = [
            "decide-and-patch",
            "--run-dir",
            args.run_dir,
            "--models-yaml",
            args.models_yaml,
            "--model-id",
            args.model_id,
            "--format",
            args.format,
        ]
        if args.threshold_profile:
            policy_argv += ["--threshold-profile", args.threshold_profile]
        if args.thresholds_root:
            policy_argv += ["--thresholds-root", args.thresholds_root]
        if args.dry_run:
            policy_argv.append("--dry-run")

    else:
        raise CLIError(f"Unknown policy compose command: {cmd}", code=2)

    # Run as ephemeral service container
    full = base + ["run", "--rm", "policy"] + policy_argv
    r = run(full, cwd=str(cfg.repo_root), verbose=args.verbose, check=False)
    return int(r.code)


# -------------------------
# K8s implementation
# -------------------------
def _job_yaml_path(cfg: GlobalConfig) -> Path:
    p = cfg.repo_root / K8S_POLICY_JOB_YAML
    if not p.exists():
        raise CLIError(f"Missing k8s policy job yaml: {p}", code=2)
    return p


def _kubectl(ns: str, *argv: str) -> List[str]:
    return ["kubectl", "-n", ns, *argv]


def _handle_k8s(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ns = args.namespace
    job_name = args.job_name
    cmd = args.k8s_cmd

    job_yaml = _job_yaml_path(cfg)

    if cmd == "delete":
        # ignore-not-found: best effort
        run(_kubectl(ns, "delete", "job", job_name, "--ignore-not-found=true"), verbose=args.verbose, check=False)
        return 0

    if cmd == "logs":
        # best effort logs (job might not exist)
        r = run(_kubectl(ns, "logs", f"job/{job_name}", "--all-containers=true"), verbose=args.verbose, check=False)
        return int(r.code)

    if cmd == "status":
        r = run(_kubectl(ns, "get", "job", job_name, "-o", "wide"), verbose=args.verbose, check=False)
        return int(r.code)

    if cmd == "run":
        # Ensure a fresh run: delete then apply
        run(_kubectl(ns, "delete", "job", job_name, "--ignore-not-found=true"), verbose=args.verbose, check=False)

        # Apply job spec (must have metadata.name matching --job-name)
        # If your YAML uses a different name, either change the YAML or pass --job-name accordingly.
        run(_kubectl(ns, "apply", "-f", str(job_yaml)), verbose=args.verbose)

        # Wait for completion OR failure
        timeout = int(getattr(args, "timeout_seconds", 600))
        wait_ok = run(
            _kubectl(ns, "wait", "--for=condition=complete", f"job/{job_name}", f"--timeout={timeout}s"),
            verbose=args.verbose,
            check=False,
        )

        # If completion wait failed, also check failed condition (for clearer exit code).
        if wait_ok.code != 0:
            _ = run(
                _kubectl(ns, "wait", "--for=condition=failed", f"job/{job_name}", f"--timeout=1s"),
                verbose=args.verbose,
                check=False,
            )

        if not getattr(args, "no_logs", False):
            run(_kubectl(ns, "logs", f"job/{job_name}", "--all-containers=true"), verbose=args.verbose, check=False)

        # Exit code: 0 if job completed, else propagate nonzero
        return 0 if wait_ok.code == 0 else int(wait_ok.code)

    raise CLIError(f"Unknown policy k8s command: {cmd}", code=2)