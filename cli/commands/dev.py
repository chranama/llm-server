# cli/commands/dev.py
from __future__ import annotations

import argparse
import os

from cli.errors import CLIError
from cli.util.proc import ensure_bins, run, run_bash
from cli.types import GlobalConfig  # type: ignore[attr-defined]


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("dev", help="Golden paths: dev-cpu/dev-gpu, smoke tests, doctor.")
    p.set_defaults(_handler=_handle)

    sp = p.add_subparsers(dest="dev_cmd", required=True)

    sp.add_parser("dev-cpu", help="infra+api (cpu) + migrations")
    sp.add_parser("dev-gpu", help="infra+api-gpu + migrations")
    sp.add_parser("dev-cpu-generate-only", help="infra+api (generate-only) + migrations")
    sp.add_parser("dev-gpu-generate-only", help="infra+api-gpu (generate-only) + migrations")

    sp.add_parser("doctor", help="Run tools/compose_doctor.sh")

    sm = sp.add_parser("smoke-cpu", help="dev-cpu + doctor + /v1/generate probe (if API_KEY set)")
    sm2 = sp.add_parser("smoke-cpu-generate-only", help="dev-cpu-generate-only + doctor")
    sm3 = sp.add_parser("smoke-gpu", help="dev-gpu + doctor")
    sm4 = sp.add_parser("smoke-gpu-generate-only", help="dev-gpu-generate-only + doctor")


def _compose(cfg: GlobalConfig, profiles: list[str], args: list[str], verbose: bool) -> None:
    cmd = ["docker", "compose", "--env-file", str(cfg.env_file), "-f", str(cfg.compose_yml)]
    for p in profiles:
        cmd += ["--profile", p]
    cmd += args
    env = {"COMPOSE_PROJECT_NAME": cfg.project_name}
    run(cmd, env=env, verbose=verbose)


def _migrate(cfg: GlobalConfig, verbose: bool) -> None:
    # Try api then api_gpu.
    env = {"COMPOSE_PROJECT_NAME": cfg.project_name}
    base = ["docker", "compose", "--env-file", str(cfg.env_file), "-f", str(cfg.compose_yml)]
    # Determine service existence by listing ps services
    # Keep it simple: just try exec in api then api_gpu.
    for svc in ["api", "api_gpu"]:
        try:
            run(base + ["exec", "-T", svc, "python", "-m", "alembic", "upgrade", "head"], env=env, verbose=verbose)
            print("✅ migrations applied (docker)")
            return
        except Exception:
            continue
    raise CLIError("No running api/api_gpu container found. Start api first.", code=2)


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker", "bash")

    c = args.dev_cmd

    if c == "dev-cpu":
        _compose(cfg, ["infra", "api"], ["up", "-d", "--build", "--remove-orphans"], args.verbose)
        print(f"✅ api up (docker) @ http://localhost:{cfg.api_port}")
        _migrate(cfg, args.verbose)
        return 0

    if c == "dev-gpu":
        _compose(cfg, ["infra", "api-gpu"], ["up", "-d", "--build", "--remove-orphans"], args.verbose)
        print(f"✅ api_gpu up (docker) @ http://localhost:{cfg.api_port}")
        _migrate(cfg, args.verbose)
        return 0

    if c == "dev-cpu-generate-only":
        env = {"MODELS_YAML": str(cfg.models_generate_only)}
        _compose(cfg, ["infra", "api"], ["up", "-d", "--build", "--remove-orphans"], args.verbose)
        # Note: MODELS_YAML is consumed at container runtime; set it via env-file or export when running compose.
        # Here, we re-run compose with env override to ensure it reaches docker compose.
        run_bash(
            f'export MODELS_YAML="{cfg.models_generate_only}"; '
            f'COMPOSE_PROJECT_NAME="{cfg.project_name}" docker compose --env-file "{cfg.env_file}" -f "{cfg.compose_yml}" '
            f'--profile infra --profile api up -d --build --remove-orphans',
            verbose=args.verbose,
        )
        _migrate(cfg, args.verbose)
        return 0

    if c == "dev-gpu-generate-only":
        run_bash(
            f'export MODELS_YAML="{cfg.models_generate_only}"; '
            f'COMPOSE_PROJECT_NAME="{cfg.project_name}" docker compose --env-file "{cfg.env_file}" -f "{cfg.compose_yml}" '
            f'--profile infra --profile api-gpu up -d --build --remove-orphans',
            verbose=args.verbose,
        )
        _migrate(cfg, args.verbose)
        return 0

    if c == "doctor":
        if not cfg.compose_doctor.exists():
            raise CLIError(f"compose doctor not found: {cfg.compose_doctor}", code=2)
        if not os.access(cfg.compose_doctor, os.X_OK):
            raise CLIError(f"compose doctor not executable: chmod +x {cfg.compose_doctor}", code=2)
        env = {
            "API_PORT": cfg.api_port,
            "UI_PORT": cfg.ui_port,
            "PGADMIN_PORT": cfg.pgadmin_port,
            "PROM_PORT": cfg.prom_port,
            "GRAFANA_PORT": cfg.grafana_port,
            "PROM_HOST_PORT": cfg.prom_host_port,
        }
        run(["bash", str(cfg.compose_doctor)], env=env, verbose=args.verbose)
        return 0

    if c == "smoke-cpu":
        # dev-cpu + doctor + probe
        _handle(cfg, argparse.Namespace(dev_cmd="dev-cpu", verbose=args.verbose))
        _handle(cfg, argparse.Namespace(dev_cmd="doctor", verbose=args.verbose))
        api_key = os.getenv("API_KEY", "").strip()
        if not api_key:
            print("ℹ️  API_KEY not set; skipping /v1/generate probe.")
            return 0
        run(
            [
                "bash",
                "-lc",
                f'curl -fsS -X POST "http://localhost:{cfg.api_port}/v1/generate" '
                f'-H "Content-Type: application/json" -H "X-API-Key: {api_key}" '
                f'--data \'{{"prompt":"smoke test","max_new_tokens":16,"temperature":0.2}}\' >/dev/null && echo "✅ /v1/generate probe OK"',
            ],
            verbose=args.verbose,
        )
        return 0

    if c == "smoke-cpu-generate-only":
        _handle(cfg, argparse.Namespace(dev_cmd="dev-cpu-generate-only", verbose=args.verbose))
        _handle(cfg, argparse.Namespace(dev_cmd="doctor", verbose=args.verbose))
        return 0

    if c == "smoke-gpu":
        _handle(cfg, argparse.Namespace(dev_cmd="dev-gpu", verbose=args.verbose))
        _handle(cfg, argparse.Namespace(dev_cmd="doctor", verbose=args.verbose))
        return 0

    if c == "smoke-gpu-generate-only":
        _handle(cfg, argparse.Namespace(dev_cmd="dev-gpu-generate-only", verbose=args.verbose))
        _handle(cfg, argparse.Namespace(dev_cmd="doctor", verbose=args.verbose))
        return 0

    raise CLIError(f"Unknown dev command: {c}", code=2)