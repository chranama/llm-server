# cli/commands/compose.py
from __future__ import annotations

import argparse
from typing import Sequence

from cli.errors import CLIError
from cli.types import GlobalConfig  # type: ignore[attr-defined]
from cli.util.proc import ensure_bins, run

# Compose verbs we recognize to auto-split profiles vs args when user omits `--`.
# (Keep this fairly broad; better to accept than to reject.)
_COMPOSE_VERBS = {
    "up",
    "down",
    "ps",
    "logs",
    "config",
    "build",
    "pull",
    "push",
    "restart",
    "stop",
    "start",
    "rm",
    "exec",
    "run",
    "kill",
    "pause",
    "unpause",
    "top",
    "events",
    "images",
    "ls",
    "port",
    "cp",
    "create",
}


def _compose_base(cfg: GlobalConfig) -> list[str]:
    return ["docker", "compose", "--env-file", str(cfg.env_file), "-f", str(cfg.compose_yml)]


def _compose_env(cfg: GlobalConfig) -> dict[str, str]:
    # Compose loads --env-file for interpolation; this pins project name deterministically.
    return {"COMPOSE_PROJECT_NAME": cfg.project_name}


def _add_profiles(cmd: list[str], profiles: Sequence[str]) -> list[str]:
    for p in profiles:
        cmd += ["--profile", p]
    return cmd


def _split_profiles_and_args(tokens: list[str]) -> tuple[list[str], list[str]]:
    """
    Accept either:
      - dc <profiles...> -- <compose args...>
      - dc <profiles...> <compose-verb> <compose args...>   (no --)

    We split on:
      - first literal "--", OR
      - first token that matches a known compose verb
    """
    if not tokens:
        return [], []

    # If explicit -- is present, split there.
    if "--" in tokens:
        i = tokens.index("--")
        profiles = tokens[:i]
        extra = tokens[i + 1 :]
        return profiles, extra

    # Otherwise split at first compose verb.
    for i, t in enumerate(tokens):
        if t in _COMPOSE_VERBS:
            profiles = tokens[:i]
            extra = tokens[i:]
            return profiles, extra

    # No separator and no verb: treat as profiles-only (args missing)
    return tokens, []


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("compose", help="Direct docker compose control (replacement for just dc).")
    p.set_defaults(_handler=_handle)

    sp = p.add_subparsers(dest="compose_cmd", required=True)

    # compose dc <profiles...> [--] <args...>
    dc = sp.add_parser(
        "dc",
        help="Compose with profiles. Examples: "
        "`llmctl compose dc infra api -- up -d --build` OR `llmctl compose dc infra api up -d --build`",
    )
    # We take everything and split ourselves to support both styles reliably.
    dc.add_argument(
        "tokens",
        nargs=argparse.REMAINDER,
        help="Profiles + compose args. Use `--` to separate, or omit it and include a compose verb (e.g. up/ps/logs).",
    )

    cfgp = sp.add_parser("config", help="docker compose config (validates compose).")
    cfgp.add_argument("--profiles", nargs="*", default=[], help="Optional profiles to include when rendering config")

    psp = sp.add_parser("ps", help="docker compose ps")
    psp.add_argument("--profiles", nargs="*", default=[], help="Optional profiles")
    psp.add_argument("args", nargs=argparse.REMAINDER)

    lg = sp.add_parser("logs", help="docker compose logs -f --tail=200 (default)")
    lg.add_argument("--profiles", nargs="*", default=[], help="Optional profiles")
    lg.add_argument("--follow", action="store_true", help="Follow logs (default)")
    lg.add_argument("--tail", type=int, default=200, help="Tail N lines (default 200)")

    dn = sp.add_parser("down", help="docker compose down --remove-orphans")
    dn.add_argument("--profiles", nargs="*", default=[], help="Optional profiles")
    dn.add_argument("--volumes", action="store_true", help="Also remove volumes (-v)")
    dn.add_argument("--remove-orphans", action="store_true", help="Remove orphans (default true)")
    dn.set_defaults(remove_orphans=True)

    up = sp.add_parser("up", help="docker compose up")
    up.add_argument("--profiles", nargs="*", default=[], help="Optional profiles")
    up.add_argument("-d", "--detach", action="store_true", help="Run detached")
    up.add_argument("--build", action="store_true", help="Build images")
    up.add_argument("--remove-orphans", action="store_true", help="Remove orphans")
    up.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed to compose up")

    rm = sp.add_parser("rm-orphans", help="Shortcut: compose up -d --remove-orphans for a profile set")
    rm.add_argument("--profiles", nargs="*", default=[], help="Profiles")

    infra = sp.add_parser("infra-up", help="Start postgres+redis (profile infra).")
    infra.set_defaults(_shortcut="infra-up")
    infra2 = sp.add_parser("infra-ps", help="Show infra status.")
    infra2.set_defaults(_shortcut="infra-ps")
    infra3 = sp.add_parser("infra-down", help="Stop infra.")
    infra3.add_argument("--volumes", action="store_true")
    infra3.set_defaults(_shortcut="infra-down")


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("docker")
    env = _compose_env(cfg)

    # Shortcuts
    if getattr(args, "_shortcut", None) == "infra-up":
        cmd = _add_profiles(_compose_base(cfg), ["infra"]) + ["up", "-d", "--remove-orphans"]
        run(cmd, env=env, verbose=args.verbose)
        print("✅ infra up (postgres/redis).")
        return 0

    if getattr(args, "_shortcut", None) == "infra-ps":
        cmd = _add_profiles(_compose_base(cfg), ["infra"]) + ["ps"]
        run(cmd, env=env, verbose=args.verbose)
        return 0

    if getattr(args, "_shortcut", None) == "infra-down":
        cmd = _add_profiles(_compose_base(cfg), ["infra"]) + ["down", "--remove-orphans"]
        if getattr(args, "volumes", False):
            cmd.append("-v")
        run(cmd, env=env, verbose=args.verbose)
        return 0

    c = args.compose_cmd

    if c == "dc":
        tokens = list(getattr(args, "tokens", []) or [])
        # argparse.REMAINDER includes a leading "--" sometimes if user wrote it; our splitter handles it.
        profiles, extra = _split_profiles_and_args(tokens)

        # Normalize: allow no profiles; require compose args.
        if not extra:
            raise CLIError(
                "compose dc requires compose args. Examples:\n"
                "  llmctl compose dc infra api -- up -d --build\n"
                "  llmctl compose dc infra api up -d --build\n"
                "  llmctl compose dc infra -- ps"
            )

        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, profiles)
        cmd += extra
        run(cmd, env=env, verbose=args.verbose)
        return 0

    if c == "config":
        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, list(args.profiles or []))
        cmd += ["config"]
        run(cmd, env=env, verbose=args.verbose)
        print("✅ compose config OK")
        return 0

    if c == "ps":
        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, list(args.profiles or []))
        cmd += ["ps"] + list(args.args or [])
        run(cmd, env=env, verbose=args.verbose)
        return 0

    if c == "logs":
        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, list(args.profiles or []))
        follow = "-f" if args.follow else "-f"  # default follow on (matches your justfile)
        cmd += ["logs", follow, f"--tail={args.tail}"]
        cmd = [x for x in cmd if x]
        run(cmd, env=env, verbose=args.verbose)
        return 0

    if c == "down":
        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, list(args.profiles or []))
        cmd += ["down"]
        if args.remove_orphans:
            cmd.append("--remove-orphans")
        if args.volumes:
            cmd.append("-v")
        run(cmd, env=env, verbose=args.verbose)
        return 0

    if c == "up":
        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, list(args.profiles or []))
        cmd += ["up"]
        if args.detach:
            cmd.append("-d")
        if args.build:
            cmd.append("--build")
        if args.remove_orphans:
            cmd.append("--remove-orphans")
        cmd += list(args.args or [])
        run(cmd, env=env, verbose=args.verbose)
        return 0

    if c == "rm-orphans":
        cmd = _compose_base(cfg)
        cmd = _add_profiles(cmd, list(args.profiles or []))
        cmd += ["up", "-d", "--remove-orphans"]
        run(cmd, env=env, verbose=args.verbose)
        return 0

    raise CLIError(f"Unknown compose command: {c}", code=2)