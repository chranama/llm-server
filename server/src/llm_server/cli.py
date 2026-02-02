# src/llm_server/cli.py
from __future__ import annotations

import asyncio
import os
from typing import Optional

import typer
import uvicorn

import llm_server.db.session as db_session
from llm_server.db.models import Role
from llm_server.tools.api_keys import CreateKeyInput, create_api_key, list_api_keys
from llm_server.tools.db_migrate import migrate_from_env

app = typer.Typer(
    name="llm",
    help="Unified CLI for llm-server (serve + operational tools).",
    no_args_is_help=True,
)

tools_app = typer.Typer(help="Operational tools (DB access, migrations, etc.)", no_args_is_help=True)
app.add_typer(tools_app, name="tools")

api_keys_app = typer.Typer(help="Manage API keys (direct DB access)", no_args_is_help=True)
tools_app.add_typer(api_keys_app, name="api-keys")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _run(coro):
    # Centralize asyncio.run so commands stay sync for Typer.
    return asyncio.run(coro)


# ---------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------
@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", envvar="HOST", help="Bind host"),
    port: int = typer.Option(8000, "--port", envvar="PORT", help="Bind port"),
    workers: int = typer.Option(1, "--workers", envvar="WORKERS", help="Uvicorn workers (prod)"),
    reload_: Optional[bool] = typer.Option(
        None,
        "--reload/--no-reload",
        help="Enable auto-reload (overrides UVICORN_RELOAD if provided).",
    ),
    proxy_headers: bool = typer.Option(True, "--proxy-headers/--no-proxy-headers", help="Respect proxy headers"),
):
    """
    Run the FastAPI service.

    Policy:
      - reload is opt-in (either --reload or UVICORN_RELOAD=1)
      - dev mode forces workers=1 (stateful in-memory LLM)
      - prod mode uses WORKERS (default 1)
    """
    env = os.getenv("ENV", "").lower()
    dev_mode = env == "dev" or os.getenv("DEV") == "1"

    if reload_ is None:
        reload_enabled = _env_flag("UVICORN_RELOAD", "0")
    else:
        reload_enabled = bool(reload_)

    if dev_mode:
        uvicorn.run(
            "llm_server.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload_enabled,
            workers=1,
            proxy_headers=proxy_headers,
        )
        return

    uvicorn.run(
        "llm_server.main:create_app",
        factory=True,
        host=host,
        port=port,
        workers=int(workers),
        proxy_headers=proxy_headers,
    )


# ---------------------------------------------------------------------
# tools: migrate-data
# ---------------------------------------------------------------------
@tools_app.command("migrate-data")
def migrate_data(
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size (env BATCH_SIZE or 500 default)"),
    no_skip_on_conflict: bool = typer.Option(
        False,
        "--no-skip-on-conflict",
        help="Fail on duplicates instead of skipping (default: skip)",
    ),
):
    """
    Copy data between DBs (SOURCE_DB_URL -> TARGET_DB_URL).
    """
    migrate_from_env(
        batch_size=batch_size,
        skip_on_conflict=(not no_skip_on_conflict),
    )


# ---------------------------------------------------------------------
# tools: api-keys
# ---------------------------------------------------------------------
@api_keys_app.command("list")
def api_keys_list(
    show_secret: bool = typer.Option(False, "--show-secret", help="Print full API key value (dangerous)"),
):
    async def _impl() -> int:
        async with db_session.get_sessionmaker()() as session:
            rows = await list_api_keys(session)

        if not rows:
            typer.echo("No API keys found.")
            return 0

        typer.echo("API Keys:")
        for key, role_name in rows:
            typer.echo("-" * 60)
            if show_secret:
                typer.echo(f"Key:            {key.key}")
            else:
                tail = key.key[-8:] if key.key else ""
                typer.echo(f"Key:            ****{tail}")
            typer.echo(f"Label:          {key.label}")
            typer.echo(f"Active:         {key.active}")
            typer.echo(f"Role:           {role_name}")
            typer.echo(f"Quota monthly:  {key.quota_monthly}")
            typer.echo(f"Quota used:     {key.quota_used}")

        return 0

    raise SystemExit(_run(_impl()))


@api_keys_app.command("create")
def api_keys_create(
    role: str = typer.Option(Role.standard.value, "--role", help="Role for the key"),
    label: str = typer.Option("dev", "--label", help="Label/name for the key"),
    quota: Optional[int] = typer.Option(None, "--quota", help="Monthly quota (omit for unlimited)"),
    unlimited: bool = typer.Option(False, "--unlimited", help="Force unlimited usage"),
):
    allowed_roles = {r.value for r in Role}
    if role not in allowed_roles:
        raise typer.BadParameter(f"role must be one of: {sorted(allowed_roles)}")

    async def _impl() -> int:
        quota_monthly = None if unlimited else quota

        async with db_session.get_sessionmaker()() as session:
            obj = await create_api_key(
                session,
                CreateKeyInput(role=role, label=label, quota_monthly=quota_monthly),
            )

        typer.echo("\n✅ API key created\n")
        typer.echo(f"Key:  {obj.key}")
        typer.echo(f"Role: {role}")
        if quota_monthly is None:
            typer.echo("Quota: UNLIMITED\n")
        else:
            typer.echo(f"Quota: {quota_monthly} / month\n")
        return 0

    raise SystemExit(_run(_impl()))


def main() -> None:
    app()