# cli/types.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlobalConfig:
    repo_root: Path
    env_file: Path
    project_name: str
    compose_yml: Path
    backend_dir: Path
    tools_dir: Path
    compose_doctor: Path

    models_full: Path
    models_generate_only: Path

    api_port: str
    ui_port: str
    pgadmin_port: str
    prom_port: str
    grafana_port: str
    prom_host_port: str

    pg_user: str
    pg_db: str