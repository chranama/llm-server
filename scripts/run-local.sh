#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "🔧 Starting LOCAL dev environment (MPS LLM on host, infra in Docker)..."
echo "   Using ENV_FILE=.env.local"
echo

# Bring up Docker infra (postgres/redis/nginx/prometheus/grafana) and run migrations on host
ENV_FILE=.env.local make dev-local

echo
echo "🚀 Local infra is ready (Postgres, Redis, Prometheus, Grafana, Nginx)."
echo
echo "📌 Next steps:"
echo "  1) Start the API (FastAPI + Llama 3.1 8B on MPS) in another terminal:"
echo "       ENV_FILE=.env.local make api-local"
echo
echo "  2) Seed an API key (once, if you haven’t already):"
echo "       make seed-key API_KEY=\$(openssl rand -hex 24)"
echo
echo "  3) Test via Nginx once API is running:"
echo "       make curl API_KEY=<your-key>"
echo
echo "Nginx API URL: http://localhost:8080/api/v1/generate"
echo