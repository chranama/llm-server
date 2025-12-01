#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "🔧 Starting CPU container dev environment (ENV=.env)..."

make dev-cpu ENV_FILE=.env

echo
echo "🚀 CPU environment is ready."
echo "📌 Test via:"
echo "    make curl API_KEY=<your-key>"
echo