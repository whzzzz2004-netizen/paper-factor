#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

FORCE="${1:-}"

# Prerequisite check
for cmd in python3 pip docker git; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: $cmd not found. Please install it first."; exit 1; }
done
docker info >/dev/null 2>&1 || { echo "Error: Docker daemon not running. Start Docker Desktop first."; exit 1; }

echo "==> Installing Python package ..."
pip install -e . 2>&1 | tail -1

echo "==> Setting up .env ..."
[[ -f .env && "$FORCE" != "--force" ]] || cp .env.example .env

echo "==> Building Docker image ..."
IMAGE="local_factor_exec:latest"
if [[ "$FORCE" == "--force" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" rdagent/components/coder/factor_coder/docker 2>&1 | tail -1
fi

echo "==> Initializing workspace ..."
python -m paper_factor_cli.main init ${FORCE:-} 2>&1 | tail -3

echo ""
echo "Done. Edit .env with your API keys, then:"
echo "  start --report-file papers/inbox/your_paper.pdf"
