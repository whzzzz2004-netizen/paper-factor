#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SKIP_DOCKER=false
FORCE=false

usage() {
  cat <<EOF
Usage: bash scripts/setup.sh [OPTIONS]

One-command setup - everything you need to run paper-factor.

Options:
  --skip-docker   Skip Docker image build
  --force         Rebuild Docker image / re-download data
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-docker) SKIP_DOCKER=true; shift ;;
    --force)       FORCE=true; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage >&2; exit 2 ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok() { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }

echo ""
echo "=============================================="
echo "  paper-factor setup"
echo "=============================================="

# ---- Step 1: Check prerequisites ----
echo ""
echo "[1/4] Checking prerequisites ..."

ERRS=0
for cmd in python3 pip docker git; do
  if command -v "$cmd" >/dev/null 2>&1; then
    ok "$cmd"
  else
    fail "$cmd - please install it first"
    ERRS=$((ERRS+1))
  fi
done

if ! docker info >/dev/null 2>&1; then
  fail "Docker daemon is not running. Start Docker Desktop first."
  ERRS=$((ERRS+1))
else
  ok "docker daemon"
fi

if [[ $ERRS -gt 0 ]]; then
  echo ""
  echo "Fix the issues above and re-run setup.sh."
  exit 1
fi

# ---- Step 2: Install Python package ----
echo ""
echo "[2/4] Installing Python package ..."
cd "$PROJECT_ROOT"
pip install -e . 2>&1 | tail -1
ok "paper-factor installed"

# ---- Step 3: Set up .env ----
echo ""
echo "[3/4] Creating .env ..."
if [[ -f "$PROJECT_ROOT/.env" ]] && ! $FORCE; then
  ok ".env already exists"
else
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  ok ".env created from .env.example"
fi

echo ""
echo "  Edit .env and fill in:"
echo "    CHAT_MODEL=       (model name, e.g. deepseek-chat)"
echo "    OPENAI_API_KEY=   (your API key)"
echo "    OPENAI_API_BASE=  (API endpoint URL)"
echo "    TUSHARE_TOKEN=    (free token from https://tushare.pro)"

# ---- Step 4: Build Docker image ----
echo ""
echo "[4/4] Building Docker image (takes ~3 minutes) ..."

if $SKIP_DOCKER; then
  echo "  Skipped (--skip-docker)."
else
  IMAGE="local_factor_exec:latest"
  if docker image inspect "$IMAGE" >/dev/null 2>&1 && ! $FORCE; then
    ok "Docker image already exists"
  else
    echo "  Building from Dockerfile ..."
    docker build --build-arg USE_CHINA_MIRROR=true -t "$IMAGE" "$PROJECT_ROOT/rdagent/components/coder/factor_coder/docker"
    ok "Docker image built"
  fi
fi

# ---- Step 5: Initialize workspace data ----
echo ""
echo "  Initializing workspace and market data ..."

FORCE_FLAG=""
if $FORCE; then FORCE_FLAG="--force"; fi
python -m paper_factor_cli.main init $FORCE_FLAG 2>&1 | tail -3

# ---- Done ----
echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "  Next:"
echo "    drop a PDF into: papers/inbox/"
echo "    then run:        start --report-file papers/inbox/your_paper.pdf"
echo ""
