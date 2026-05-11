#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECK_HEALTH=false
SKIP_DOCKER=false
SKIP_DATA=false
FORCE=false
NONINTERACTIVE=false

usage() {
  cat <<EOF
Usage: bash scripts/setup.sh [OPTIONS]

One-command setup for paper-factor. Walks through all installation steps:
  1. Check prerequisites (python3, pip, docker, git)
  2. Install Python package (pip install -e .)
  3. Set up .env from .env.example
  4. Install Docker execution image
  5. Initialize workspace and market data

Options:
  --skip-docker      Skip Docker image installation
  --skip-data        Skip market data initialization
  --check            Run health check after setup
  --force            Force re-initialization of existing resources
  --non-interactive  Skip interactive prompts (use existing env vars)
  -h, --help         Show this help
EOF
}

section() {
  echo ""
  echo "=== $1 ==="
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-docker)      SKIP_DOCKER=true; shift ;;
    --skip-data)        SKIP_DATA=true; shift ;;
    --check)            CHECK_HEALTH=true; shift ;;
    --force)            FORCE=true; shift ;;
    --non-interactive)  NONINTERACTIVE=true; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage >&2; exit 2 ;;
  esac
done

echo "=============================================="
echo "  paper-factor setup"
echo "=============================================="

# ---- Step 1: Check prerequisites ----
section "Step 1/5: Checking prerequisites"

MISSING=""
for cmd in python3 pip docker git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "  [MISSING] $cmd"
    MISSING="$MISSING $cmd"
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "  [ERROR]   Docker daemon is not running"
  MISSING="$MISSING docker-daemon"
fi

if [[ -n "$MISSING" ]]; then
  echo ""
  echo "Missing prerequisites:$MISSING"
  echo "Please install/start them and re-run setup.sh."
  exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  python  $PY_VER"
echo "  pip     $(pip --version | awk '{print $2}')"
echo "  docker  $(docker --version | awk '{print $3}' | tr -d ',')"
echo "  git     $(git --version | awk '{print $3}')"
echo "  All prerequisites met."

# ---- Step 2: Install Python package ----
section "Step 2/5: Installing Python package"

cd "$PROJECT_ROOT"
pip install -e . 2>&1 | tail -1
echo "  paper-factor package installed."

# ---- Step 3: Set up .env ----
section "Step 3/5: Setting up configuration"

if [[ -f "$PROJECT_ROOT/.env" ]] && ! $FORCE; then
  echo "  .env already exists (use --force to overwrite)."
else
  if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "  Created .env from .env.example"
  else
    echo "  WARNING: .env.example not found, skipping."
  fi
fi

if ! $NONINTERACTIVE; then
  echo ""
  echo "  Configure your .env file before running the pipeline:"
  echo "    $PROJECT_ROOT/.env"
  echo ""
  echo "  Required settings:"
  echo "    OPENAI_API_KEY   - your LLM API key"
  echo "    CHAT_MODEL       - model name (e.g., gpt-4o, deepseek-chat)"
  echo "    OPENAI_API_BASE  - API endpoint URL (if using a proxy or alternate provider)"
  echo ""
  echo "  Optional for market data:"
  echo "    TUSHARE_TOKEN    - get one at https://tushare.pro"
  echo ""
fi

# ---- Step 4: Install Docker image ----
section "Step 4/5: Installing Docker image"

if $SKIP_DOCKER; then
  echo "  Skipped (--skip-docker)."
else
  bash "$SCRIPT_DIR/install_factor_docker_image.sh"
fi

# ---- Step 5: Initialize workspace data ----
section "Step 5/5: Initializing workspace data"

if $SKIP_DATA; then
  echo "  Skipped (--skip-data)."
else
  FORCE_FLAG=""
  if $FORCE; then
    FORCE_FLAG="--force"
  fi
  python -m paper_factor_cli.main init $FORCE_FLAG
fi

# ---- Health check ----
if $CHECK_HEALTH; then
  section "Extra: Health check"
  python -m rdagent.app.utils.health_check 2>&1 || true
fi

# ---- Done ----
echo ""
echo "=============================================="
echo "  Setup complete"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "    1. Edit .env with your API keys if you haven't already"
echo "       $PROJECT_ROOT/.env"
echo ""
echo "    2. Drop a PDF paper into papers/inbox/"
echo ""
echo "    3. Run the pipeline:"
echo "       start --report-file papers/inbox/your_paper.pdf"
echo ""
echo "    Quick test (paper extraction only, no factor generation):"
echo "       start --report-file papers/inbox/<paper>.pdf --extract-only"
echo ""
