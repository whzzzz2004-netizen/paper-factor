#!/bin/bash
# Run remaining minute_cs factors SEQUENTIALLY to avoid ProcessPoolExecutor conflicts
# Usage: ./scripts/run_minute_cs_sequentially.sh [--date YYYYMMDD]
set -e

DATE="${2:-20260727}"
BASE="/home/dministrator/paper-factor/git_ignore_folder/factor_outputs/literature_reports/${DATE}"
DATA_DIR="/home/dministrator/paper-factor/git_ignore_folder/factor_implementation_source_data"

export FACTOR_DATA_DIR="$DATA_DIR"
export FACTOR_CACHE_SIZE="20"
export FACTOR_CHUNK_SIZE="100"

export PATH="/home/dministrator/miniconda3/envs/rdagent/bin:$PATH"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# ==============================================================
# 1. IntradayVolatilityTiming (LB=5, 1 column: return) - QUICK
# ==============================================================
log "=== 1/4: IntradayVolatilityTiming (LB=5) ==="
cd "$BASE/日内波动择时因子/IntradayVolatilityTiming"
rm -rf .checkpoints_IntradayVolatilityTiming
export FACTOR_N_WORKERS="8"
timeout 7200 python3 -u IntradayVolatilityTiming.code.py 2>&1
log "=== IntradayVolatilityTiming done (exit=$?) ==="

# ==============================================================
# 2. JumpBeta (LB=5, memory-heavy - needs N_WORKERS=1)
# ==============================================================
log "=== 2/4: JumpBeta (LB=5, N_WORKERS=1 for memory) ==="
cd "$BASE/跳跃Beta与连续Beta/JumpBeta"
rm -rf .checkpoints_JumpBeta
export FACTOR_N_WORKERS="1"
export FACTOR_CACHE_SIZE="10"
timeout 14400 python3 -u JumpBeta.code.py 2>&1
log "=== JumpBeta done (exit=$?) ==="

# ==============================================================
# 3. SDRVOL (LB=30)
# ==============================================================
log "=== 3/4: SDRVOL (LB=30) ==="
cd "$BASE/日内交易特征稳定性与股票收益/SDRVOL"
rm -rf .checkpoints_SDRVOL
export FACTOR_N_WORKERS="4"
export FACTOR_CACHE_SIZE="40"
timeout 14400 python3 -u SDRVOL.code.py 2>&1
log "=== SDRVOL done (exit=$?) ==="

# ==============================================================
# 4. 跳跃Beta_连续Beta (LB=252, heaviest - needs N_WORKERS=1)
# ==============================================================
log "=== 4/4: 跳跃Beta_连续Beta (LB=252, N_WORKERS=1) ==="
cd "$BASE/跳跃Beta与连续Beta/跳跃Beta_连续Beta"
rm -rf .checkpoints_跳跃Beta_连续Beta
export FACTOR_N_WORKERS="1"
export FACTOR_CACHE_SIZE="252"
export FACTOR_CHUNK_SIZE="50"
timeout 28800 python3 -u 跳跃Beta_连续Beta.code.py 2>&1
log "=== 跳跃Beta_连续Beta done (exit=$?) ==="

log "=== ALL REMAINING FACTORS COMPLETE ==="
