#!/bin/bash
# Run remaining minute_cs factors sequentially
# Each factor runs with 2 workers, saves checkpoints, and can be resumed

set -e

DATA_DIR="/home/dministrator/paper-factor/git_ignore_folder/factor_implementation_source_data"
BASE="/home/dministrator/paper-factor/git_ignore_folder/factor_outputs/literature_reports/20260727"

export FACTOR_DATA_DIR="$DATA_DIR"
export FACTOR_N_WORKERS=2
export FACTOR_CACHE_SIZE=10
export FACTOR_CHUNK_SIZE=200

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# 1. 量能分歧 (LB=20)
log "=== Starting 量能分歧 ==="
cd "$BASE/基于资金推动力的价量张力因子构建/量能分歧"
rm -rf .checkpoints_量能分歧
timeout 7200 python3 -u 量能分歧.code.py 2>&1
log "=== 量能分歧 done (exit=$?) ==="

# 2. 弹力势差 (LB=20)
log "=== Starting 弹力势差 ==="
cd "$BASE/基于资金推动力的价量张力因子构建/弹力势差"
rm -rf .checkpoints_弹力势差
timeout 7200 python3 -u 弹力势差.code.py 2>&1
log "=== 弹力势差 done (exit=$?) ==="

# 3. SDRVOL (LB=30)
log "=== Starting SDRVOL ==="
cd "$BASE/日内交易特征稳定性与股票收益/SDRVOL"
rm -rf .checkpoints_SDRVOL
timeout 7200 python3 -u SDRVOL.code.py 2>&1
log "=== SDRVOL done (exit=$?) ==="

# 4. JumpBeta (LB=5) - resume from checkpoints if any
log "=== Starting JumpBeta ==="
cd "$BASE/跳跃Beta与连续Beta/JumpBeta"
# Keep checkpoints if they exist for resume
timeout 7200 python3 -u JumpBeta.code.py 2>&1
log "=== JumpBeta done (exit=$?) ==="

# 5. 跳跃Beta_连续Beta (LB=252)
log "=== Starting 跳跃Beta_连续Beta ==="
cd "$BASE/跳跃Beta与连续Beta/跳跃Beta_连续Beta"
rm -rf .checkpoints_跳跃Beta_连续Beta
timeout 7200 python3 -u 跳跃Beta_连续Beta.code.py 2>&1
log "=== 跳跃Beta_连续Beta done (exit=$?) ==="

log "=== ALL FACTORS COMPLETE ==="
