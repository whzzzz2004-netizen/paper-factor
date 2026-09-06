#!/bin/bash
# ============================================================
# 从仓库 drive_files/ 同步交付副本到 D:\paper-factor-data
# 用法: bash sync_drive.sh
#
# 作用: 把 数据转换脚本(scripts) + 研报(papers) + 字段注册表(schema.json)
#       复制到 D 盘数据根目录。老板/新机器 git clone 后跑一次即可。
#
# 可配环境变量:
#   PAPER_FACTOR_DATA_ROOT  数据根目录 (默认 /mnt/d/paper-factor-data)
# ============================================================
set -e

# 仓库根 = 本脚本所在目录
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${PAPER_FACTOR_DATA_ROOT:-/mnt/d/paper-factor-data}"

echo "=========================================="
echo " 同步 drive_files → $DATA_ROOT"
echo "=========================================="

# 1. 数据转换脚本
echo "[1/4] scripts ..."
mkdir -p "$DATA_ROOT/scripts"
cp -f "$REPO_DIR"/drive_files/scripts/*.py "$DATA_ROOT/scripts/"

# 2. 研报 papers
echo "[2/4] papers ..."
mkdir -p "$DATA_ROOT/papers/inbox" "$DATA_ROOT/papers/website"
cp -f "$REPO_DIR"/drive_files/papers/inbox/* "$DATA_ROOT/papers/inbox/" 2>/dev/null || true
cp -f "$REPO_DIR"/drive_files/papers/website/* "$DATA_ROOT/papers/website/" 2>/dev/null || true

# 3. 字段注册表
echo "[3/4] schema.json ..."
mkdir -p "$DATA_ROOT"
cp -f "$REPO_DIR"/drive_files/schema.json "$DATA_ROOT/schema.json"

# 4. 写仓库路径（import_new_data.py 用它定位项目内 prompts.yaml）
echo "[4/4] repo_path.txt ..."
echo "$REPO_DIR" > "$DATA_ROOT/repo_path.txt"

echo ""
echo "✅ 同步完成 → $DATA_ROOT"
echo "   (scripts/papers/schema.json + repo_path.txt 已就位)"
