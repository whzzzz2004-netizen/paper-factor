#!/bin/bash
# ============================================================
# 新机器复现配置脚本
# 用法: bash setup.sh
# ============================================================
set -e

REPO_URL="https://github.com/whzzzz2004-netizen/paper-factor.git"
REPO_DIR="$HOME/paper-factor"

echo "=========================================="
echo " paper-factor 复现配置"
echo "=========================================="

# ---------- 1. 克隆代码 ----------
if [ -d "$REPO_DIR" ]; then
    echo "[1/5] 仓库已存在，更新..."
    cd "$REPO_DIR" && git pull
else
    echo "[1/5] 克隆仓库..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ---------- 2. 创建数据目录 ----------
echo "[2/5] 创建数据目录结构..."
mkdir -p /mnt/d/paper-factor-data/数据仓库/行情数据/日线/{测试,全量}/stock_data/daily
mkdir -p /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/{测试,全量}/stock_data/{minute,minute_by_date}
mkdir -p /mnt/d/paper-factor-data/数据仓库/非行情数据/{测试,全量}/stock_data/daily
mkdir -p /mnt/d/paper-factor-data/数据仓库/因子产出/{测试,全量}
mkdir -p /mnt/d/paper-factor-data/数据仓库/barra_model
mkdir -p 原始数据
mkdir -p workspace/logs workspace/ideas
ln -sfn ..//mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量 workspace/factor_implementation_source_data
ln -sfn ..//mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试 workspace/factor_implementation_source_data_1000

echo ""
echo "  ⚠️  需要把数据仓库复制到 $REPO_DIR//mnt/d/paper-factor-data/数据仓库/"
echo "      从硬盘/开发机 rsync 即可:"
echo "      rsync -avhP user@dev:$REPO_DIR//mnt/d/paper-factor-data/数据仓库/ .//mnt/d/paper-factor-data/数据仓库/"

# ---------- 3. 环境配置 ----------
echo "[3/5] 配置 conda 环境..."
if command -v conda &>/dev/null; then
    if conda env list | grep -q "rdagent"; then
        echo "  conda 环境 rdagent 已存在，更新依赖..."
        conda run -n rdagent pip install -r requirements.txt
    else
        echo "  创建 conda 环境 rdagent (Python 3.10)..."
        conda create -n rdagent python=3.10 -y
        conda run -n rdagent pip install -r requirements.txt
    fi
else
    echo "  ⚠️ 未检测到 conda，请手动安装依赖:"
    echo "     pip install -r requirements.txt"
fi

# ---------- 4. 环境变量 ----------
echo "[4/5] 配置环境变量..."
if [ ! -f .env ]; then
    cat > .env <<'ENVEOF'
JQDATA_USERNAME=your_username
JQDATA_PASSWORD=your_password
FACTOR_DATA_DIR=$PWD//mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量
RDAGENT_FACTOR_DATA_DIR=$PWD//mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量
ENVEOF
    echo "  ✅ .env 已创建（请填写 JQData 账号）"
else
    echo "  .env 已存在，跳过"
fi

# ---------- 5. 验证 ----------
echo "[5/5] 验证..."
STOCK_COUNT=$(ls /mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量/stock_data/daily/*.parquet 2>/dev/null | wc -l)
if [ "$STOCK_COUNT" -gt 1000 ]; then
    echo "  ✅ 数据仓库正常: $STOCK_COUNT 只股票"
else
    echo "  ⚠️ 数据仓库未就绪（仅 $STOCK_COUNT 只股票，需要 5435 只）"
fi

echo ""
echo "=========================================="
echo " ✅ 配置完成！"
echo "=========================================="