#!/bin/bash
# 一键全量因子运行
# 双击即可执行（需先 chmod +x）

cd "$(dirname "$0")"
echo "============================================"
echo " 一键全量因子运行"
echo " 自动挂载远程E盘 → 扫描待运行因子 → 逐一计算"
echo "============================================"
echo ""

python3 scripts/run_all_pending_full.py

echo ""
echo "按 Enter 退出..."
read -r