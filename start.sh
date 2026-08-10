#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo "📊 Paper Factor 环境就绪"

# 找 Python
PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "❌ 未找到 Python，请先安装 Python 3"
  read -p "按回车退出..."
  exit 1
fi

echo "✅ Python: $($PYTHON --version)"
echo "📁 项目根目录: $(pwd)"
