#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo "📊 Paper Factor Dashboard 启动中..."

# ── 挂载远程E盘（数据源）──
MOUNT_POINT="/mnt/remote_e"
if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  mkdir -p "$MOUNT_POINT"
  sudo -n mount -t cifs "//192.168.1.13/E" "$MOUNT_POINT" \
    -o user=pc,password=123456,uid=$(id -u),gid=$(id -g),file_mode=0644,dir_mode=0755,iocharset=utf8,noperm 2>/dev/null
  if mountpoint -q "$MOUNT_POINT"; then
    echo "  ✅ 远程E盘已挂载"
  else
    echo "  ⚠️ 远程E盘未挂载，使用本地数据"
  fi
fi

# 设置数据路径环境变量
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  export FACTOR_DATA_DIR="$MOUNT_POINT/_paper_factor_unified/factor_implementation_source_data"
  export PAPER_FACTOR_BARRA_DIR="$MOUNT_POINT/_paper_factor_unified/barra_model"
  export FACTOR_DATA_DIR_1000="$MOUNT_POINT/_paper_factor_unified/factor_implementation_source_data_1000"
fi
# 不设回退到默认本地路径

# 确保 cron 运行（用于每日更新定时任务）
if ! pgrep -a cron | grep -q cron; then
  /usr/sbin/cron -P 2>/dev/null
  echo "  ⏰ cron 已启动"
fi
mkdir -p git_ignore_folder/logs

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

# 杀死旧的 dashboard 进程（避免装了旧版本不退出）
OLDPIDS=$(pgrep -f "dashboard_server.py" | grep -v $$ 2>/dev/null)
if [ -n "$OLDPIDS" ]; then
  echo "🔄 关闭旧版 Dashboard..."
  echo "$OLDPIDS" | xargs kill 2>/dev/null
  sleep 1
fi

"$PYTHON" dashboard_server.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo ""
  echo "❌ 启动失败 (代码: $EXIT_CODE)"
  echo "请尝试手动运行: $PYTHON dashboard_server.py"
  read -p "按回车退出..."
fi
