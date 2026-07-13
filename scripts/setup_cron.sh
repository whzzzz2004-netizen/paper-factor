#!/bin/bash
# 因子每日更新定时任务管理脚本
# 用法:
#   ./scripts/setup_cron.sh status    # 查看 cron 状态
#   ./scripts/setup_cron.sh enable HH:MM  # 启用定时任务 (如: enable 17:00)
#   ./scripts/setup_cron.sh disable   # 禁用定时任务
#   ./scripts/setup_cron.sh set-time HH:MM  # 修改执行时间

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CRON_CMD="cd $PROJECT_DIR && python scripts/sync_data.py && python scripts/daily_update.py --workers 3 >> git_ignore_folder/logs/daily_update.log 2>&1"
CRON_ID="paper_factor_daily_update"

case "${1:-status}" in
  status)
    echo "=== Cron 服务状态 ==="
    if pgrep -a cron | grep -q cron; then
      echo "  ✅ cron 运行中"
    else
      echo "  ❌ cron 未运行"
    fi
    echo ""
    echo "=== 当前定时任务 ==="
    crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$" || echo "  (无)"
    echo ""
    echo "=== 每日更新因子数 ==="
    python "$PROJECT_DIR/scripts/daily_update.py" --dry-run 2>&1 | grep -E "因子数|失败|已最新"
    ;;

  enable)
    if [ -z "$2" ]; then
      echo "用法: $0 enable HH:MM  (如: $0 enable 17:00)"
      exit 1
    fi
    HOUR="${2%%:*}"
    MIN="${2##*:}"
    # 去掉前导0
    HOUR=$((10#$HOUR))
    MIN=$((10#$MIN))

    # 移除旧的 paper_factor 任务，添加新任务
    (crontab -l 2>/dev/null | grep -v "$CRON_ID"; echo "$MIN $HOUR * * 1-5 $CRON_CMD # $CRON_ID") | crontab -
    echo "✅ 已启用: 每个交易日 $2 执行"
    echo "  命令: python scripts/sync_data.py && python scripts/daily_update.py --workers 3"
    echo "  日志: git_ignore_folder/logs/daily_update.log"
    ;;

  disable)
    (crontab -l 2>/dev/null | grep -v "$CRON_ID") | crontab -
    echo "✅ 已禁用定时任务"
    ;;

  set-time)
    if [ -z "$2" ]; then
      echo "用法: $0 set-time HH:MM"
      exit 1
    fi
    # 先检查是否已启用，如已启用则直接改时间
    if crontab -l 2>/dev/null | grep -q "$CRON_ID"; then
      "$0" enable "$2"
    else
      echo "定时任务未启用，请先用 enable"
    fi
    ;;

  *)
    echo "用法: $0 {status|enable|disable|set-time}"
    exit 1
    ;;
esac