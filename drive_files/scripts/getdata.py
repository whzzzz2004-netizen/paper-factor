#!/usr/bin/env python3
"""
getdata 一键导入入口（替代 getdata.bat）。

在 Windows PowerShell 中直接运行：
  python getdata.py            # 完成 check → import → update-prompts → summary
  python getdata.py --check    # 仅扫描
  python getdata.py --dry-run  # 仅预览
  python getdata.py --summary  # 仅总结
  python getdata.py --update-prompts  # 仅刷提示词

实现 getdata.bat 的功能：顺序执行 4 步并给出阶段汇总。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN = SCRIPT_DIR / "import_new_data.py"


def run_py(args: list) -> int:
    cmd = [sys.executable, str(MAIN)] + args
    print(f"\n$ python import_new_data.py {' '.join(args)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="getdata 一键导入")
    p.add_argument("--check", action="store_true", help="仅扫描原始数据")
    p.add_argument("--dry-run", action="store_true", help="预览将导入内容")
    p.add_argument("--summary", action="store_true", help="仅打印总结")
    p.add_argument("--update-prompts", action="store_true", help="仅更新提示词")
    p.add_argument("--rebuild-trade-dates", action="store_true",
                   help="仅重建权威交易日历（日线 per-stock 实际 ∪ 分钟 per-date），不导入数据")
    args = p.parse_args()

    if args.check:
        return run_py(["--check"])
    if args.dry_run:
        return run_py(["--dry-run"])
    if args.summary:
        return run_py(["--summary"])
    if args.update_prompts:
        return run_py(["--update-prompts"])
    if args.rebuild_trade_dates:
        return run_py(["--rebuild-trade-dates"])

    # 默认：完整流程
    steps = [
        ("扫描检查", ["--check"]),
        ("导入数据", []),
        ("重建权威日历", ["--rebuild-trade-dates"]),
        ("刷新提示词", ["--update-prompts"]),
        ("结果总结", ["--summary"]),
    ]
    print("=" * 60)
    print(" getdata —— 一键增量导入")
    print("=" * 60)
    for title, argv in steps:
        print(f"\n>>> [{title}]")
        rc = run_py(argv)
        if rc != 0:
            print(f"\n❌ [{title}] 失败（exit {rc}），终止。")
            return rc
    print("\n" + "=" * 60)
    print(" getdata 全部完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())