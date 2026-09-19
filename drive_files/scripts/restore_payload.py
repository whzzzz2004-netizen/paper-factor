#!/usr/bin/env python3
"""
restore_payload.py — 把 scripts/deploy_payload/ 里的元数据 json 还原到数据仓库对应位置。

用法:
  python restore_payload.py            # 还原所有 deploy_payload 文件
  python restore_payload.py --dry-run  # 只打印将放到哪，不实际拷贝

命名规律：deploy_payload 里的文件名 = 目标相对路径（用「__」表示目录分隔）。
  数据仓库__行情数据__日线__全量__stock_data__daily__trade_dates.json
  → 数据仓库/行情数据/日线/全量/stock_data/daily/trade_dates.json
  schema.json → 数据根/schema.json

注意：目标机数据根默认 D:\paper-factor-data；若不同，请修改 DATA_ROOT。
"""

import argparse
import shutil
from pathlib import Path

DATA_ROOT = Path(r"D:\paper-factor-data")
PAYLOAD = DATA_ROOT / "scripts" / "deploy_payload"


def target_path(flat_name: str) -> Path:
    # 「数据仓库__…」→ 数据仓库/…；schema.json → 根；其它同理
    if flat_name.startswith("数据仓库__"):
        rel = flat_name.replace("__", "\\", 1)
    else:
        rel = flat_name
    rel = rel.replace("__", "\\")
    return DATA_ROOT / rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印不拷贝")
    args = ap.parse_args()

    if not PAYLOAD.is_dir():
        print("未找到", PAYLOAD)
        return 1
    items = sorted(p for p in PAYLOAD.iterdir() if p.is_file())
    if not items:
        print("deploy_payload 为空")
        return 1

    for f in items:
        dst = target_path(f.name)
        print(" ", f.name)
        print("   ->", dst)
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)

    print()
    print("dry-run 完成（未改动）" if args.dry_run else "✅ 全部还原完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())