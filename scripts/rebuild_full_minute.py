#!/usr/bin/env python3
"""用真实分钟原始数据重建全量分钟数据仓库（minute_by_date）。

输入:  /mnt/d/market_minute_daily_new/{YYYYMMDD}.parquet
        flat 格式, 10 列: symbol/trade_date/open/high/low/close/volume/return/factor/date
输出:  /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/{YYYYMMDD}.parquet
        MultiIndex[instrument, datetime], 7 数据列, instrument 统一 6 位, datetime ns 精度

用法:
    python scripts/rebuild_full_minute.py                 # 转换全部日期到目标目录
    python scripts/rebuild_full_minute.py --dry-run       # 只打印计划, 不写盘
    python scripts/rebuild_full_minute.py --workers 8     # 并行进程数
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW_MINUTE_DIR = Path("/mnt/d/market_minute_daily_new")
WH = Path("/mnt/d/paper-factor-data/数据仓库")
FULL_MINUTE_BY_DATE = WH / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute_by_date"

# 冗余列：flat 格式里有, 目标 parquet 不需要（symbol→instrument 6位, trade_date→datetime index, date 冗余）
DROP_COLS = ["symbol", "trade_date", "date"]


def _convert_one(date_str: str, dst_dir: Path) -> str:
    """转换单日文件。返回 date_str（成功）或抛异常（由外层收集错误）。"""
    src = RAW_MINUTE_DIR / f"{date_str}.parquet"
    dst = dst_dir / f"{date_str}.parquet"

    df = pd.read_parquet(src)

    # 统一股票代码为 6 位字符串（用户明确要求：股票代码统一填充到 6 位）
    if df["symbol"].dtype.kind in "iu":
        df["instrument"] = df["symbol"].astype(str).str.zfill(6)
    else:
        df["instrument"] = df["symbol"].astype(str).str.zfill(6)

    # trade_date → datetime 列
    if df["trade_date"].dtype.kind != "M":
        df["datetime"] = pd.to_datetime(df["trade_date"], errors="coerce")
    else:
        df["datetime"] = df["trade_date"]

    # 丢弃冗余列
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 数据列顺序固定
    data_cols = ["open", "high", "low", "close", "volume", "return", "factor"]
    extra = [c for c in df.columns if c not in data_cols + ["instrument", "datetime"]]
    keep_cols = [c for c in data_cols if c in df.columns] + extra
    df = df[keep_cols + ["instrument", "datetime"]]

    df = df.set_index(["instrument", "datetime"]).sort_index()
    # datetime 层统一 ns 精度（与模板/现有数据一致，防分片 schema 不匹配）
    df.index = df.index.set_levels(df.index.levels[1].as_unit("ns"), level="datetime")

    df.to_parquet(dst, compression="snappy")
    return date_str


def main():
    parser = argparse.ArgumentParser(description="用真实分钟原始数据重建全量分钟 minute_by_date")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不写盘")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数")
    parser.add_argument("--dst", default=str(FULL_MINUTE_BY_DATE), help="输出目录")
    args = parser.parse_args()

    dst_dir = Path(args.dst)
    if not RAW_MINUTE_DIR.exists():
        print(f"❌ 原始数据目录不存在: {RAW_MINUTE_DIR}")
        return 1

    dates = sorted(p.stem for p in RAW_MINUTE_DIR.glob("2*.parquet"))
    if not dates:
        print("❌ 原始数据目录下没有 2*.parquet 文件")
        return 1

    print(f"原始数据: {RAW_MINUTE_DIR}")
    print(f"目标目录: {dst_dir}")
    print(f"共 {len(dates)} 天: {dates[0]} ~ {dates[-1]}", flush=True)

    # 跳过已存在的（断点续跑）
    dst_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in dst_dir.glob("2*.parquet")}
    need = [d for d in dates if d not in existing]
    print(f"待转换 {len(need)} 天（已存在 {len(existing)}）", flush=True)
    if args.dry_run or not need:
        return 0

    t0 = time.time()
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_convert_one, d, dst_dir): d for d in need}
        for f in as_completed(futs):
            d = futs[f]
            try:
                f.result()
                done += 1
            except Exception as e:
                errors.append((d, str(e)))
                print(f"  ❌ {d}: {e}", flush=True)
            if done % 100 == 0 or done == len(need):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                print(f"  [{done}/{len(need)}] {time.time()-t0:.0f}s ({rate:.1f} 文件/秒)", flush=True)

    if errors:
        print(f"\n⚠️ {len(errors)} 个文件转换失败: {[d for d, _ in errors][:20]}")
        return 1

    print(f"\n✅ 完成: {len(need)} 个文件, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
