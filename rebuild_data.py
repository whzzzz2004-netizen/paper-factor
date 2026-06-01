"""从远程 parquet 文件重建本地数据（全部股票 ≈ 5185 只）
日线 → daily_pv.h5 (HDF5, 兼容现有读取代码)
分钟 → minute_pv.parquet/ (目录, zstd 压缩, 比 H5 小 60-70%)
"""
import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

LOCAL_DATA_DIR = Path("/home/dministrator/paper-factor/git_ignore_folder/factor_implementation_source_data")
REMOTE_BASE = Path("/mnt/remote_e")
DAILY_SRC = REMOTE_BASE / "market_daily_daily_new"
MINUTE_SRC = REMOTE_BASE / "market_minute_daily_new"

OLD_DAILY_H5 = LOCAL_DATA_DIR / "daily_pv.h5"
NEW_DAILY_H5 = LOCAL_DATA_DIR / "daily_pv.h5"
MINUTE_PARQUET_DIR = LOCAL_DATA_DIR / "minute_pv.parquet"

# 日期范围：2024-05-01 ~ 2026-05-15（近两年，与远程最新文件对齐）
START_DATE = "2024-05-01"
END_DATE = "2026-05-15"

print("=" * 60)
print("Step 1: 读取旧 H5 中的 111 个基本面因子（80 只股票）")
print("=" * 60)
with pd.HDFStore(str(OLD_DAILY_H5), "r") as store:
    old_daily = store["/data"]
    # 基本面因子列名：以 $ 开头且不是 price/volume 基础列
    base_cols = {"$open", "$close", "$high", "$low", "$factor", "$volume", "$pre_close", "$pct_chg", "$turnover_rate"}
    fundamental_cols = [c for c in old_daily.columns if c not in base_cols]
    old_fundamentals = old_daily[fundamental_cols].copy()
    print(f"  基本面因子列数: {len(fundamental_cols)}")
    print(f"  原数据行数: {len(old_daily)}")
    print(f"  原股票数: {old_daily.index.get_level_values('instrument').nunique()}")

print()
print("=" * 60)
print("Step 2: 重建日线 H5（5185 只股票）")
print("=" * 60)

daily_files = sorted([f for f in os.listdir(str(DAILY_SRC)) if f.endswith(".parquet")])
daily_files = [f for f in daily_files if START_DATE <= f.replace(".parquet", "") <= END_DATE.replace("-", "")]
print(f"  处理 {len(daily_files)} 个日线文件...")

all_daily_rows = []
for i, fname in enumerate(daily_files):
    if i % 100 == 0:
        print(f"  进度: {i}/{len(daily_files)}")
    df = pd.read_parquet(str(DAILY_SRC / fname))
    # 重命名列
    df = df.rename(columns={
        "symbol": "instrument",
        "date": "datetime",
        "open": "$open",
        "close": "$close",
        "high": "$high",
        "low": "$low",
        "volume": "$volume",
        "factor": "$factor",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    # 只保留需要的列
    keep_cols = ["datetime", "instrument", "$open", "$close", "$high", "$low", "$factor", "$volume"]
    df = df[keep_cols]
    # 计算 $pct_chg（fill_method=None 避免 FutureWarning）
    df["$pct_chg"] = df.groupby("instrument")["$close"].pct_change(fill_method=None) * 100
    # pre_close: 前一天的 close
    df["$pre_close"] = df.groupby("instrument")["$close"].shift(1)
    # $turnover_rate 远程数据中没有，留 NaN
    df["$turnover_rate"] = np.nan
    all_daily_rows.append(df)

print(f"  合并数据...")
daily_full = pd.concat(all_daily_rows, ignore_index=True)
daily_full["datetime"] = daily_full["datetime"].dt.strftime("%Y-%m-%d")
print(f"  总行数: {len(daily_full)}, 股票数: {daily_full['instrument'].nunique()}")

# 合并基本面因子（只对原有的 80 只股票有效）—— 用 pd.concat 避免碎片化
print(f"  合并基本面因子...")
daily_full = daily_full.set_index(["datetime", "instrument"])
common_idx = daily_full.index.intersection(old_fundamentals.index)

# 一次性创建所有基本面因子列，避免逐列插入导致碎片化
fundamental_data = {}
for col in fundamental_cols:
    series = pd.Series(np.nan, index=daily_full.index, dtype=old_fundamentals[col].dtype)
    series.loc[common_idx] = old_fundamentals.loc[common_idx, col]
    fundamental_data[col] = series

daily_full = pd.concat([daily_full, pd.DataFrame(fundamental_data)], axis=1)

# 排序 index
daily_full = daily_full.sort_index()

print(f"  写入 H5（{len(daily_full)} 行 × {len(daily_full.columns)} 列）...")
daily_full.to_hdf(str(NEW_DAILY_H5), key="/data", mode="w", format="fixed")
print(f"  日线 H5 完成！大小: {NEW_DAILY_H5.stat().st_size / 1e9:.2f} GB")

del daily_full, all_daily_rows, old_fundamentals

print()
print("=" * 60)
print("Step 3: 重建分钟数据（Parquet + zstd, 全部股票）")
print("=" * 60)

minute_files = sorted([f for f in os.listdir(str(MINUTE_SRC)) if f.endswith(".parquet")])
minute_files = [f for f in minute_files if START_DATE <= f.replace(".parquet", "") <= END_DATE.replace("-", "")]
print(f"  处理 {len(minute_files)} 个分钟文件...")

# 分钟数据量很大，按月分批处理
monthly_batches = defaultdict(list)
for fname in minute_files:
    ym = fname[:6]  # YYYYMM
    monthly_batches[ym].append(fname)

print(f"  共 {len(monthly_batches)} 个月份批次")

# 清理并重建输出目录
import shutil
if MINUTE_PARQUET_DIR.exists():
    shutil.rmtree(str(MINUTE_PARQUET_DIR))
MINUTE_PARQUET_DIR.mkdir(parents=True)

# 逐月处理，每月输出一个 parquet 文件
processed = 0
for ym in sorted(monthly_batches.keys()):
    if processed % 3 == 0:
        print(f"  进度: {processed}/{len(minute_files)}")

    frames = []
    for fname in monthly_batches[ym]:
        df = pd.read_parquet(str(MINUTE_SRC / fname))
        df = df.rename(columns={
            "symbol": "instrument",
            "trade_date": "datetime",
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "return": "$return",
            "factor": "$factor",
        })
        df["$vwap"] = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4
        keep_cols = ["datetime", "instrument", "$open", "$high", "$low", "$close", "$volume", "$return", "$factor", "$vwap"]
        frames.append(df[keep_cols])

    monthly_df = pd.concat(frames, ignore_index=True)
    monthly_df["datetime"] = pd.to_datetime(monthly_df["datetime"])

    # 写入按月分片的 parquet 文件（zstd 压缩）
    out_path = MINUTE_PARQUET_DIR / f"{ym}.parquet"
    monthly_df.to_parquet(str(out_path), compression="zstd", index=False)
    processed += len(monthly_batches[ym])
    del monthly_df, frames

print(f"  分钟 Parquet 完成！共 {len(list(MINUTE_PARQUET_DIR.iterdir()))} 个文件")

# 验证 & 总大小
total_size = sum(f.stat().st_size for f in MINUTE_PARQUET_DIR.iterdir())
print(f"  总大小: {total_size / 1e9:.2f} GB")

# 验证
print()
print("=" * 60)
print("验证新数据")
print("=" * 60)

# 验证日线
with pd.HDFStore(str(NEW_DAILY_H5), "r") as store:
    df = store["/data"]
    instruments = df.index.get_level_values("instrument").nunique()
    dates = df.index.get_level_values("datetime").nunique()
    print(f"日线: {len(df)} 行, {instruments} 只股票, {dates} 个交易日")
    print(f"  列: {list(df.columns[:10])}...")

# 验证分钟（读取列信息和统计摘要）
print(f"分钟: 读取分片验证...")
minute_files = sorted(MINUTE_PARQUET_DIR.glob("*.parquet"))
sample_df = pd.read_parquet(str(minute_files[0]))
print(f"  列: {list(sample_df.columns)}")
print(f"  分片数: {len(minute_files)}")

# 快速统计总行数（逐文件读 datetime 列避免 OOM）
total_minute_rows = 0
all_instruments = set()
for f in minute_files:
    meta = pd.read_parquet(str(f), columns=["instrument"])
    total_minute_rows += len(meta)
    all_instruments.update(meta["instrument"].unique())
    del meta
print(f"  总行数: {total_minute_rows}, 股票数: {len(all_instruments)}")

# 更新 meta 文件
meta = {
    "source": "remote_e",
    "updated_on": datetime.now().strftime("%Y-%m-%d"),
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily_rows": len(pd.HDFStore(str(NEW_DAILY_H5), "r")["/data"]),
    "minute_rows": total_minute_rows,
    "minute_files": len(minute_files),
    "instruments_full": int(len(all_instruments)),
    "fundamental_factors": len(fundamental_cols),
    "minute_format": "parquet_zstd",
    "daily_format": "hdf5",
}
with open(str(LOCAL_DATA_DIR / "remote_data_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(f"\nmeta 已更新: {meta}")
print("\n✅ 数据重建完成！")
