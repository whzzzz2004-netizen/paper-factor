#!/usr/bin/env python3
"""
预计算市场代理：全市场等权平均分钟收益率。

扫描 minute_by_date/*.parquet，对每分钟计算全市场等权平均 return，
保存为 market_minute_return.parquet（与 minute_by_date 同目录）。

用法:
    python scripts/precompute_market_proxy.py --data-dir "数据仓库/行情数据/分钟线/测试"
    python scripts/precompute_market_proxy.py --data-dir "数据仓库/行情数据/分钟线/全量"
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="预计算市场代理：全市场等权平均分钟收益率")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="分钟线数据目录（含 stock_data/minute_by_date/）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    minute_by_date_dir = data_dir / "stock_data" / "minute_by_date"
    if not minute_by_date_dir.exists():
        print(f"错误: 目录不存在 {minute_by_date_dir}")
        return 1

    output_path = minute_by_date_dir / "market_minute_return.parquet"
    print(f"输出: {output_path}", flush=True)

    # 扫描所有 parquet 文件（排除 chunk 目录和已有 market 文件）
    all_files = sorted(minute_by_date_dir.glob("*.parquet"))
    all_files = [f for f in all_files if "market_minute_return" not in f.name]
    print(f"找到 {len(all_files)} 个日频分钟文件", flush=True)

    t0 = time.time()
    records = []
    for i, f in enumerate(all_files):
        df = pd.read_parquet(f, columns=["return"])
        # 按 datetime 分组取等权平均
        # minute_by_date 的 index 是 MultiIndex[instrument, datetime]
        # 需要按 datetime level 分组
        grp = df.groupby(level="datetime")["return"].mean()
        for dt, val in grp.items():
            if pd.notna(val) and np.isfinite(val):
                records.append({"datetime": dt, "market_return": float(val)})
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(all_files)} 文件, {time.time()-t0:.0f}s", flush=True)

    if not records:
        print("错误: 没有产生任何记录", flush=True)
        return 1

    result = pd.DataFrame(records)
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.set_index("datetime").sort_index()
    # 去重（同一分钟可能有多个条目，取均值）
    result = result.groupby(level="datetime").mean()
    result.to_parquet(output_path)
    print(f"完成: {len(result)} 分钟, {time.time()-t0:.0f}s", flush=True)
    print(f"  日期范围: {result.index[0]} ~ {result.index[-1]}", flush=True)
    print(f"  market_return 统计: mean={result['market_return'].mean():.6f}, "
          f"std={result['market_return'].std():.6f}", flush=True)
    return 0

if __name__ == "__main__":
    exit(main())