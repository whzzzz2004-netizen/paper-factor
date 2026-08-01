#!/usr/bin/env python3
"""
分钟数据预分片：一趟扫描，按 stock chunk 拆分。
跨因子复用，只需跑一次。

用法:
  python scripts/pre_shard_minute.py                           # 测试 + 全量都分片
  python scripts/pre_shard_minute.py --data-type test          # 只分片测试数据
  python scripts/pre_shard_minute.py --data-type full          # 只分片全量数据
"""
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).parent.parent / "git_ignore_folder"

DATA_CONFIGS = {
    "test": {
        "data_dir": BASE / "factor_implementation_source_data_1000",
        "chunk_size": 25,
    },
    "full": {
        "data_dir": BASE / "factor_implementation_source_data",
        "chunk_size": 25,
    },
}


def pre_shard(data_dir: Path, chunk_size: int, label: str):
    minute_dir = data_dir / "stock_data" / "minute_by_date"
    chunk_dir = minute_dir / "_minute_chunks"

    stock_list = __import__("json").load(open(minute_dir / "stock_list.json"))
    trade_dates = __import__("json").load(open(minute_dir / "trade_dates.json"))
    all_files = sorted([minute_dir / f"{d}.parquet" for d in trade_dates
                        if (minute_dir / f"{d}.parquet").exists()])

    N = len(stock_list)
    n_files = len(all_files)
    chunks_list = [stock_list[i:i+chunk_size] for i in range(0, N, chunk_size)]
    n_chunks = len(chunks_list)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = [chunk_dir / f"_chunk_{ci}.pq" for ci in range(n_chunks)]

    if all(cf.exists() for cf in chunk_files):
        total_gb = sum(f.stat().st_size for f in chunk_files if f.exists()) / 1024**3
        print(f"[{label}] 分片已存在: {chunk_dir} ({total_gb:.1f} GB, {n_chunks} files), 跳过")
        return

    print(f"[{label}] 开始分片: {N} 只股票, {n_files} 文件, {n_chunks} chunks x{chunk_size}")

    # 读取所有列名
    import pyarrow.parquet as pq
    all_cols = sorted(set(pq.read_schema(next(minute_dir.glob("*.parquet"))).names)
                      - {'datetime', 'instrument'})
    print(f"[{label}] 列: {all_cols}")

    t0 = time.time()
    writers = [None] * n_chunks
    stock2ci = {}
    for ci, cstocks in enumerate(chunks_list):
        for s in cstocks:
            stock2ci[s] = ci

    try:
        for f_idx, f_path in enumerate(all_files):
            df = pd.read_parquet(f_path, columns=all_cols)
            ix = df.index.get_level_values('instrument')
            ci_s = pd.Series(ix).map(stock2ci)
            valid_mask = ci_s.notna()
            if not valid_mask.any():
                continue
            df = df.iloc[valid_mask.values].copy()
            df['_chunk_ci'] = ci_s[valid_mask].astype(int).values
            for ci, gdf in df.groupby('_chunk_ci'):
                gdf = gdf.drop(columns=['_chunk_ci'])
                table = pa.Table.from_pandas(gdf, preserve_index=True)
                if writers[ci] is None:
                    writers[ci] = pq.ParquetWriter(chunk_files[ci], table.schema)
                writers[ci].write_table(table)
            if f_idx % 200 == 0:
                print(f"  [{label}] 分片进度: {f_idx}/{n_files} 文件", flush=True)
    finally:
        for w in writers:
            if w is not None:
                w.close()

    n_created = sum(1 for f in chunk_files if f.exists())
    total_gb = sum(f.stat().st_size for f in chunk_files if f.exists()) / 1024**3
    print(f"[{label}] 分片完成: {time.time()-t0:.0f}s, {n_created}/{n_chunks} files, {total_gb:.1f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="分钟数据预分片")
    parser.add_argument("--data-type", choices=["test", "full", "all"], default="all")
    args = parser.parse_args()

    if args.data_type in ("test", "all"):
        cfg = DATA_CONFIGS["test"]
        if cfg["data_dir"].exists():
            pre_shard(cfg["data_dir"], cfg["chunk_size"], "test")
        else:
            print(f"[test] 数据目录不存在: {cfg['data_dir']}, 跳过")

    if args.data_type in ("full", "all"):
        cfg = DATA_CONFIGS["full"]
        if cfg["data_dir"].exists():
            pre_shard(cfg["data_dir"], cfg["chunk_size"], "full")
        else:
            print(f"[full] 数据目录不存在: {cfg['data_dir']}, 跳过")
