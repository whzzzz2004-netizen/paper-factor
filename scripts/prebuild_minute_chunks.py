#!/usr/bin/env python3
"""从全历史分钟 chunk 缓存快速生成「起点截止」的 chunk 缓存。

分钟模板首次遇到新起点（默认 2022-01-01）时，会逐日读取 1100+ 个分钟文件
重新预分片，实测约 40~60 分钟。本脚本复用已有的全历史缓存
`_minute_chunks`，按日期行组过滤（Parquet predicate pushdown）后写出
目标缓存，实测约 11 分钟（16.9GB），产物与模板预分片完全兼容。

用法：
  python scripts/prebuild_minute_chunks.py                # 默认 2022-01-01
  python scripts/prebuild_minute_chunks.py --start 2023-01-01
  python scripts/prebuild_minute_chunks.py --workers 4

说明：
  - 全历史缓存不存在 / 已过期（数据文件有更新）/ chunk 结构与当前股票列表
    不一致时，脚本会拒绝生成并返回非 0，此时让模板自己重建即可。
  - 目标缓存已有效时直接跳过（幂等）。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_paths import DATA_ROOT  # noqa: E402
from scripts.factor_utils import MINUTE_FULL_WARMUP_DAYS  # noqa: E402

MINUTE_FULL_DIR = DATA_ROOT / "数据仓库" / "行情数据" / "分钟线" / "全量"
MB_DIR = MINUTE_FULL_DIR / "stock_data" / "minute_by_date"
SRC_DIR = MB_DIR / "_minute_chunks"


def _window_date_files(start: str) -> list[str]:
    td = json.loads((MB_DIR / "trade_dates.json").read_text())
    idx = pd.DatetimeIndex(td)
    pos = max(0, idx.searchsorted(pd.Timestamp(start)) - int(MINUTE_FULL_WARMUP_DAYS))
    return td[pos:]


def _data_sig(date_files: list[str]) -> dict:
    files = [MB_DIR / f"{d}.parquet" for d in date_files if (MB_DIR / f"{d}.parquet").exists()]
    return {
        "n_files": len(files),
        "last_file": files[-1].name if files else None,
        "max_mtime": max((f.stat().st_mtime_ns for f in files), default=0),
    }


def _copy_chunk(args) -> tuple[int, int]:
    """读源 chunk（按 datetime 行组过滤）→ 写目标 chunk。"""
    src, dst, start_ts = args
    filters = [("datetime", ">=", start_ts)]
    df = pd.read_parquet(src, filters=filters)
    inc = src.with_name(src.name.replace(".pq", ".inc.pq"))
    if inc.exists():
        try:
            _inc = pd.read_parquet(inc, filters=filters)
            if len(_inc):
                df = pd.concat([df, _inc])
        except Exception:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=True)
    return len(df), dst.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description="快速生成分钟因子起点 chunk 缓存")
    ap.add_argument("--start", default=os.environ.get("FACTOR_MINUTE_FULL_START_DATE", "2022-01-01"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    start = args.start
    target = MB_DIR / f"_minute_chunks_{pd.Timestamp(start).strftime('%Y%m%d')}"
    date_files = _window_date_files(start)
    expected_sig = _data_sig(date_files)

    # 目标缓存已有效 → 跳过
    tgt_manifest = target / "_manifest.json"
    if tgt_manifest.exists():
        try:
            m = json.loads(tgt_manifest.read_text())
            if m.get("data_sig") == expected_sig:
                print(f"✅ 目标缓存已有效，跳过: {target}")
                return 0
        except Exception:
            pass

    if not (SRC_DIR / "_manifest.json").exists():
        print(f"❌ 全历史缓存不存在，无法快速生成: {SRC_DIR}")
        return 1

    src_m = json.loads((SRC_DIR / "_manifest.json").read_text())
    stock_list = json.loads((MB_DIR / "stock_list.json").read_text())
    chunk_size = int(src_m.get("chunk_size", 25))
    n_chunks = int(src_m.get("n_chunks") or ((len(stock_list) + chunk_size - 1) // chunk_size))
    n_expected = (len(stock_list) + chunk_size - 1) // chunk_size

    if src_m.get("stocks") != sorted(stock_list):
        print("❌ 股票列表与全历史缓存不一致（缓存需重建），跳过快速生成")
        return 1
    if n_chunks != n_expected:
        print(f"❌ chunk 数不匹配: manifest={n_chunks}, 当前股票列表应为 {n_expected}")
        return 1

    # 全历史缓存必须是最新的（数据更新后模板会重建，这里也拒绝生成）
    full_sig = _data_sig(json.loads((MB_DIR / "trade_dates.json").read_text()))
    if src_m.get("data_sig") != full_sig:
        print("❌ 全历史缓存已过期（数据文件有更新），跳过快速生成")
        return 1

    missing = [i for i in range(n_chunks) if not (SRC_DIR / f"_chunk_{i}.pq").exists()]
    if missing:
        print(f"❌ 全历史缓存缺少 {len(missing)} 个 chunk（如 {missing[0]}），跳过快速生成")
        return 1

    # 源 chunk 的列必须覆盖当前数据文件的全部列（否则过滤后仍会缺列）
    sample_raw = next(MB_DIR.glob("*.parquet"))
    raw_cols = set(pq.read_schema(sample_raw).names) - {"datetime", "instrument"}
    src_cols = set(pq.read_schema(SRC_DIR / "_chunk_0.pq").names) - {
        "datetime", "instrument", "__index_level_0__"
    }
    if not raw_cols.issubset(src_cols):
        print(f"❌ 全历史缓存缺少列 {sorted(raw_cols - src_cols)}，跳过快速生成")
        return 1

    target.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(date_files[0])
    print(f"源: {SRC_DIR} ({n_chunks} chunks × {chunk_size})")
    print(f"目标: {target}")
    print(f"窗口: {date_files[0]} ~ {date_files[-1]} ({len(date_files)} 个交易日), workers={args.workers}")

    t0 = time.time()
    tasks = [(SRC_DIR / f"_chunk_{i}.pq", target / f"_chunk_{i}.pq", start_ts)
             for i in range(n_chunks)]
    total_rows = total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (rows, nbytes) in enumerate(pool.map(_copy_chunk, tasks), 1):
            total_rows += rows
            total_bytes += nbytes
            if i % 50 == 0 or i == n_chunks:
                el = time.time() - t0
                print(f"  进度 {i}/{n_chunks} chunks, {el:.0f}s, {total_bytes/1e9:.1f} GB", flush=True)

    # 清理超出当前 chunk 数的旧文件（历史 chunk_size 切换可能留下）
    for p in target.glob("_chunk_*.pq"):
        try:
            ci = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        if ci >= n_chunks:
            p.unlink()
    for p in target.glob("_chunk_*.inc.pq"):
        p.unlink()

    manifest = {
        "stocks": sorted(stock_list),
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "data_sig": expected_sig,
        "done_files": [f"{d}.parquet" for d in date_files],
    }
    tgt_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    print(f"✅ 完成: {n_chunks} chunks, {total_rows} 行, {total_bytes/1e9:.1f} GB, "
          f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
