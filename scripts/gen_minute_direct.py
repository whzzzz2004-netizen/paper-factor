#!/usr/bin/env python3
"""生成分钟数据直接写入 minute_by_date（MultiIndex 格式，跳过 import 管道）。"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/mnt/d/paper-factor-data")
WH = DATA_ROOT / "数据仓库"
MINUTE_BY_DATE = WH / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute_by_date"
MINUTE_META = MINUTE_BY_DATE.parent  # stock_data/
DAILY_STOCK_LIST = WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily" / "stock_list.json"
DAILY_TRADE_DATES = WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily" / "trade_dates.json"


def _ensure_unique_dates():
    """确保 minute_by_date 的 trade_dates.json 与 stock_data/ 层级的保持一致。"""
    src = MINUTE_META / "trade_dates.json"
    dst = MINUTE_BY_DATE / "trade_dates.json"
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"  同步 trade_dates.json: {dst}")


def _gen_one_file(date_str: str, stocks_int: list, n_minutes: int = 242):
    """生成一个 MultiIndex 格式的分钟 parquet 文件。"""
    n = len(stocks_int)
    base = np.random.uniform(5, 100, n)
    rets = np.random.normal(0, 0.0003, (n_minutes, n))
    prices = base * np.exp(np.cumsum(rets, axis=0))

    # 展平为长格式: stock-major, minute-minor
    n_total = n * n_minutes
    instrument = np.char.zfill(np.repeat(stocks_int, n_minutes).astype(str), 6)
    dt = pd.Timestamp(date_str)
    datetime = np.empty(n_total, dtype="datetime64[us]")
    datetime[:] = dt.to_datetime64()

    open_vals = np.repeat(prices[0], n_minutes)
    high_vals = np.repeat(prices.max(axis=0), n_minutes)
    low_vals = np.repeat(prices.min(axis=0), n_minutes)
    close_vals = prices.T.ravel()
    volume_vals = np.random.randint(10000, 10000000, n_total).astype(np.int64)
    return_vals = rets.T.ravel()
    factor_vals = np.ones(n_total, dtype=np.float64)

    df = pd.DataFrame({
        "open": open_vals, "high": high_vals, "low": low_vals,
        "close": close_vals, "volume": volume_vals,
        "return": return_vals, "factor": factor_vals,
    }, index=pd.MultiIndex.from_arrays(
        [instrument, datetime], names=["instrument", "datetime"]
    ))

    dst = MINUTE_BY_DATE / f"{date_str}.parquet"
    pq.write_table(pa.Table.from_pandas(df), dst)
    return date_str


def main():
    MINUTE_BY_DATE.mkdir(parents=True, exist_ok=True)

    with open(DAILY_TRADE_DATES) as f:
        all_dates = json.load(f)
    with open(DAILY_STOCK_LIST) as f:
        stocks = json.load(f)
    stocks_int = [int(s) for s in stocks]

    # 已有文件
    existing = {f.stem for f in MINUTE_BY_DATE.glob("2*.parquet")}
    need = sorted(d for d in all_dates if d not in existing)
    total = len(need)
    print(f"分钟: 已有 {len(existing)} 天, 需生成 {total} 天 × {len(stocks)} 只股票", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        fs = {pool.submit(_gen_one_file, d, stocks_int): d for d in need}
        for f in as_completed(fs):
            f.result()
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  [{done}/{total}] {fs[f]}  {time.time()-t0:.0f}s", flush=True)

    # 写 trade_dates.json（全量 + minute_by_date）
    for p in [MINUTE_META / "trade_dates.json", MINUTE_BY_DATE / "trade_dates.json"]:
        with open(p, "w") as fh:
            json.dump(all_dates, fh)

    # 写 stock_list.json
    for p in [MINUTE_META / "stock_list.json", MINUTE_BY_DATE / "stock_list.json"]:
        with open(p, "w") as fh:
            json.dump(stocks, fh)

    elapsed = time.time() - t0
    print(f"✅ 完成: {total} 个文件, {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()