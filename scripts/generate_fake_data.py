#!/usr/bin/env python3
"""生成假分钟数据和基本面数据，补全 2018~2026 的全量历史。

用法: python3 scripts/generate_fake_data.py
输出: 原始数据/分钟线/YYYYMMDD.parquet + 原始数据/非行情/{因子名}.parquet
"""

import json
import random
import shutil
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

# 日线元数据（已导入）
DAILY_TRADE_DATES = WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily" / "trade_dates.json"
DAILY_STOCK_LIST = WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily" / "stock_list.json"

# 分钟目录（已有文件）
MINUTE_BY_DATE = WH / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute_by_date"
MINUTE_OUT = DATA_ROOT / "原始数据" / "分钟线"

# 非行情输出
FUND_OUT = DATA_ROOT / "原始数据" / "非行情"

FUNDAMENTAL_COLS = [
    "roe", "roa", "pe_ttm", "pb", "revenue_yoy", "profit_yoy", "gross_margin",
    "net_margin", "debt_to_asset", "ocf_per_share", "market_cap",
    "circulating_market_cap", "total_shares", "float_shares", "adjusted_profit",
    "gross_profit", "total_holders", "holder_change_pct",
]

# 每个因子的大致 realistic range（用于生成不奇怪的值）
FUND_RANGES = {
    "roe": (-30, 40), "roa": (-20, 25), "pe_ttm": (0, 200), "pb": (0, 20),
    "revenue_yoy": (-50, 100), "profit_yoy": (-80, 150), "gross_margin": (0, 90),
    "net_margin": (-20, 50), "debt_to_asset": (0, 90), "ocf_per_share": (-5, 10),
    "market_cap": (1, 50000), "circulating_market_cap": (0.5, 30000),
    "total_shares": (1000, 500000), "float_shares": (500, 300000),
    "adjusted_profit": (-1e9, 1e10), "gross_profit": (-1e8, 1e10),
    "total_holders": (1000, 500000), "holder_change_pct": (-30, 30),
}


def _load_json_list(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def _gen_minute_file(date_str: str, stocks: list) -> Path:
    """生成一个分钟 flat-format parquet 文件。"""
    n_stocks = len(stocks)
    n_minutes = 242

    # 生成随机walk价格
    base_prices = np.random.uniform(5, 100, n_stocks)  # 每只股票开盘基准价
    returns = np.random.normal(0, 0.0003, (n_minutes, n_stocks))  # 分钟收益率
    prices = base_prices * np.exp(np.cumsum(returns, axis=0))  # 分钟价格序列

    open_p = prices[0, :]
    close_p = prices[-1, :]
    high_p = np.max(prices, axis=0)
    low_p = np.min(prices, axis=0)

    # 膨胀到长格式 (每只股票242行)
    stock_ids = np.char.zfill(np.repeat(stocks, n_minutes).astype(str), 6)
    minutes = np.tile(np.arange(n_minutes), n_stocks)

    # 每只股票的价格序列
    open_vals = np.repeat(open_p, n_minutes)
    close_vals = prices.T.ravel()  # stock-major, minute-minor, 刚好对应 np.repeat + np.tile
    high_vals = np.repeat(high_p, n_minutes)
    low_vals = np.repeat(low_p, n_minutes)
    volume_vals = np.random.randint(10000, 10000000, n_stocks * n_minutes).astype(np.int64)
    factor_vals = np.ones(n_stocks * n_minutes, dtype=np.float64)
    return_vals = returns.T.ravel()

    # 构造 DataFrame
    df = pd.DataFrame({
        "symbol": stock_ids,
        "trade_date": pd.Timestamp(date_str),
        "open": open_vals,
        "high": high_vals,
        "low": low_vals,
        "close": close_vals,
        "volume": volume_vals,
        "return": return_vals,
        "factor": factor_vals,
    })

    # 写入 parquet
    dst = MINUTE_OUT / f"{date_str}.parquet"
    pq.write_table(pa.Table.from_pandas(df), dst)
    return dst


def generate_minute_data(dates: list, stocks: list, max_workers: int = 16):
    """并行生成所有分钟的 flat-format parquet。"""
    MINUTE_OUT.mkdir(parents=True, exist_ok=True)

    # 已有文件
    existing = {f.stem for f in MINUTE_BY_DATE.glob("2*.parquet")}
    need = [d for d in dates if d not in existing]
    total = len(need)
    print(f"分钟数据: 已有 {len(existing)} 天, 需生成 {total} 天, {len(stocks)} 只股票", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fs = {pool.submit(_gen_minute_file, d, stocks): d for d in need}
        for f in as_completed(fs):
            d = fs[f]
            f.result()
            done += 1
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total}] {d} 完成, {rate:.1f} 文件/秒", flush=True)

    elapsed = time.time() - t0
    print(f"分钟数据生成完成: {total} 个文件, {elapsed:.0f}s", flush=True)


def generate_fundamental_data(dates: list, stocks: list):
    """生成18个基本面截面因子 parquet。"""
    FUND_OUT.mkdir(parents=True, exist_ok=True)
    n_dates = len(dates)
    n_stocks = len(stocks)

    # 生成新因子描述.csv
    lines = ["因子名,因子描述"]
    for col in FUNDAMENTAL_COLS:
        lines.append(f"{col},{col}因子描述 — 合成数据")
    (FUND_OUT / "新因子描述.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    t0 = time.time()
    for i, col in enumerate(FUNDAMENTAL_COLS):
        lo, hi = FUND_RANGES.get(col, (-1, 1))
        # 带时间序列自相关的随机数据
        data = np.random.normal(0, 1, (n_dates, n_stocks)).astype(np.float64)
        # 映射到 realistic range
        data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        data = data * (hi - lo) + lo
        # 部分列不能为负
        if col in ("pe_ttm", "pb", "gross_margin", "debt_to_asset", "market_cap",
                   "circulating_market_cap", "total_shares", "float_shares", "total_holders"):
            data = np.abs(data)

        df = pd.DataFrame(data, index=pd.DatetimeIndex(dates), columns=stocks)
        dst = FUND_OUT / f"{col}.parquet"
        pq.write_table(pa.Table.from_pandas(df), dst)
        print(f"  [{i+1}/{len(FUNDAMENTAL_COLS)}] {col}: {n_dates}×{n_stocks} → {dst.name}", flush=True)

    elapsed = time.time() - t0
    print(f"基本面数据生成完成: {len(FUNDAMENTAL_COLS)} 个因子, {elapsed:.0f}s", flush=True)


def main():
    # 加载日线元数据
    dates = _load_json_list(DAILY_TRADE_DATES)
    stocks = sorted(_load_json_list(DAILY_STOCK_LIST))
    stocks_int = [int(s) for s in stocks]

    print(f"日线共 {len(dates)} 天, {len(stocks)} 只股票", flush=True)
    print(f"日期范围: {dates[0]} ~ {dates[-1]}", flush=True)

    generate_minute_data(dates, stocks_int)
    generate_fundamental_data(dates, stocks_int)

    print("\n✅ 生成完毕，可以运行 python3 scripts/import_new_data.py 导入", flush=True)


if __name__ == "__main__":
    main()