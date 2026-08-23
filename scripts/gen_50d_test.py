
#!/usr/bin/env python3
"""生成 50 天增量数据测导入速度。"""
import json, shutil, time
from pathlib import Path
DATA_ROOT = Path("/mnt/d/paper-factor-data")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WH = DATA_ROOT / "数据仓库"
MINUTE_BY_DATE = WH / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute_by_date"
MINUTE_OUT = DATA_ROOT / "新建文件/分钟线"
FUND_OUT = DATA_ROOT / "新建文件/非行情"
DAILY_TRADE_DATES = WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily" / "trade_dates.json"
DAILY_STOCK_LIST = WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily" / "stock_list.json"
FUNDAMENTAL_COLS = [
    "roe","roa","pe_ttm","pb","revenue_yoy","profit_yoy","gross_margin",
    "net_margin","debt_to_asset","ocf_per_share","market_cap",
    "circulating_market_cap","total_shares","float_shares","adjusted_profit",
    "gross_profit","total_holders","holder_change_pct",
]

with open(DAILY_TRADE_DATES) as f: all_dates = json.load(f)
with open(DAILY_STOCK_LIST) as f: stocks = json.load(f)
stocks_int = [int(s) for s in stocks]

# 新 50 天：从现有最后一天往后推 50 个交易日
last_date = pd.Timestamp(all_dates[-1])
new_dates = []
d = last_date + pd.Timedelta(days=1)
while len(new_dates) < 50:
    if d.weekday() < 5:
        new_dates.append(d.strftime("%Y%m%d"))
    d += pd.Timedelta(days=1)

print(f"新日期: {new_dates[0]} ~ {new_dates[-1]} ({len(new_dates)} 天)", flush=True)

# 1. 生成 50 个分钟 MultiIndex 格式文件（与数据仓库格式一致，导入即复制）
MINUTE_OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for i, ds in enumerate(new_dates):
    n = len(stocks_int)
    n_min = 242
    base = np.random.uniform(5, 100, n)
    rets = np.random.normal(0, 0.0003, (n_min, n))
    prices = base * np.exp(np.cumsum(rets, axis=0))
    n_total = n * n_min

    instrument = np.char.zfill(np.repeat(stocks_int, n_min).astype(str), 6)
    dt = pd.Timestamp(ds)
    datetime = np.empty(n_total, dtype="datetime64[us]")
    datetime[:] = dt.to_datetime64()

    df = pd.DataFrame({
        "open": np.repeat(prices[0], n_min),
        "high": np.repeat(prices.max(axis=0), n_min),
        "low": np.repeat(prices.min(axis=0), n_min),
        "close": prices.T.ravel(),
        "volume": np.random.randint(10000, 10000000, n_total).astype(np.int64),
        "return": rets.T.ravel(),
        "factor": np.ones(n_total, dtype=np.float64),
    }, index=pd.MultiIndex.from_arrays(
        [instrument, datetime], names=["instrument", "datetime"]
    ))

    pq.write_table(pa.Table.from_pandas(df), MINUTE_OUT / f"{ds}.parquet")
    if (i+1) % 10 == 0:
        print(f"  分钟 [{i+1}/{len(new_dates)}] {ds}", flush=True)
print(f"分钟文件: {time.time()-t0:.0f}s", flush=True)

# 2. 更新 18 个截面因子 parquet（追加新行）
t0 = time.time()
for col in FUNDAMENTAL_COLS:
    src = FUND_OUT / f"{col}.parquet"
    if src.exists():
        old = pd.read_parquet(src)
    else:
        old = pd.DataFrame()
    new_rows = pd.DataFrame(
        np.random.randn(len(new_dates), len(stocks_int)).astype(np.float64),
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in new_dates]),
        columns=stocks_int
    )
    combined = pd.concat([old, new_rows])
    pq.write_table(pa.Table.from_pandas(combined), src)
    if FUNDAMENTAL_COLS.index(col) % 5 == 0:
        print(f"  截面 [{FUNDAMENTAL_COLS.index(col)+1}/{len(FUNDAMENTAL_COLS)}] {col}", flush=True)
print(f"截面因子: {time.time()-t0:.0f}s", flush=True)

print(f"\n生成完毕，共 {len(new_dates)} 天", flush=True)