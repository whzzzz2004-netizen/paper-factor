"""
将分钟线数据从按股票存储转换为按日期存储。

输入: stock_data/minute/{stock}.parquet (每只股票一个文件, 全部日期)
输出: stock_data/minute_by_date/{date}.parquet (每天一个文件, 全部股票)

用法: python scripts/convert_minute_by_date.py
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

STOCK_DATA_DIR = Path(__file__).parent.parent / "git_ignore_folder" / "factor_implementation_source_data" / "stock_data"
INPUT_DIR = STOCK_DATA_DIR / "minute"
OUTPUT_DIR = STOCK_DATA_DIR / "minute_by_date"


def main():
    # 读取股票列表和交易日
    with open(INPUT_DIR / "stock_list.json") as f:
        stocks = json.load(f)
    with open(INPUT_DIR / "trade_dates.json") as f:
        trade_dates = json.load(f)

    print(f"股票数: {len(stocks)}, 交易日: {len(trade_dates)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载所有股票数据
    print("加载数据...")
    st = time.time()
    all_data = {}
    for stock in stocks:
        all_data[stock] = pd.read_parquet(INPUT_DIR / f"{stock}.parquet")
    print(f"加载完成: {time.time()-st:.1f}s")

    # 按日期拆分并保存
    print("按日期拆分...")
    st = time.time()
    for i, td in enumerate(trade_dates):
        td_ts = pd.Timestamp(td)
        frames = []
        for stock, df in all_data.items():
            day_df = df[df.index.normalize() == td_ts]
            if not day_df.empty:
                day_df = day_df.copy()
                day_df["instrument"] = stock
                frames.append(day_df)

        if frames:
            combined = pd.concat(frames)
            combined = combined.set_index("instrument", append=True)
            combined.index = combined.index.swaplevel(0, 1)
            combined.index.names = ["instrument", "datetime"]
            combined = combined.sort_index()
            combined.to_parquet(OUTPUT_DIR / f"{td}.parquet")

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(trade_dates)} ({time.time()-st:.1f}s)")

    # 保存元信息
    with open(OUTPUT_DIR / "trade_dates.json", "w") as f:
        json.dump(trade_dates, f, indent=2)
    with open(OUTPUT_DIR / "stock_list.json", "w") as f:
        json.dump(stocks, f, indent=2)

    elapsed = time.time() - st
    print(f"完成! 耗时: {elapsed:.1f}s")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"文件数: {len(trade_dates)}")

    # 验证
    sample = pd.read_parquet(OUTPUT_DIR / f"{trade_dates[-1]}.parquet")
    print(f"验证 {trade_dates[-1]}: {sample.shape}, 索引: {sample.index.names}")


if __name__ == "__main__":
    main()
