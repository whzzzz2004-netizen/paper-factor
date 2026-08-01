#!/usr/bin/env python3
"""
缩减测试数据到 300 天。

对 git_ignore_folder/factor_implementation_source_data_1000/ 执行：
- 日线 (stock_data/daily): trade_dates.json 截取最后 300 天；每只股票 parquet 按日期过滤
- 分钟 (stock_data/minute_by_date): trade_dates.json 截取最后 300 天；删除不在 300 天内的 parquet 文件

stock_list.json 和 industry.json 保持不变。
"""

import json
import shutil
from pathlib import Path

import pandas as pd

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "git_ignore_folder" / "factor_implementation_source_data_1000"
DAILY_DIR = TEST_DATA_DIR / "stock_data" / "daily"
MINUTE_DIR = TEST_DATA_DIR / "stock_data" / "minute_by_date"

N_DAYS = 300


def truncate_daily():
    """截取日线数据最后 N_DAYS 天。"""
    dates_path = DAILY_DIR / "trade_dates.json"
    dates = json.loads(dates_path.read_text(encoding="utf-8"))
    assert len(dates) > N_DAYS, f"日线交易日 {len(dates)} <= {N_DAYS}，不需要截取"
    truncated = dates[-N_DAYS:]
    keep_set = set(truncated)

    # 写回截断后的 trade_dates.json
    dates_path.write_text(json.dumps(truncated, ensure_ascii=False), encoding="utf-8")
    print(f"日线 trade_dates.json: {len(dates)} -> {len(truncated)} 天")

    # 过滤每只股票 parquet
    stock_files = sorted(DAILY_DIR.glob("*.parquet"))
    filtered = 0
    for fpath in stock_files:
        df = pd.read_parquet(fpath)
        # datetime 列或 index
        if "datetime" in df.columns:
            df = df[df["datetime"].isin(keep_set)]
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df[df.index.strftime("%Y-%m-%d").isin(keep_set)]
        elif df.index.name == "datetime":
            df = df[df.index.strftime("%Y-%m-%d").isin(keep_set)]
        else:
            print(f"  警告: {fpath.name} 无法识别日期列，跳过过滤")
            continue
        df.to_parquet(fpath)
        filtered += 1

    print(f"日线 parquet: 过滤 {filtered} 个文件")


def truncate_minute():
    """截取分钟数据最后 N_DAYS 天。"""
    dates_path = MINUTE_DIR / "trade_dates.json"
    dates = json.loads(dates_path.read_text(encoding="utf-8"))
    assert len(dates) > N_DAYS, f"分钟交易日 {len(dates)} <= {N_DAYS}，不需要截取"
    truncated = dates[-N_DAYS:]
    keep_set = set(truncated)

    # 写回截断后的 trade_dates.json
    dates_path.write_text(json.dumps(truncated, ensure_ascii=False), encoding="utf-8")
    print(f"分钟 trade_dates.json: {len(dates)} -> {len(truncated)} 天")

    # 删除不在 300 天内的 parquet 文件
    kept = 0
    deleted = 0
    for fpath in sorted(MINUTE_DIR.glob("*.parquet")):
        fname = fpath.name
        # 跳过非日期文件
        if fname == "market_5min_returns.parquet":
            kept += 1
            continue
        date_str = fname.removesuffix(".parquet")
        if date_str in keep_set:
            kept += 1
        else:
            fpath.unlink()
            deleted += 1

    print(f"分钟 parquet: 保留 {kept} 个, 删除 {deleted} 个")


def main():
    print(f"测试数据目录: {TEST_DATA_DIR}")
    print(f"目标天数: {N_DAYS}")

    truncate_daily()
    truncate_minute()

    # 验证
    daily_dates = json.loads((DAILY_DIR / "trade_dates.json").read_text())
    minute_dates = json.loads((MINUTE_DIR / "trade_dates.json").read_text())
    print(f"\n验证:")
    print(f"  日线交易天数: {len(daily_dates)} (期望 {N_DAYS})")
    print(f"  分钟交易天数: {len(minute_dates)} (期望 {N_DAYS})")
    print(f"  股票数量: {len(json.loads((DAILY_DIR / 'stock_list.json').read_text()))}")

    daily_parquets = len(list(DAILY_DIR.glob("*.parquet")))
    minute_parquets = len(list(MINUTE_DIR.glob("*.parquet")))
    print(f"  日线 parquet 文件数: {daily_parquets}")
    print(f"  分钟 parquet 文件数: {minute_parquets}")

    # 检查各股 parquet 行数
    sample = pd.read_parquet(next(DAILY_DIR.glob("*.parquet")))
    print(f"  日线样本行数: {len(sample)} (期望 <={N_DAYS})")

    print("\n完成!")


if __name__ == "__main__":
    main()