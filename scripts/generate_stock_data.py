"""
生成模拟 A 股数据：500 只股票，最近 4 年，按股票存储为 parquet。

日线字段: open, close, high, low, volume, factor, market_cap, industry_sw, turnover
分钟线字段: open, high, low, close, volume, vwap

用法: python scripts/generate_stock_data.py
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent.parent / "git_ignore_folder" / "factor_implementation_source_data" / "stock_data"

# ============================================================
# 配置
# ============================================================

NUM_STOCKS = 500
DAILY_YEARS = 4        # 日线和分钟线都是 4 年

# 申万一级行业（31 个）
SW_INDUSTRIES = [
    "SW110000", "SW210000", "SW220000", "SW230000", "SW240000",
    "SW270000", "SW280000", "SW330000", "SW340000", "SW350000",
    "SW360000", "SW370000", "SW410000", "SW420000", "SW430000",
    "SW440000", "SW450000", "SW460000", "SW480000", "SW490000",
    "SW510000", "SW610000", "SW620000", "SW630000", "SW640000",
    "SW650000", "SW710000", "SW720000", "SW730000", "SW740000",
    "SW750000",
]


def generate_stock_codes(n):
    """生成 n 个股票代码：SH600xxx + SZ000xxx + SZ001xxx + SZ002xxx + SZ300xxx。"""
    codes = []
    # SH600xxx: 约 200 只
    for i in range(min(200, n)):
        codes.append(f"SH60{i:04d}")
    # SZ000xxx: 约 100 只
    for i in range(min(100, max(0, n - 200))):
        codes.append(f"SZ00{i:04d}")
    # SZ002xxx: 约 100 只
    for i in range(min(100, max(0, n - 300))):
        codes.append(f"SZ002{i:03d}")
    # SZ300xxx: 约 100 只
    for i in range(min(100, max(0, n - 400))):
        codes.append(f"SZ300{i:03d}")
    # 补齐
    while len(codes) < n:
        codes.append(f"SH68{len(codes):04d}")
    return codes[:n]


def generate_trade_dates(start_date, end_date):
    """生成交易日序列（排除周末和部分节假日）。"""
    all_dates = pd.bdate_range(start=start_date, end=end_date, freq="B")
    # 简单排除一些常见假期（春节、国庆各约 7 天）
    holidays = []
    for year in range(all_dates[0].year, all_dates[-1].year + 1):
        # 春节（约 1 月底 ~ 2 月初，取 7 个连续工作日）
        spring_start = pd.Timestamp(f"{year}-01-28")
        for d in range(7):
            holidays.append(spring_start + pd.Timedelta(days=d))
        # 国庆（10/1 ~ 10/7）
        for d in range(7):
            holidays.append(pd.Timestamp(f"{year}-10-01") + pd.Timedelta(days=d))
    holidays = pd.DatetimeIndex(holidays)
    trade_dates = all_dates.difference(holidays)
    return trade_dates


def generate_minute_timestamps(trade_date):
    """生成一个交易日的分钟时间戳（9:30-11:30 + 13:00-15:00，共 240 个）。"""
    morning = pd.date_range(
        start=f"{trade_date} 09:30", end=f"{trade_date} 11:29", freq="min"
    )
    afternoon = pd.date_range(
        start=f"{trade_date} 13:00", end=f"{trade_date} 14:59", freq="min"
    )
    return morning.append(afternoon)


def generate_daily_data(codes, trade_dates):
    """生成日线数据，返回 {stock_code: DataFrame}。"""
    n_days = len(trade_dates)
    stock_data = {}

    for idx, code in enumerate(codes):
        # 随机初始价格（10 ~ 200）
        init_price = np.random.uniform(10, 200)
        # 日收益率：均值 0.0002，标准差 0.025
        daily_returns = np.random.normal(0.0002, 0.025, n_days)
        # 加一点趋势和均值回复
        trend = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 0.001
        daily_returns += trend

        # 收盘价序列
        close = init_price * np.cumprod(1 + daily_returns)
        # 开盘 = 前收盘 * (1 + 小幅跳空)
        gap = np.random.normal(0, 0.005, n_days)
        open_price = np.roll(close, 1) * (1 + gap)
        open_price[0] = close[0] * (1 + gap[0])
        # 最高、最低
        intraday_range = np.abs(np.random.normal(0, 0.015, n_days))
        high = np.maximum(open_price, close) * (1 + intraday_range)
        low = np.minimum(open_price, close) * (1 - intraday_range)

        # 成交量：基础量 + 随机波动，量价相关
        base_volume = np.random.uniform(5e6, 5e8)
        vol_noise = np.exp(np.random.normal(0, 0.5, n_days))
        price_change = np.abs(daily_returns)
        volume = base_volume * vol_noise * (1 + price_change * 10)
        volume = volume.astype(int)

        # 复权因子：大部分为 1，少数时间有变化（模拟分红）
        factor = np.ones(n_days)
        # 每年约 1-2 次分红，每次 factor 调整约 0.95~1.0
        dividend_days = np.random.choice(n_days, size=min(6, n_days // 100), replace=False)
        for d in sorted(dividend_days):
            factor[d:] *= np.random.uniform(0.97, 1.0)
        # 确保最后 factor 不低于 0.5
        if factor[-1] < 0.5:
            factor = factor / factor[-1] * 0.8

        # 总市值：价格 * 随机股本
        shares = np.random.uniform(1e8, 5e10)  # 流通股本
        market_cap = close * shares

        # 换手率：volume / 流通股本
        turnover = volume / shares

        # 行业
        industry = SW_INDUSTRIES[idx % len(SW_INDUSTRIES)]

        df = pd.DataFrame({
            "open": np.round(open_price, 2),
            "close": np.round(close, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "volume": volume,
            "factor": np.round(factor, 6),
            "market_cap": np.round(market_cap, 0),
            "industry_sw": industry,
            "turnover": np.round(turnover, 6),
        }, index=trade_dates)
        df.index.name = "datetime"

        stock_data[code] = df

    return stock_data


def generate_minute_data(daily_stock_data, trade_dates):
    """基于日线数据生成分钟线数据，返回 {stock_code: DataFrame}。"""
    minute_data = {}

    for code, daily_df in daily_stock_data.items():
        all_minutes = []
        n_days = len(trade_dates)

        for i, date in enumerate(trade_dates):
            day_open = daily_df.loc[date, "open"]
            day_close = daily_df.loc[date, "close"]
            day_high = daily_df.loc[date, "high"]
            day_low = daily_df.loc[date, "low"]
            day_volume = daily_df.loc[date, "volume"]

            timestamps = generate_minute_timestamps(date)
            n_bars = len(timestamps)  # 240

            # 生成分钟级价格路径（GBM + 均值回复到日收盘）
            returns = np.random.normal(0, 0.003, n_bars)
            # 向日收盘方向漂移
            drift = (day_close / day_open - 1) / n_bars
            returns += drift

            minute_close = day_open * np.cumprod(1 + returns)
            # 缩放到日线的 high/low 范围
            actual_max = minute_close.max()
            actual_min = minute_close.min()
            if actual_max > actual_min:
                minute_close = day_low + (minute_close - actual_min) / (actual_max - actual_min) * (day_high - day_low)
            else:
                minute_close = np.full(n_bars, day_open)

            # 开盘价
            minute_open = np.roll(minute_close, 1)
            minute_open[0] = day_open
            # 高低
            bar_range = np.abs(np.random.normal(0, 0.002, n_bars))
            minute_high = np.maximum(minute_open, minute_close) * (1 + bar_range)
            minute_low = np.minimum(minute_open, minute_close) * (1 - bar_range)

            # 成交量：U 型分布（开盘和收盘量大，中午量小）
            u_shape = np.concatenate([
                np.linspace(3, 1, 60),    # 上午前半：高→低
                np.linspace(1, 0.8, 60),  # 上午后半：低→更低
                np.linspace(0.8, 1, 60),  # 下午前半：低→中
                np.linspace(1, 2.5, 60),  # 下午后半：中→高
            ])
            u_shape = u_shape / u_shape.mean()
            minute_volume_noise = np.exp(np.random.normal(0, 0.3, n_bars))
            minute_volume = day_volume * u_shape * minute_volume_noise / n_bars
            minute_volume = np.maximum(minute_volume, 100).astype(int)

            # VWAP: 用价格和成交量加权
            cumvol = np.cumsum(minute_volume)
            cumvp = np.cumsum(minute_close * minute_volume)
            vwap = cumvp / np.maximum(cumvol, 1)

            minute_df = pd.DataFrame({
                "open": np.round(minute_open, 2),
                "high": np.round(minute_high, 2),
                "low": np.round(minute_low, 2),
                "close": np.round(minute_close, 2),
                "volume": minute_volume,
                "vwap": np.round(vwap, 2),
            }, index=timestamps)
            minute_df.index.name = "datetime"
            all_minutes.append(minute_df)

        minute_data[code] = pd.concat(all_minutes)

    return minute_data


def save_data(stock_data, freq, trade_dates):
    """保存为按股票分割的 parquet 文件 + HDF5 大表（供 rdagent 流水线使用）。"""
    out_dir = OUTPUT_DIR / freq
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = sorted(stock_data.keys())
    for code in codes:
        stock_data[code].to_parquet(out_dir / f"{code}.parquet")

    date_strs = sorted(trade_dates.strftime("%Y-%m-%d").tolist())
    with open(out_dir / "stock_list.json", "w") as f:
        json.dump(codes, f, indent=2)
    with open(out_dir / "trade_dates.json", "w") as f:
        json.dump(date_strs, f, indent=2)

    print(f"已保存 {len(codes)} 只股票到 {out_dir}")
    print(f"  时间范围: {date_strs[0]} ~ {date_strs[-1]} ({len(date_strs)} 个交易日)")
    sample = stock_data[codes[0]]
    print(f"  每只股票: {len(sample)} 行, 列={list(sample.columns)}")


def main():
    print(f"生成 {NUM_STOCKS} 只股票的模拟数据...")
    print()

    # 1. 生成股票代码和交易日
    codes = generate_stock_codes(NUM_STOCKS)
    end_date = "2025-05-23"
    start_date = pd.Timestamp(end_date) - pd.DateOffset(years=DAILY_YEARS)
    trade_dates = generate_trade_dates(start_date, end_date)
    print(f"股票代码: {codes[0]} ~ {codes[-1]} ({len(codes)} 只)")
    print(f"交易日: {trade_dates[0].strftime('%Y-%m-%d')} ~ {trade_dates[-1].strftime('%Y-%m-%d')} ({len(trade_dates)} 天)")
    print()

    # 2. 生成日线数据
    print("生成日线数据...")
    daily_data = generate_daily_data(codes, trade_dates)
    save_data(daily_data, "daily", trade_dates)
    print()

    # 3. 生成分钟线数据（与日线同一时间范围）
    print("生成分钟线数据...")
    minute_data = generate_minute_data(daily_data, trade_dates)
    save_data(minute_data, "minute", trade_dates)
    print()

    # 4. 验证
    print("验证...")
    sample_code = codes[0]
    d = pd.read_parquet(OUTPUT_DIR / "daily" / f"{sample_code}.parquet")
    m = pd.read_parquet(OUTPUT_DIR / "minute" / f"{sample_code}.parquet")
    print(f"  日线 {sample_code}: {d.shape}, 列={list(d.columns)}")
    print(f"  分钟线 {sample_code}: {m.shape}, 列={list(m.columns)}")
    print(f"  分钟线每天: {len(m) // len(trade_dates)} bars")
    print()
    print("完成！")


if __name__ == "__main__":
    main()
