"""
分钟线因子计算模板。

用法：
    1. 在下方 calc_factors_one_day() 中写入因子计算逻辑
    2. 运行: python templates/minute_factor_template.py

你只需要关注 calc_factors_one_day 函数，其余数据加载、并行、格式全部由框架处理。
本模板参考 1min.py 的 factor() 函数设计——每只股票每天调用一次，传入当天的分钟线 DataFrame。

输入说明：
    df —— 单只股票某一天的全部分钟线数据（按时间排序），普通 DatetimeIndex
         可用列: open, high, low, close, volume, vwap
         每天 240 个 bar（9:30-11:30 + 13:00-15:00）

返回说明：
    dict，key 为因子名，value 为该股票在该交易日的因子值（float 或 np.nan）
    返回 None 表示跳过
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from factor_utils import (
    get_trade_dates,
    load_minute_by_date,
    run_parallel,
    aggregate_factors,
)

# ============================================================
# 用户配置区
# ============================================================

# 输出路径
SAVE_PATH = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "minute_factors.parquet"

# 并行数
N_JOBS = 10

# ============================================================
# 用户因子计算逻辑（核心：只写这里，参考 1min.py 的 factor() 函数）
# ============================================================


def calc_factors_one_day(df):
    """
    计算单只股票某一天的分钟线因子。

    Args:
        df: 当天的分钟线 DataFrame，DatetimeIndex，240 行
            列: open, high, low, close, volume, vwap
            已按时间排序（9:30 -> 15:00）

    Returns:
        dict: {因子名: 因子值}
    """
    if df.empty or len(df) < 10:
        return None

    close = df["close"]
    volume = df["volume"]
    vwap = df["vwap"]

    factors = {}

    # --- 示例因子 1: 成交量集中度（前5分钟占全天比例）---
    factors["volume_concentration"] = volume.nlargest(5).sum() / (volume.sum() + 1e-6)

    # --- 示例因子 2: VWAP 偏离度（尾盘 vs 全天）---
    factors["vwap_deviation"] = vwap.iloc[-5:].mean() / (vwap.mean() + 1e-6) - 1

    return factors


# ============================================================
# 框架代码（一般不需要修改）
# ============================================================


def _compute_one(stock, trade_date_str, day_df):
    """计算单只股票在单个交易日的分钟线因子（供并行调用）。"""
    try:
        result = calc_factors_one_day(day_df)
        return (stock, trade_date_str, result)
    except Exception as e:
        print(f"[ERROR] {stock} @ {trade_date_str}: {e}")
        return (stock, trade_date_str, None)


def main():
    trade_dates = get_trade_dates(freq="minute")
    print(f"共 {len(trade_dates)} 个交易日")

    # 一次性预加载全部分钟线数据并按日期分组（~4.3GB，500 次文件读取）
    print("预加载分钟线数据（约需 10~20 秒）...")
    by_date = load_minute_by_date()
    print(f"加载完成，开始计算...")

    all_results = []

    # 按日期迭代，每天并行计算所有股票（纯内存操作，零 I/O）
    for td in trade_dates:
        day_map = by_date.get(td, {})
        if not day_map:
            continue

        tasks = [(stock, td, day_df) for stock, day_df in day_map.items()]
        day_results = run_parallel(_compute_one, tasks, n_jobs=N_JOBS, desc=td)
        all_results.extend(day_results)

    # 聚合并保存
    df = aggregate_factors(all_results, SAVE_PATH)
    print(f"\n因子统计:")
    print(df.describe())
    return df


if __name__ == "__main__":
    main()
