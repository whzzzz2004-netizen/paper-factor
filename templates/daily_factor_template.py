"""
日线因子计算模板。

用法：
    1. 在下方 calc_factor_single_stock() 中写入因子计算逻辑
    2. 运行: python templates/daily_factor_template.py

你只需要关注 calc_factor_single_stock 函数，其余数据加载、并行、格式全部由框架处理。

输入说明：
    df —— 单只股票从上市到 trade_date 的全部日线数据（已按日期排序），普通 DatetimeIndex
         可用列: open, close, high, low, volume, factor(复权因子), market_cap, industry_sw, turnover
    trade_date —— 当前要计算因子的交易日 (pd.Timestamp)

返回说明：
    dict，key 为因子名，value 为该股票在该交易日的因子值（float 或 np.nan）
    返回 None 表示跳过该股票-日期组合
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from factor_utils import (
    get_stock_list,
    get_trade_dates,
    load_all_stocks,
    run_parallel,
    aggregate_factors,
)

# ============================================================
# 用户配置区
# ============================================================

# 输出路径
SAVE_PATH = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "daily_factors.parquet"

# 并行数
N_JOBS = 10

# ============================================================
# 用户因子计算逻辑（核心：只写这里）
# ============================================================


def calc_factor_single_stock(df, trade_date):
    """
    计算单只股票在指定日期的因子值。

    Args:
        df: 单只股票的全部历史日线数据（截至 trade_date），DatetimeIndex
            列: open, close, high, low, volume, factor, market_cap, industry_sw, turnover
        trade_date: 目标交易日 (pd.Timestamp)

    Returns:
        dict: {因子名: 因子值}，或 None 表示跳过
    """
    # --- 在下方编写你的因子逻辑 ---
    if len(df) < 6:
        return None

    close = df["close"]

    momentum_5d = close.iloc[-1] / close.iloc[-6] - 1 if close.iloc[-6] != 0 else np.nan
    reversal_5d = -momentum_5d

    return {
        "momentum_5d": momentum_5d,
        "reversal_5d": reversal_5d,
    }


# ============================================================
# 框架代码（一般不需要修改）
# ============================================================


def _compute_one_stock(stock, trade_date_str, stock_cache):
    """计算单只股票在单个交易日的因子（供并行调用）。"""
    trade_date = pd.Timestamp(trade_date_str)
    try:
        df = stock_cache[stock]
        df = df[df.index <= trade_date]
        if df.empty:
            return (stock, trade_date_str, None)
        result = calc_factor_single_stock(df, trade_date)
        return (stock, trade_date_str, result)
    except Exception as e:
        print(f"[ERROR] {stock} @ {trade_date_str}: {e}")
        return (stock, trade_date_str, None)


def main():
    stocks = get_stock_list(freq="daily")
    trade_dates = get_trade_dates(freq="daily")

    # 预加载所有股票数据到内存（日线 ~36MB，一次性读完避免 50 万次文件 I/O）
    print("预加载日线数据...")
    stock_cache = load_all_stocks(freq="daily")
    print(f"已加载 {len(stock_cache)} 只股票")

    # 构建任务列表：(stock, trade_date, cache)
    tasks = [(stock, td, stock_cache) for td in trade_dates for stock in stocks]
    print(f"共 {len(stocks)} 只股票, {len(trade_dates)} 个交易日, {len(tasks)} 个任务")

    # 并行计算
    results = run_parallel(_compute_one_stock, tasks, n_jobs=N_JOBS, desc="计算日线因子")

    # 聚合并保存
    df = aggregate_factors(results, SAVE_PATH)
    print(f"\n因子统计:")
    print(df.describe())
    return df


if __name__ == "__main__":
    main()
