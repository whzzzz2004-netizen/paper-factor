"""
因子计算工具库。

提供数据加载、并行计算、结果聚合等通用功能。
用户写因子时只需 import 本模块，无需关心数据格式和并行逻辑。
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning, ignore::UserWarning, ignore::FutureWarning"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# === 默认数据路径 ===
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "git_ignore_folder" / "factor_implementation_source_data" / "stock_data"


def get_stock_list(data_dir=None, freq="daily"):
    """获取可用股票列表。"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with open(data_dir / freq / "stock_list.json") as f:
        return json.load(f)


def get_trade_dates(data_dir=None, freq="daily"):
    """获取交易日列表（字符串格式 YYYY-MM-DD）。"""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    with open(data_dir / freq / "trade_dates.json") as f:
        return json.load(f)


def load_stock(stock, freq="daily", data_dir=None):
    """
    加载单只股票的 parquet 数据。

    Args:
        stock: 股票代码，如 'SH600000'
        freq: 'daily' 或 'minute'
        data_dir: 数据目录，默认为 stock_data/

    Returns:
        DataFrame，索引为 DatetimeIndex，列为该频率的字段
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = data_dir / freq / f"{stock}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"股票数据文件不存在: {path}")
    return pd.read_parquet(path)


def load_all_stocks(freq="daily", data_dir=None):
    """
    预加载所有股票数据到内存，返回 {stock_code: DataFrame}。

    适用于日线（~36MB）等小数据集，避免重复读文件。
    """
    stocks = get_stock_list(data_dir=data_dir, freq=freq)
    cache = {}
    for stock in stocks:
        cache[stock] = load_stock(stock, freq=freq, data_dir=data_dir)
    return cache


def load_minute_by_date(data_dir=None):
    """
    预加载全部分钟线数据并按日期分组，返回 {date_str: {stock: day_df}}。

    一次性读取 500 只股票（~4.3GB），之后按日期取数据为内存操作，零 I/O。
    """
    stocks = get_stock_list(data_dir=data_dir, freq="minute")
    trade_dates = get_trade_dates(data_dir=data_dir, freq="minute")

    # 预加载所有股票
    all_data = {}
    for stock in stocks:
        all_data[stock] = load_stock(stock, freq="minute", data_dir=data_dir)

    # 按日期分组
    by_date = {}
    for td in trade_dates:
        td_ts = pd.Timestamp(td)
        day_map = {}
        for stock, df in all_data.items():
            day_df = df[df.index.normalize() == td_ts]
            if not day_df.empty:
                day_map[stock] = day_df
        by_date[td] = day_map

    return by_date


def aggregate_factors(results, save_path):
    """
    将因子计算结果聚合为 (datetime × instrument) 的 DataFrame 并保存。

    Args:
        results: list of (stock, trade_date, factor_dict)
            - stock: 股票代码
            - trade_date: 交易日 (str 'YYYY-MM-DD' 或 datetime)
            - factor_dict: dict of {factor_name: value}
        save_path: 保存路径 (.parquet)

    Returns:
        DataFrame, MultiIndex (datetime, instrument), 每列一个因子
    """
    if not results:
        print("警告: 没有计算结果")
        return pd.DataFrame()

    records = []
    for stock, trade_date, fdict in results:
        if fdict is None or (isinstance(fdict, float) and np.isnan(fdict)):
            continue
        row = {"datetime": trade_date, "instrument": stock}
        if isinstance(fdict, dict):
            row.update(fdict)
        records.append(row)

    if not records:
        print("警告: 所有结果为空")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index(["datetime", "instrument"]).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path)
    print(f"已保存因子到 {save_path}, shape={df.shape}")
    return df


def run_parallel(func, tasks, n_jobs=10, desc="计算中"):
    """
    通用并行执行函数。

    Args:
        func: 可并行调用的函数
        tasks: 参数列表，每个元素是 func 的一组参数
        n_jobs: 并行数
        desc: 进度条描述

    Returns:
        结果列表
    """
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=50,
        pre_dispatch="4 * n_jobs",
    )(delayed(func)(*task) for task in tqdm(tasks, desc=desc))
    return results
