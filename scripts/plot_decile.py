#!/usr/bin/env python3
"""
生成因子十分组收益图。

用法:
  python scripts/plot_decile.py <factor.parquet> [--data-dir <dir>] [--output <path>]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from evaluate_factor import load_full_data_label, compute_decile_returns
except ImportError:
    from scripts.evaluate_factor import load_full_data_label, compute_decile_returns

PROJECT_ROOT = Path(__file__).parent.parent
FULL_DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data")))


def plot_decile_returns(factor_df: pd.DataFrame, label_df: pd.DataFrame, factor_name: str, output_path: str):
    """生成十分组累计收益图"""
    # 合并因子和标签
    if factor_df.index.name == "Date" and factor_df.columns.name == "Code":
        factor_long = factor_df.stack()
        factor_long.index.names = ["datetime", "instrument"]
        factor_series = factor_long
    else:
        factor_series = factor_df.iloc[:, 0]
    factor_series.name = "factor"
    factor_series = pd.to_numeric(factor_series, errors="coerce").dropna()

    merged = factor_series.to_frame().join(label_df, how="inner").dropna()
    if merged.empty:
        print("无重叠数据，无法生成图表")
        return False

    # 逐日计算十分组
    dates = merged.index.get_level_values("datetime").unique().sort_values()
    decile_daily = {i: [] for i in range(10)}
    date_list = []

    for dt in dates:
        try:
            slab = merged.xs(dt, level="datetime")
        except (KeyError, ValueError):
            continue
        if isinstance(slab, pd.Series) or len(slab) < 10:
            continue
        g = slab.dropna(subset=["factor", "ret_next"]).copy()
        if len(g) < 10:
            continue

        g["decile"] = pd.qcut(g["factor"].rank(method="first"), 10, labels=False)
        date_list.append(dt)
        for i in range(10):
            d_ret = g.loc[g["decile"] == i, "ret_next"].mean()
            decile_daily[i].append(d_ret)

    if len(date_list) == 0:
        print("无有效日期数据，无法生成图表")
        return False

    # 计算累计收益
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.RdYlGn(np.linspace(0, 1, 10))

    # X轴：将交易天数索引替换为实际日期
    x_dates = pd.to_datetime(date_list)

    # 左图：累计收益（累加而非累乘）
    for i in range(10):
        cum = pd.Series(decile_daily[i]).cumsum()
        ax1.plot(x_dates, cum, label=f"D{i+1}", color=colors[i], linewidth=1.5)

    ax1.set_title(f"{factor_name}\nDecile Cumulative Returns", fontsize=11)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax1.tick_params(axis="x", rotation=30)

    # 右图：多空收益（累加而非累乘）
    ls_daily = [decile_daily[0][j] - decile_daily[9][j] for j in range(len(date_list))]
    ls_cum = pd.Series(ls_daily).cumsum()
    ax2.plot(x_dates, ls_cum, color="darkblue", linewidth=2, label="Long-Short (D1-D10)")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title(f"{factor_name}\nLong-Short Cumulative Return", fontsize=11)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="生成因子十分组收益图")
    parser.add_argument("factor_parquet", help="因子parquet文件路径")
    parser.add_argument("--data-dir", default=str(FULL_DATA_DIR), help="全量数据目录")
    parser.add_argument("--output", "-o", default=None, help="输出图片路径")
    args = parser.parse_args()

    factor_path = Path(args.factor_parquet)
    if not factor_path.exists():
        print(f"❌ 因子文件不存在: {factor_path}")
        return 1

    factor_name = factor_path.stem
    data_dir = Path(args.data_dir)

    factor_df = pd.read_parquet(factor_path)
    print(f"加载标签数据...")
    label_df = load_full_data_label(data_dir)

    output_path = args.output or str(factor_path.parent / f"{factor_name}.decile.png")
    plot_decile_returns(factor_df, label_df, factor_name, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
