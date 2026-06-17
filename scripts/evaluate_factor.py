#!/usr/bin/env python3
"""
对全量因子数据进行 IC/IR 和十分组收益回测。

用法:
  python scripts/evaluate_factor.py <factor.parquet> [--data-dir <dir>] [--output <json>]

输出:
  - IC mean, std, IR (Pearson)
  - Rank IC mean, std, IR (Spearman)
  - 十分组收益 (D1~D10) + 多空收益 (D1-D10)
  - 结果写入 meta.json 和 eval_results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
FULL_DATA_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"


def load_full_data_label(data_dir: Path) -> pd.DataFrame:
    """从全量日线数据中构建标签(T+1收益率)"""
    import json as _json
    stock_data_dir = data_dir / "stock_data" / "daily"
    stock_list = _json.load(open(stock_data_dir / "stock_list.json"))

    rows = []
    for stock in stock_list:
        df = pd.read_parquet(stock_data_dir / f"{stock}.parquet", columns=["close"])
        ret = df["close"].pct_change(fill_method=None).shift(-1)
        sub = pd.DataFrame({
            "datetime": df.index,
            "instrument": stock,
            "ret_next": ret.values
        }).dropna()
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    return combined.set_index(["datetime", "instrument"]).sort_index()


def compute_decile_returns(merged: pd.DataFrame) -> dict:
    """计算十分组收益"""
    dates = merged.index.get_level_values("datetime").unique().sort_values()
    decile_returns = {f"D{i}": [] for i in range(1, 11)}
    ls_returns = []

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
        for i in range(10):
            d_ret = g.loc[g["decile"] == i, "ret_next"].mean()
            decile_returns[f"D{i+1}"].append(d_ret)

        d1 = g.loc[g["decile"] == 0, "ret_next"].mean()
        d10 = g.loc[g["decile"] == 9, "ret_next"].mean()
        if not math.isnan(d1) and not math.isnan(d10):
            ls_returns.append(float(d1) - float(d10))

    result = {}
    for k, v in decile_returns.items():
        s = pd.Series(v).dropna()
        result[f"{k}_mean"] = float(s.mean()) if len(s) > 0 else None
        result[f"{k}_std"] = float(s.std(ddof=1)) if len(s) > 1 else None

    ls_series = pd.Series(ls_returns).dropna()
    if len(ls_series) > 0:
        ls_mean = float(ls_series.mean())
        ls_std = float(ls_series.std(ddof=1)) if len(ls_series) > 1 else 0.0
        result["long_short_mean"] = ls_mean
        result["long_short_sharpe"] = (ls_mean / ls_std * math.sqrt(252)) if ls_std > 0 else None
        result["long_short_days"] = int(len(ls_series))

        cum = (1.0 + ls_series).cumprod()
        roll_max = cum.cummax()
        mdd = float((cum / roll_max - 1.0).min())
        result["long_short_max_drawdown"] = mdd if not math.isnan(mdd) else None
    else:
        result["long_short_mean"] = None
        result["long_short_sharpe"] = None
        result["long_short_days"] = 0
        result["long_short_max_drawdown"] = None

    return result


def evaluate_factor(factor_df: pd.DataFrame, data_dir: Path, label_df: pd.DataFrame = None) -> dict:
    """对因子进行完整回测评估"""
    if label_df is None:
        label_df = load_full_data_label(data_dir)

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
        return {"error": "no_overlapping_data"}

    # 每天截面 IC
    dates = merged.index.get_level_values("datetime").unique().sort_values()
    daily_ic_p, daily_ic_r = [], []

    for dt in dates:
        try:
            slab = merged.xs(dt, level="datetime")
        except (KeyError, ValueError):
            continue
        if isinstance(slab, pd.Series) or len(slab) < 3:
            continue
        g = slab.dropna(subset=["factor", "ret_next"])
        if len(g) < 3:
            continue

        ic_p = g["factor"].corr(g["ret_next"])
        ic_r = g["factor"].corr(g["ret_next"], method="spearman")
        daily_ic_p.append(float(ic_p) if pd.notna(ic_p) else float("nan"))
        daily_ic_r.append(float(ic_r) if pd.notna(ic_r) else float("nan"))

    s_ic_p = pd.Series(daily_ic_p).replace([np.inf, -np.inf], np.nan).dropna()
    s_ic_r = pd.Series(daily_ic_r).replace([np.inf, -np.inf], np.nan).dropna()

    result = {}

    # IC / IR
    if len(s_ic_p) > 0:
        ic_mean = float(s_ic_p.mean())
        ic_std = float(s_ic_p.std(ddof=1)) if len(s_ic_p) > 1 else 0.0
        result["ic_mean"] = ic_mean
        result["ic_std"] = ic_std
        result["ic_ir"] = (ic_mean / ic_std) if ic_std > 0 else None
        result["ic_days"] = int(len(s_ic_p))
        result["ic_positive_ratio"] = float((s_ic_p > 0).mean())

    # Rank IC / IR
    if len(s_ic_r) > 0:
        ric_mean = float(s_ic_r.mean())
        ric_std = float(s_ic_r.std(ddof=1)) if len(s_ic_r) > 1 else 0.0
        result["rank_ic_mean"] = ric_mean
        result["rank_ic_std"] = ric_std
        result["rank_ic_ir"] = (ric_mean / ric_std) if ric_std > 0 else None
        result["rank_ic_days"] = int(len(s_ic_r))

    # 十分组收益
    decile_info = compute_decile_returns(merged)
    result.update(decile_info)

    # 基础统计
    result["n_observations"] = int(len(merged))
    result["n_dates"] = int(len(dates))

    return result


def main():
    parser = argparse.ArgumentParser(description="因子回测评估")
    parser.add_argument("factor_parquet", help="因子parquet文件路径")
    parser.add_argument("--data-dir", default=str(FULL_DATA_DIR), help="全量数据目录")
    args = parser.parse_args()

    factor_path = Path(args.factor_parquet)
    if not factor_path.exists():
        print(f"❌ 因子文件不存在: {factor_path}")
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return 1

    print(f"加载因子: {factor_path}")
    factor_df = pd.read_parquet(factor_path)
    print(f"  规模: {factor_df.shape[0]}天 × {factor_df.shape[1]}只股票")

    print(f"加载标签数据 (T+1收益率)...")
    label_df = load_full_data_label(data_dir)
    print(f"  标签: {len(label_df)} 行")

    print(f"计算评估指标...")
    result = evaluate_factor(factor_df, data_dir, label_df)

    if "error" in result:
        print(f"❌ 评估失败: {result['error']}")
        return 1

    # 打印结果
    print(f"\n{'='*50}")
    print(f"因子评估结果")
    print(f"{'='*50}")
    ic_mean = result.get("ic_mean", float("nan"))
    print(f"IC (Pearson):   均值={ic_mean:.6f}, IR={result.get('ic_ir', 'N/A')}")
    ric_mean = result.get("rank_ic_mean", float("nan"))
    print(f"Rank IC:        均值={ric_mean:.6f}, IR={result.get('rank_ic_ir', 'N/A')}")
    print(f"IC 正天数占比:   {result.get('ic_positive_ratio', 'N/A')}")
    print(f"\n十分组收益 (均值):")
    for i in range(1, 11):
        key = f"D{i}_mean"
        val = result.get(key, float("nan"))
        print(f"  D{i:2d}: {val:.6%}" if isinstance(val, float) else f"  D{i:2d}: {val}")
    ls_mean = result.get("long_short_mean", None)
    print(f"\n多空收益 (D1-D10):")
    print(f"  日均值:   {ls_mean:.6%}" if isinstance(ls_mean, float) else f"  日均值:   N/A")
    print(f"  Sharpe:   {result.get('long_short_sharpe', 'N/A')}")
    print(f"  Max DD:   {result.get('long_short_max_drawdown', 'N/A')}")
    print(f"  天数:     {result.get('long_short_days', 0)}")
    print(f"{'='*50}")

    # 保存结果到meta.json
    meta_path = factor_path.parent / f"{factor_path.stem}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["evaluation"] = result
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"已更新: {meta_path}")
    else:
        # 没有meta.json就写一个临时的
        output_path = factor_path.parent / f"{factor_path.stem}.eval.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
