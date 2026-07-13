#!/usr/bin/env python3
"""
Barra 暴露分析 — 对因子多空组合收益做 Barra 风格因子归因。

用法:
  python scripts/barra_evaluate.py <factor.parquet> [--data-dir <dir>] [--barra-dir <dir>]

输出（写入 meta.json）:
  - alpha: Barra 无法解释的超额收益
  - exposures: 各风格因子的暴露系数 (β) + t 统计量 + p 值
  - r_squared / adj_r_squared: Barra 因子对多空收益的解释程度
  - n_days: 回归使用的有效天数
"""

import argparse
import json
import sys
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).parent.parent
FULL_DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data")))
DEFAULT_BARRA_DIR = Path(os.environ.get("PAPER_FACTOR_BARRA_DIR", str(PROJECT_ROOT / "git_ignore_folder" / "barra_model")))

# Trading Model 的 20 个风格因子（不含行业哑变量）
STYLE_FACTORS = [
    "BETA", "MOMENTUM", "SIZE", "EARNYILD", "RESVOL",
    "GROWTH", "BTOP", "LEVERAGE", "LIQUIDTY", "MIDCAP",
    "DIVYILD", "EARNQLTY", "EARNVAR", "INVSQLTY", "LTREVRSL",
    "PROFIT", "ANALSENTI", "INDMOM", "SEASON", "STREVRSL",
]


def compute_long_short_returns(factor_df: pd.DataFrame, data_dir: Path) -> pd.Series:
    """计算因子多空组合日收益率序列（D1 均值 - D10 均值）。

    Args:
        factor_df: Date × Code 因子矩阵。
        data_dir: 全量数据目录（含日线 parquet）。

    Returns:
        按日期排序的多空收益 Series（index=datetime, value=return）。
    """
    # 构建 T+1 收益率标签
    stock_data_dir = data_dir / "stock_data" / "daily"
    stock_list = json.load(open(stock_data_dir / "stock_list.json"))

    label_rows = []
    for stock in stock_list:
        try:
            df = pd.read_parquet(stock_data_dir / f"{stock}.parquet", columns=["close"])
        except Exception:
            continue
        ret = df["close"].pct_change(fill_method=None).shift(-1)
        sub = pd.DataFrame({
            "datetime": df.index,
            "instrument": stock,
            "ret_next": ret.values,
        }).dropna()
        label_rows.append(sub)
    if not label_rows:
        return pd.Series(dtype=float)
    label_df = pd.concat(label_rows, ignore_index=True)
    label_df["datetime"] = pd.to_datetime(label_df["datetime"])
    label_df = label_df.set_index(["datetime", "instrument"]).sort_index()

    # 将因子矩阵转为长表
    if factor_df.index.name == "Date" and factor_df.columns.name == "Code":
        factor_long = factor_df.stack()
        factor_long.index.names = ["datetime", "instrument"]
    else:
        factor_long = factor_df.iloc[:, 0]
    factor_long.name = "factor"
    factor_long = pd.to_numeric(factor_long, errors="coerce").dropna()

    merged = factor_long.to_frame().join(label_df, how="inner").dropna()
    if merged.empty:
        return pd.Series(dtype=float)

    # 每天分 10 组，算 D1 - D10 的多空收益
    dates = merged.index.get_level_values("datetime").unique().sort_values()
    ls_returns = []
    ls_dates = []

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
        d1 = g.loc[g["decile"] == 0, "ret_next"].mean()
        d10 = g.loc[g["decile"] == 9, "ret_next"].mean()
        if not math.isnan(d1) and not math.isnan(d10):
            ls_returns.append(float(d1) - float(d10))
            ls_dates.append(dt)

    return pd.Series(ls_returns, index=pd.to_datetime(ls_dates)).sort_index()


def load_barra_factor_returns(barra_dir: Path, model: str = "Trading Model") -> pd.DataFrame:
    """加载 Barra 风格因子日收益率。

    Args:
        barra_dir: Barra 数据目录。
        model: "Trading Model" 或 "Long-Term Model"。

    Returns:
        DataFrame: index=datetime, columns=STYLE_FACTORS（20 个风格因子）。
    """
    csv_path = barra_dir / f"因子收益率表({model}).csv"
    if not csv_path.exists():
        print(f"  ⚠️ Barra 因子收益率表不存在: {csv_path}", flush=True)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["tradeDate"] = pd.to_datetime(df["tradeDate"], format="%Y%m%d")
    df = df.set_index("tradeDate")

    # 只保留 20 个风格因子列，去掉行业、COUNTRY、updateTime
    available = [c for c in STYLE_FACTORS if c in df.columns]
    missing = [c for c in STYLE_FACTORS if c not in df.columns]
    if missing:
        print(f"  ⚠️ Barra 数据缺少风格因子: {missing}", flush=True)
    return df[available].dropna()


def ols_regression(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """手动 OLS 回归（兼容无 statsmodels 环境）。

    Returns:
        dict: {alpha/exposure名: {coef, tstat, pvalue}, r_squared, adj_r_squared, n_days}
    """
    n, k = X.shape
    # 最小二乘解
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    # 残差方差
    mse = np.sum(residuals ** 2) / (n - k)
    # 标准误
    try:
        var_beta = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se = np.ones(k) * np.nan
    # t 统计量 & p 值
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

    coefs = {}
    for i, name in enumerate(names):
        coefs[name] = {
            "coef": round(float(beta[i]), 6),
            "tstat": round(float(t_stats[i]), 4),
            "pvalue": round(float(p_values[i]), 6),
        }

    return {
        "exposures": coefs,
        "r_squared": round(float(r_squared), 6),
        "adj_r_squared": round(float(adj_r_squared), 6),
        "n_days": int(n),
    }


def evaluate_barra(factor_df: pd.DataFrame, data_dir: Path, barra_dir: Path,
                   model: str = "Trading Model") -> dict:
    """完整 Barra 暴露分析入口。

    Args:
        factor_df: Date × Code 因子矩阵。
        data_dir: 全量数据目录。
        barra_dir: Barra 数据目录。
        model: "Trading Model" 或 "Long-Term Model"。

    Returns:
        包含 alpha 和各风格因子暴露的 dict。
    """
    # 1. 计算多空收益
    ls_returns = compute_long_short_returns(factor_df, data_dir)
    if len(ls_returns) < 30:
        return {"error": f"多空收益有效天数不足: {len(ls_returns)}"}

    # 2. 加载 Barra 因子收益率
    barra_rets = load_barra_factor_returns(barra_dir, model)
    if barra_rets.empty:
        return {"error": "Barra 因子收益率数据为空"}

    # 3. 对齐日期
    aligned = pd.concat({"ls": ls_returns, "barra": barra_rets}, axis=1).dropna()
    # barra 是多列，需要重新对齐
    common_dates = ls_returns.index.intersection(barra_rets.index)
    y = ls_returns.loc[common_dates].values
    X = barra_rets.loc[common_dates].values
    if len(y) < 30:
        return {"error": f"对齐后有效天数不足: {len(y)}"}

    # 4. 加截距项，做 OLS 回归
    X_with_const = np.column_stack([np.ones(len(X)), X])
    names = ["alpha"] + list(barra_rets.columns)
    result = ols_regression(y, X_with_const, names)

    # 5. 附加元信息
    result["model"] = model
    result["n_stocks_per_day"] = int(
        factor_df.notna().sum(axis=1).median() if factor_df.shape[0] > 0 else 0
    )
    # 多空收益的均值（年化）和波动率（年化）
    annual_factor = 252
    ls_mean = float(y.mean()) * annual_factor
    ls_std = float(y.std()) * np.sqrt(annual_factor)
    result["long_short_annual_return"] = round(ls_mean, 6)
    result["long_short_annual_vol"] = round(ls_std, 6)

    return result


def main():
    parser = argparse.ArgumentParser(description="Barra 暴露分析")
    parser.add_argument("factor_parquet", help="因子 parquet 文件路径")
    parser.add_argument("--data-dir", default=str(FULL_DATA_DIR), help="全量数据目录")
    parser.add_argument("--barra-dir", default=str(DEFAULT_BARRA_DIR), help="Barra 数据目录")
    parser.add_argument("--model", default="Trading Model", choices=["Trading Model", "Long-Term Model"],
                        help="Barra 模型（默认 Trading Model）")
    args = parser.parse_args()

    factor_path = Path(args.factor_parquet)
    if not factor_path.exists():
        print(f"文件不存在: {factor_path}")
        return 1

    print(f"加载因子: {factor_path.name}")
    factor_df = pd.read_parquet(factor_path)
    print(f"  规模: {factor_df.shape[0]}天 x {factor_df.shape[1]}只股票")

    data_dir = Path(args.data_dir)
    barra_dir = Path(args.barra_dir)

    print(f"Barra 模型: {args.model}")
    print(f"计算多空收益并回归...")
    result = evaluate_barra(factor_df, data_dir, barra_dir, args.model)

    if "error" in result:
        print(f"  ❌ {result['error']}")
        return 1

    # 打印结果
    print(f"\n{'=' * 50}")
    print(f"Barra 暴露分析结果 ({args.model})")
    print(f"{'=' * 50}")
    alpha = result["exposures"]["alpha"]
    print(f"Alpha:     {alpha['coef']:.6f}  (t={alpha['tstat']:.2f}, p={alpha['pvalue']:.4f})")
    print(f"R²:       {result['r_squared']:.4f}")
    print(f"Adj R²:   {result['adj_r_squared']:.4f}")
    print(f"有效天数: {result['n_days']}")
    print(f"多空年化: {result['long_short_annual_return']:.2%}")
    print(f"\n风格因子暴露 (|t|>2 的显著因子):")
    for name, exp in result["exposures"].items():
        if name == "alpha":
            continue
        if abs(exp["tstat"]) > 2:
            sig = "**" if exp["pvalue"] < 0.01 else "*"
            print(f"  {name:12s}: β={exp['coef']:+.6f}  t={exp['tstat']:+.2f} {sig}")
    print(f"{'=' * 50}")

    # 写入 meta.json
    meta_path = factor_path.parent / f"{factor_path.stem}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["barra_analysis"] = result
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"\n已更新: {meta_path}")
    else:
        output_path = factor_path.parent / f"{factor_path.stem}.barra.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n结果已保存: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
