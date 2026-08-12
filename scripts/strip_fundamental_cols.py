#!/usr/bin/env python3
"""
从行情日线 parquet 中剥离非行情列（行情只保留价量）。

对 数据仓库/行情数据/日线/{全量,测试}/stock_data/daily/*.parquet：
- 删除 16 个非行情列（roe/roa/pe_ttm/pb/成长/盈利/市值/股本 等），
  这些列已迁移到 数据仓库/非行情数据/{全量,测试}/stock_data/daily/。
- 幂等：文件已不含非行情列则跳过。

行情列（保留，20 个）：
  open, close, high, low, factor, volume, pct_chg, pre_close, turnover_rate,
  EMA5, EMA10, EMA20, jhjj_hsl, net_pct_main, net_pct_xl, net_pct_l, net_pct_m,
  net_pct_s, net_amount_main, amount

非行情列（剥离，18 个）：
  roe, roa, pe_ttm, pb, revenue_yoy, profit_yoy, gross_margin, net_margin,
  debt_to_asset, ocf_per_share, market_cap, circulating_market_cap, total_shares,
  float_shares, adjusted_profit, gross_profit, total_holders, holder_change_pct
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKET_DAILY_ROOTS = [
    PROJECT_ROOT / "数据仓库" / "行情数据" / "日线" / "全量" / "stock_data" / "daily",
    PROJECT_ROOT / "数据仓库" / "行情数据" / "日线" / "测试" / "stock_data" / "daily",
]

FUNDAMENTAL_COLS = [
    "roe", "roa", "pe_ttm", "pb", "revenue_yoy", "profit_yoy",
    "gross_margin", "net_margin", "debt_to_asset", "ocf_per_share",
    "market_cap", "circulating_market_cap", "total_shares", "float_shares",
    "adjusted_profit", "gross_profit", "total_holders", "holder_change_pct",
]


def strip_dir(daily_dir: Path) -> int:
    if not daily_dir.exists():
        print(f"  目录不存在: {daily_dir}")
        return 0
    changed = 0
    skipped = 0
    for fpath in sorted(daily_dir.glob("*.parquet")):
        if fpath.name in ("stock_list.json", "trade_dates.json", "industry.json"):
            continue
        df = pd.read_parquet(fpath)
        present = [c for c in FUNDAMENTAL_COLS if c in df.columns]
        if not present:
            skipped += 1
            continue
        df = df.drop(columns=present)
        df.to_parquet(fpath)
        changed += 1
    print(f"  {daily_dir.name}: 修改 {changed} 个, 已剥离跳过 {skipped} 个")
    return changed


def main() -> None:
    total = 0
    for root in MARKET_DAILY_ROOTS:
        print(f"剥离目录: {root}")
        total += strip_dir(root)
    print(f"\n完成! 共剥离 {total} 个行情 parquet 文件的非行情列。")
    print("非行情列现仅存在于: 数据仓库/非行情数据/{全量,测试}/stock_data/daily/")


if __name__ == "__main__":
    main()
