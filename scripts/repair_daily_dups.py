#!/usr/bin/env python3
"""
修复日线 per-stock parquet 中重复日期问题。

背景:
  33 只新股 (301xxx/601xxx/603xxx/688xxx) 的同一日期存在两行:
    - rich 行: 来自 dailyData.parquet 全量同步 (含 turnover_rate/roe/market_cap 等财务列)
    - sparse 行: 来自增量同步 (仅 open/close/high/low/factor/volume/amount)
  导致 _series.loc[td] 返回 Series, 因子计算报 "truth value of a Series is ambiguous"。

本脚本:
  1. 扫描所有 per-stock parquet, 找出有重复日期的文件
  2. 对每个重复日期, 按列合并: 每列取组内第一个非NaN值 (rich+sparse 合并成完整行)
  3. 修改前备份到 daily_dups_backup/
  4. 验证无重复
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DAILY_DIR = Path(__file__).resolve().parent.parent / "git_ignore_folder" / "factor_implementation_source_data" / "stock_data" / "daily"
BACKUP_DIR = DAILY_DIR / "daily_dups_backup"


def merge_dup_dates(df: pd.DataFrame) -> pd.DataFrame:
    """按日期去重合并：每列取组内第一个非NaN值。"""
    if not df.index.duplicated().any():
        return df
    return df.groupby(level=0).agg(
        lambda s: next((v for v in s.dropna() if not pd.isna(v)), np.nan)
    ).sort_index()


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    stock_list = json.loads((DAILY_DIR / "stock_list.json").read_text())
    fixed = []
    skipped_err = []

    for s in stock_list:
        f = DAILY_DIR / f"{s}.parquet"
        if not f.exists():
            skipped_err.append(f"{s}: 文件不存在")
            continue
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            skipped_err.append(f"{s}: 读取失败 {e}")
            continue
        if not df.index.duplicated().any():
            continue
        # 备份
        bak = BACKUP_DIR / f.name
        if not bak.exists():
            shutil.copy2(f, bak)
        n_dup = int(df.index.duplicated().sum())
        merged = merge_dup_dates(df)
        merged.to_parquet(f)
        fixed.append((s, len(df), len(merged), n_dup))

    print(f"扫描 {len(stock_list)} 只, 修复 {len(fixed)} 只:")
    for s, before, after, ndup in fixed:
        print(f"  {s}: {before} → {after} 行 (去重 {ndup} 行)")
    if skipped_err:
        print(f"⚠️ 异常 {len(skipped_err)} 只:")
        for e in skipped_err:
            print(f"  {e}")

    # 验证
    bad = 0
    for s in stock_list:
        try:
            df = pd.read_parquet(DAILY_DIR / f"{s}.parquet")
            if df.index.duplicated().any():
                bad += 1
                print(f"  ❌ 仍重复: {s}")
        except Exception:
            pass
    print(f"验证完成: {'全部干净' if bad == 0 else f'{bad} 只仍有重复'}")


if __name__ == "__main__":
    main()
