#!/usr/bin/env python3
"""
修复 sync_data.py 造成的日线数据命名冲突。

背景:
  本地日线数据原始命名是 str(int(code))（去前导零）: 1.parquet, 300725.parquet
  sync_data.py 的 sync_daily_incremental 用 str(symbol).zfill(6) 写文件:
    - code>=4000 (如 300725) → 同名追加, 正常
    - code<4000 (如 000001) → 新写 000001.parquet (重复文件, 只含新100天)
  同时把 stock_list.json 覆盖成了仅含新数据股票的 padded 代码。

本脚本:
  1. 把每个 padded 文件合并回非 padded 原始文件（按日期去重, 列取并集）
  2. 对新追加的行计算 pct_chg / pre_close（其余缺失列保持 NaN）
  3. 删除 padded 重复文件
  4. 用非 padded 文件重建 stock_list.json（还原 5435 只全量股票）
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "git_ignore_folder" / "factor_implementation_source_data"
DAILY_DIR = DATA_DIR / "stock_data" / "daily"
BACKUP_DIR = DATA_DIR / "stock_data" / "daily_reconcile_backup"


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DAILY_DIR.glob("*.parquet"))
    # 分类: padded (000xxx 等 6位前导0) vs 非 padded
    padded = []
    nonpadded = []
    for f in files:
        s = f.stem
        if not s or not s[0].isdigit():
            continue
        try:
            n = str(int(s))
        except ValueError:
            continue
        if len(s) == 6 and s[0] == "0" and n != s:
            padded.append(f)
        else:
            nonpadded.append(f)

    print(f"非 padded 文件: {len(nonpadded)}, padded 文件: {len(padded)}")

    merged = 0
    for pf in padded:
        np_code = str(int(pf.stem))
        npf = DAILY_DIR / f"{np_code}.parquet"
        if not npf.exists():
            print(f"  ⚠️ 跳过 {pf.name}: 无对应非padded文件 {npf.name}")
            continue

        # 备份原始非 padded 文件（只备份一次）
        bak = BACKUP_DIR / npf.name
        if not bak.exists():
            shutil.copy2(npf, bak)

        old = pd.read_parquet(npf)
        new = pd.read_parquet(pf)

        # 统一索引
        old = _ensure_dt_index(old)
        new = _ensure_dt_index(new)

        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        # 补算 pct_chg / pre_close（仅 NaN 处填充, 不覆盖已有值）
        if "pct_chg" in combined.columns:
            computed = combined["close"].pct_change() * 100.0
            combined["pct_chg"] = combined["pct_chg"].fillna(computed)
        if "pre_close" in combined.columns:
            combined["pre_close"] = combined["pre_close"].fillna(combined["close"].shift(1))

        combined.to_parquet(npf)
        merged += 1

    print(f"已合并 {merged} 个 padded 文件")

    # 删除 padded 文件
    deleted = 0
    for pf in padded:
        if pf.exists():
            pf.unlink()
            deleted += 1
    print(f"已删除 {deleted} 个 padded 文件")

    # 重建 stock_list.json
    codes = sorted(int(f.stem) for f in nonpadded)
    stock_list = [str(c) for c in codes]
    (DAILY_DIR / "stock_list.json").write_text(json.dumps(stock_list, ensure_ascii=False))
    print(f"stock_list.json: {len(stock_list)} 只")

    # 验证
    df = pd.read_parquet(DAILY_DIR / "1.parquet")
    print(f"验证 1.parquet: {len(df)} 行, last={df.index[-1].date()}, "
          f"pct_chg 非NaN={df['pct_chg'].notna().sum()}")


def _ensure_dt_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    main()
