#!/usr/bin/env python3
"""用全量分钟数据补齐测试集缺失的 (股票, 日期) 组合。

背景：测试分钟数据 300 只股票中，114 只不齐（89 只仅有前 50 天），
共缺 22,670 个 (股票,日期) 组合。全量数据完整覆盖这些组合。

做法：逐日打开测试文件，找出测试股票清单里缺失的股票，
从全量同日文件中取出这些股票的全部分钟行，追加、去重、排序后写回。
"""
import json
import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线")
T = ROOT / "测试" / "stock_data" / "minute_by_date"
F = ROOT / "全量" / "stock_data" / "minute_by_date"
BK = ROOT / "测试" / "stock_data" / "_minute_by_date_backup"

td = json.load(open(ROOT / "测试" / "stock_data" / "trade_dates.json"))
sl = set(json.load(open(ROOT / "测试" / "stock_data" / "stock_list.json")))


def main():
    if not BK.exists():
        print(f"备份 → {BK}", flush=True)
        shutil.copytree(T, BK, ignore=shutil.ignore_patterns("_*"))
    else:
        print(f"备份已存在: {BK}", flush=True)

    t0 = time.time()
    fixed_days = 0
    added_rows = 0
    for i, d in enumerate(td):
        tp = T / f"{d}.parquet"
        if not tp.exists():
            print(f"  ⚠️ 测试文件缺失: {d}", flush=True)
            continue
        tdf = pd.read_parquet(tp)
        present = set(tdf.index.get_level_values("instrument").unique())
        missing = sl - present
        if not missing:
            continue
        fp = F / f"{d}.parquet"
        if not fp.exists():
            print(f"  ⚠️ 全量文件缺失: {d}（{len(missing)} 只无法补齐）", flush=True)
            continue
        fdf = pd.read_parquet(fp)
        add = fdf[fdf.index.get_level_values("instrument").isin(missing)]
        if add.empty:
            print(f"  ⚠️ {d}: 全量中未找到这 {len(missing)} 只", flush=True)
            continue
        out = pd.concat([tdf, add])
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_parquet(tp)
        fixed_days += 1
        added_rows += len(add)
        if (i + 1) % 30 == 0:
            print(f"  进度 {i+1}/{len(td)} 天，已修 {fixed_days} 天，补 {added_rows} 行，"
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"\n完成: 修复 {fixed_days} 个日期文件，新增 {added_rows} 行，"
          f"耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
