
#!/usr/bin/env python3
"""修复分钟数据 datetime 时间戳：所有值为 00:00:00 → 恢复为正确的分钟时间戳。

真实分钟数据（D:\\market_minute_daily_new 导入）每只股票每天 240 分钟：
  - 09:31-11:30 (120 分钟)
  - 13:01-15:00 (120 分钟)

本脚本只用于修复历史损坏的分钟 parquet（datetime 全为 00:00:00）。
新导入的真实数据自带正确分钟时间戳，无需运行本脚本。

用法:
    python /mnt/d/paper-factor-data/scripts/repair_minute_datetime.py                          # 全量
    python /mnt/d/paper-factor-data/scripts/repair_minute_datetime.py --data-dir "..."         # 指定目录
    python /mnt/d/paper-factor-data/scripts/repair_minute_datetime.py --dry-run                # 预览
    python /mnt/d/paper-factor-data/scripts/repair_minute_datetime.py --workers 8              # 并行数
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
DATA_ROOT = Path(os.environ.get("PAPER_FACTOR_DATA_ROOT", "/mnt/d/paper-factor-data"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def generate_timestamps(date_str: str, n_per_instrument: int) -> pd.DatetimeIndex:
    """生成某一天的真实分钟时间戳（240 个：09:31-11:30 + 13:01-15:00）。

    若 n_per_instrument 不是 240（极少情况，历史假数据曾为 242），按需截断或填充。
    """
    base = pd.Timestamp(date_str)
    morning = pd.date_range(base.replace(hour=9, minute=31), base.replace(hour=11, minute=30), freq="1min")
    afternoon = pd.date_range(base.replace(hour=13, minute=1), base.replace(hour=15, minute=0), freq="1min")
    full = morning.append(afternoon)
    if len(full) >= n_per_instrument:
        return full[:n_per_instrument]
    # 填充：用最后时间重复
    pad = [full[-1]] * (n_per_instrument - len(full))
    return full.append(pd.DatetimeIndex(pad))


def repair_one_file(file_path: Path) -> dict:
    """修复单个文件的 datetime 索引"""
    fname = file_path.name  # e.g. 20260105.parquet
    date_str = fname.replace(".parquet", "")
    # 解析日期：YYYYMMDD → YYYY-MM-DD
    try:
        dt = pd.Timestamp(date_str)
    except ValueError:
        return {"file": fname, "status": "skipped", "reason": f"cannot parse date: {date_str}"}

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return {"file": fname, "status": "error", "reason": str(e)}

    if df.empty:
        return {"file": fname, "status": "skipped", "reason": "empty"}

    # 检查当前 datetime 值
    try:
        current_dts = df.index.get_level_values("datetime")
    except Exception:
        return {"file": fname, "status": "skipped", "reason": "no datetime level"}

    # 如果已经有非午夜时间戳，跳过
    unique_times = current_dts.nunique()
    if unique_times > 1:
        # 可能已有正确时间戳
        first_time = current_dts[0]
        if first_time.hour != 0 or first_time.minute != 0:
            return {"file": fname, "status": "skipped", "reason": f"already has {unique_times} unique times"}

    # 获取 instrument 数量
    instruments = df.index.get_level_values("instrument")
    unique_instruments = instruments.unique()
    n_instruments = len(unique_instruments)

    # 计算每只股票的行数
    counts = instruments.value_counts()
    n_per = counts.iloc[0]
    if not (counts == n_per).all():
        # 不一致的行数 — 按 instrument 逐个处理
        # 这比较慢，但安全
        new_levels = []
        for inst in unique_instruments:
            sub = df.xs(inst, level="instrument")
            ts = generate_timestamps(dt.strftime("%Y-%m-%d"), len(sub))
            new_levels.append(ts)
        all_timestamps = pd.DatetimeIndex(np.concatenate([l.values for l in new_levels]))
    else:
        # 所有 instrument 行数一致 → 向量化
        ts = generate_timestamps(dt.strftime("%Y-%m-%d"), n_per)
        # 为每个 instrument 重复
        all_timestamps = pd.DatetimeIndex(np.tile(ts.values, n_instruments))

    # 构建新的 MultiIndex
    new_index = pd.MultiIndex.from_arrays(
        [instruments, all_timestamps],
        names=["instrument", "datetime"],
    )
    df.index = new_index

    # 写回
    df.to_parquet(file_path)
    return {"file": fname, "status": "ok", "rows": len(df), "unique_times": len(all_timestamps.unique())}


def main():
    parser = argparse.ArgumentParser(description="修复分钟数据 datetime 时间戳")
    parser.add_argument("--data-dir", default=None, help="分钟线 minute_by_date 目录")
    parser.add_argument("--dry-run", action="store_true", help="预览不执行")
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    args = parser.parse_args()

    if args.data_dir:
        minute_dir = Path(args.data_dir)
    else:
        proj_root = Path(__file__).resolve().parent.parent
        minute_dir = DATA_ROOT / "数据仓库" / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute_by_date"

    if not minute_dir.exists():
        print(f"错误: 目录不存在 {minute_dir}")
        return 1

    files = sorted(minute_dir.glob("*.parquet"))
    # 排除 market_minute_return
    files = [f for f in files if "market_minute_return" not in f.name]
    print(f"找到 {len(files)} 个文件待处理")

    # 快速检查文件数量
    if args.dry_run:
        # 检查几个文件
        for f in files[:5]:
            df = pd.read_parquet(f)
            dts = df.index.get_level_values("datetime")
            print(f"  {f.name}: {len(df)} 行, {dts.nunique()} 唯一时间, 示例: {dts[0]}")
        print(f"\n共 {len(files)} 个文件待修复。实际修复将逐个重写 parquet。")
        return 0

    t0 = time.time()
    ok = skipped = errors = 0

    n_workers = min(args.workers, len(files))
    if n_workers <= 1:
        # 单进程
        for i, f in enumerate(files):
            result = repair_one_file(f)
            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                errors += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  进度: {i+1}/{len(files)}, 耗时 {elapsed:.0f}s, ETA {(elapsed/(i+1))*(len(files)-i-1):.0f}s")
    else:
        # 并行处理
        print(f"使用 {n_workers} 个进程并行处理...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            fut_map = {pool.submit(repair_one_file, f): f for f in files}
            done = 0
            for fut in as_completed(fut_map):
                done += 1
                try:
                    result = fut.result(timeout=300)
                    if result["status"] == "ok":
                        ok += 1
                    elif result["status"] == "skipped":
                        skipped += 1
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1
                    fname = fut_map[fut].name
                    print(f"  ⚠️ {fname} 异常: {e}")
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  进度: {done}/{len(files)}, 耗时 {elapsed:.0f}s, ETA {(elapsed/done)*(len(files)-done):.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {ok} 修复, {skipped} 跳过, {errors} 错误, 耗时 {elapsed:.0f}s")

    # 修复后验证
    print("\n验证修复效果:")
    for f in files[:3]:
        df = pd.read_parquet(f)
        dts = df.index.get_level_values("datetime")
        sample_times = sorted(dts.unique())[:5]
        print(f"  {f.name}: {len(df)} 行, {dts.nunique()} 唯一时间, 示例: {sample_times}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())