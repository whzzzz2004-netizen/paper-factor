#!/usr/bin/env python3
"""
一键全量流水线：全量/增量补算因子（串行，自动重扫直到队列清空）。

流程:
  1. 扫描全量因子产出目录 下所有因子:
     ├─ 无 .parquet → 全量计算
     ├─ 有 .parquet 但日期落后 → 增量补算（只算新日期，merge 回全量 parquet）
     └─ 已最新 → 跳过
  2. 处理完所有待处理因子后，自动重扫目录
  3. 如果 /factor 在此期间 deploy 了新因子，继续处理
  4. 队列清空后自动退出

和 /factor 并行使用：
  ┌─ dialog 1: 反复跑 /factor 生成因子
  └─ dialog 2: /all 2026-08-16  → 自动全量，队列清空后退出

用法:
  python scripts/run_all.py                        # 扫描最近日期目录
  python scripts/run_all.py 2026-08-15             # 指定日期目录
  python scripts/run_all.py --report 研报名        # 只跑指定研报
  python scripts/run_all.py --force                # 强制重跑（无视状态）
  python scripts/run_all.py --dry-run              # 只打印计划，不执行
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 确保项目根目录在 sys.path 中（用于 from scripts.xxx import）
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from scripts.factor_utils import (
    load_trade_dates,
    run_factor_subprocess,
    merge_incremental_result,
    backup_parquet,
    cleanup_parquet_backup,
    update_factor_meta,
    evaluate_factor,
    detect_factor_type,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path("/mnt/d/paper-factor-data")

# ── 路径（仅本地） ──
OUTPUT_BASE = DATA_ROOT / "数据仓库" / "因子产出" / "全量"
FULL_DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(DATA_ROOT / "数据仓库" / "行情数据" / "日线" / "全量")))


# ── 扫描 ──

def find_pending_factors(report_filter: str | None, force: bool, base_dir: Path | None = None) -> list[dict]:
    """
    返回待处理因子列表，每项含 {report, factor, code_path, output_dir, meta_path, parquet_path, status}
    status: "pending" (无 parquet), "stale" (有 parquet 但日期老), "current" (已最新)
    """
    base = base_dir or OUTPUT_BASE
    if not base.exists():
        return []

    factors = []
    trade_dates = None  # 延迟加载

    report_dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if report_filter:
        report_dirs = [d for d in report_dirs if report_filter in d.name]

    for report_dir in report_dirs:
        report_name = report_dir.name
        factor_dirs = sorted(d for d in report_dir.iterdir() if d.is_dir())
        for factor_dir in factor_dirs:
            factor_name = factor_dir.name
            code_path = factor_dir / f"{factor_name}.code.py"
            parquet_path = factor_dir / f"{factor_name}.parquet"
            meta_path = factor_dir / f"{factor_name}.meta.json"

            if not code_path.exists():
                continue

            # 强制重跑 → 直接标记 pending
            if force:
                factors.append({
                    "report": report_name,
                    "factor": factor_name,
                    "code_path": code_path,
                    "output_dir": factor_dir,
                    "parquet_path": parquet_path,
                    "meta_path": meta_path,
                    "status": "pending",
                })
                continue

            # 无 parquet → pending
            if not parquet_path.exists():
                factors.append({
                    "report": report_name,
                    "factor": factor_name,
                    "code_path": code_path,
                    "output_dir": factor_dir,
                    "parquet_path": parquet_path,
                    "meta_path": meta_path,
                    "status": "pending",
                })
                continue

            # 有 parquet → 检查日期
            try:
                df = pd.read_parquet(parquet_path)
                last_date = pd.Timestamp(df.index.max())
            except Exception:
                # 损坏 → 重跑
                factors.append({
                    "report": report_name,
                    "factor": factor_name,
                    "code_path": code_path,
                    "output_dir": factor_dir,
                    "parquet_path": parquet_path,
                    "meta_path": meta_path,
                    "status": "pending",
                })
                continue

            # 延迟加载 trade_dates
            if trade_dates is None:
                try:
                    trade_dates = load_trade_dates(FULL_DATA_DIR)
                except Exception as e:
                    print(f"  ⚠️ 无法读取 trade_dates: {e}")
                    continue
            latest_date = pd.Timestamp(trade_dates[-1])

            if latest_date <= last_date:
                # 已最新
                factors.append({
                    "report": report_name,
                    "factor": factor_name,
                    "code_path": code_path,
                    "output_dir": factor_dir,
                    "parquet_path": parquet_path,
                    "meta_path": meta_path,
                    "status": "current",
                })
            else:
                # 需要增量
                factors.append({
                    "report": report_name,
                    "factor": factor_name,
                    "code_path": code_path,
                    "output_dir": factor_dir,
                    "parquet_path": parquet_path,
                    "meta_path": meta_path,
                    "status": "stale",
                    "last_date": last_date,
                    "latest_date": latest_date,
                })

    return factors


# ── 执行（仅串行） ──

def run_full_pipeline_for_factor(item: dict) -> dict:
    """跑单个因子的全量流水线（调用 factor_full_pipeline）"""
    factor_name = item["factor"]
    report_name = item["report"]
    code_path = item["code_path"]
    output_dir = item["output_dir"]

    print(f"\n{'='*60}")
    print(f"▶ [全量] {report_name}/{factor_name}")
    print(f"{'='*60}\n")

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from rdagent.app.qlib_rd_loop.factor_full_pipeline import run_full_pipeline

        ok = run_full_pipeline(
            factor_name=factor_name,
            code_path=code_path,
            output_dir=output_dir,
            factor_type=None,
            test_meta=None,
            source_excerpt="",
        )

        status = "success" if ok else "failed"
        print(f"  {'✅' if ok else '❌'} {report_name}/{factor_name} {'完成' if ok else '失败'}")
        return {"report": report_name, "factor": factor_name, "status": status}

    except Exception as e:
        print(f"  ❌ {report_name}/{factor_name} 异常: {e}")
        return {"report": report_name, "factor": factor_name, "status": "error", "error": str(e)}


def run_incremental_for_factor(item: dict) -> dict:
    """跑单个因子的增量更新"""
    factor_name = item["factor"]
    report_name = item["report"]
    code_path = item["code_path"]
    output_dir = item["output_dir"]
    parquet_path = item["parquet_path"]
    last_date = item["last_date"]
    latest_date = item["latest_date"]

    print(f"\n{'='*60}")
    print(f"▶ [增量] {report_name}/{factor_name}")
    print(f"  {last_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")

    result = {"report": report_name, "factor": factor_name}

    # 1. 读已有的 parquet
    try:
        existing_df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  ❌ 读取现有 parquet 失败: {e}，降级为全量")
        return run_full_pipeline_for_factor(item)

    # 2. 判断因子类型，确定数据目录
    code_text = code_path.read_text(encoding="utf-8")
    factor_type = detect_factor_type(code_text)
    market_data_dir = (DATA_ROOT / "数据仓库" / "行情数据" / "分钟线" / "全量"
                       if factor_type in ("minute", "minute_cross_section") else FULL_DATA_DIR)
    start_date_str = last_date.strftime("%Y-%m-%d")
    result_parquet = run_factor_subprocess(
        code_text, factor_name, market_data_dir,
        start_date=start_date_str, n_workers=4, timeout=7200,
    )
    if result_parquet is None:
        result["status"] = "failed"
        result["error"] = "子进程执行失败"
        return result

    # 3. 读取结果，裁掉重叠，合并
    new_df = pd.read_parquet(result_parquet)
    result_parquet.unlink(missing_ok=True)  # 清理临时文件

    # 传权威交易日历给 merge：裁剪周末/非交易日脏行（旧数据若含周末也一并清除）
    _cal = load_trade_dates(market_data_dir)
    combined = merge_incremental_result(existing_df, new_df, last_date, trade_dates=_cal)
    if combined is existing_df:
        print("  ⚠️ 增量结果为空")
        result["status"] = "skipped"
        return result

    # 备份 + 写回
    backup_parquet(parquet_path)
    combined.to_parquet(parquet_path)
    print(f"  ✅ 合并完成: {combined.shape[0]} 行 ({combined.shape[0] - len(existing_df)} 新增)")

    # 4. 评估 + 绘图（重新生成 decile.png，更新 meta.json 的 evaluation）
    _eval_result = evaluate_factor(parquet_path, factor_name, output_dir, FULL_DATA_DIR)

    # 5. 更新 meta.json
    meta_extra = {"pipeline_status": "completed"}
    if _eval_result:
        meta_extra["evaluation"] = _eval_result
    update_factor_meta(output_dir / f"{factor_name}.meta.json", combined, extra=meta_extra)

    # 6. 动态清理：删除写回时创建的 .parquet.bak 备份，
    #    让因子目录看着就和全新计算的一样（无中间产物残留）
    cleanup_parquet_backup(parquet_path)

    print(f"  ✅ [增量] {report_name}/{factor_name} 完成")
    result["status"] = "success"
    result["new_dates"] = max(0, combined.shape[0] - len(existing_df))
    return result


# ── 主流程（串行，自动重扫直到队列清空） ──

def main():
    parser = argparse.ArgumentParser(description="一键全量流水线：全量/增量补算因子（串行，自动重扫直到队列清空）")
    parser.add_argument("subdir", nargs="?", default=None, help="日期子目录 (如 2026-08-15)，默认最近日期")
    parser.add_argument("--report", help="指定研报名 (模糊匹配)", default=None)
    parser.add_argument("--force", action="store_true", help="强制重跑（无视状态）")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待跑因子，不执行")
    args = parser.parse_args()

    t_start = time.time()

    print("📌 本地模式（串行，自动重扫）")

    # ── Step 1: 确定目标目录 ──
    if args.subdir:
        date_str = args.subdir
        target_base = OUTPUT_BASE / date_str
        if target_base.exists():
            scan_base = target_base
            print(f"📅 扫描子目录: {date_str}")
        else:
            print(f"❌ 子目录不存在: {target_base}")
            return 1
    else:
        # 未指定日期 → 找最近的日期目录
        date_dirs = []
        for d in OUTPUT_BASE.iterdir():
            if d.is_dir():
                try:
                    dt = datetime.strptime(d.name, '%Y-%m-%d')
                    date_dirs.append((dt, d))
                except ValueError:
                    continue
        if not date_dirs:
            print("❌ 未找到日期目录")
            return 1
        date_dirs.sort(key=lambda x: x[0], reverse=True)
        scan_base = date_dirs[0][1]
        date_str = scan_base.name
        print(f"📅 扫描最近日期目录: {date_str}")

    # ── Step 2-3: 循环：扫描 → 处理 → 重扫，直到队列清空 ──
    round_num = 0
    total_success = 0
    total_fail = 0
    total_skip = 0

    while True:
        round_num += 1
        if round_num > 1:
            print(f"\n{'='*60}")
            print(f"🔄 第 {round_num} 轮重扫：检查是否有新因子…")
            print(f"{'='*60}")

        pending = find_pending_factors(args.report, args.force, base_dir=scan_base)
        pending_list = [p for p in pending if p["status"] in ("pending", "stale")]

        if not pending_list:
            if round_num == 1:
                print("\n✅ 无待处理因子")
            else:
                print("\n✅ 队列已清空，无新因子")
            break

        pending_count = sum(1 for p in pending_list if p["status"] == "pending")
        stale_count = sum(1 for p in pending_list if p["status"] == "stale")
        current_list = [p for p in pending if p["status"] == "current"]

        print(f"\n📊 本轮 {len(pending_list)} 个待处理（{pending_count} 全量 + {stale_count} 增量）")
        if current_list:
            print(f"   ✅ 已最新跳过: {len(current_list)} 个")

        if args.dry_run:
            print("\n待处理列表:")
            for p in pending_list:
                tag = "全量" if p["status"] == "pending" else "增量"
                detail = f"({p['last_date'].strftime('%Y-%m-%d')} → {p['latest_date'].strftime('%Y-%m-%d')})" if p["status"] == "stale" else ""
                print(f"  [{tag}] {p['report']}/{p['factor']} {detail}")
            return 0

        # 串行执行本轮因子
        for item in pending_list:
            if item["status"] == "pending":
                r = run_full_pipeline_for_factor(item)
            else:
                r = run_incremental_for_factor(item)
            if r["status"] == "success":
                total_success += 1
            elif r["status"] == "skipped":
                total_skip += 1
            else:
                total_fail += 1

        # 本轮完成 → 自动重扫（看 /factor 是否 deploy 了新因子）
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"🏁 第 {round_num} 轮完成 (累计耗时 {elapsed/60:.1f}min)，即将重扫检查新因子…")
        print(f"{'='*60}")

    # ── 汇总 ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏁 全部完成: {total_success} 成功, {total_fail} 失败, {total_skip} 跳过 (耗时 {elapsed/60:.1f}min)")
    print(f"{'='*60}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())