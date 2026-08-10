#!/usr/bin/env python3
"""
一键全量流水线：全量/增量补算因子。

流程:
  1. 扫描全量因子产出目录 下所有因子:
     ├─ 无 .parquet → 全量计算
     ├─ 有 .parquet 但日期落后 → 增量补算（只算新日期，merge 回全量 parquet）
     └─ 已最新 → 跳过

用法:
  python scripts/run_all.py                        # 扫描所有因子
  python scripts/run_all.py 20260726               # 只扫描指定目录
  python scripts/run_all.py --report 研报名        # 只跑指定研报
  python scripts/run_all.py --force                # 强制重跑（无视状态）
  python scripts/run_all.py --workers 3            # 并行数
  python scripts/run_all.py --dry-run              # 只打印计划，不执行
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ── 路径（仅本地） ──
OUTPUT_BASE = PROJECT_ROOT / "数据仓库" / "因子产出" / "全量"
FULL_DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(PROJECT_ROOT / "数据仓库" / "行情数据" / "日线" / "全量")))


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


# ── 执行 ──

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
    data_dir = (PROJECT_ROOT / "数据仓库" / "行情数据" / "分钟线" / "全量") if factor_type == "minute" else FULL_DATA_DIR

    # 3. 执行子进程
    start_date_str = last_date.strftime("%Y-%m-%d")
    result_parquet = run_factor_subprocess(
        code_text, factor_name, data_dir,
        start_date=start_date_str, n_workers=4, timeout=7200,
    )
    if result_parquet is None:
        result["status"] = "failed"
        result["error"] = "子进程执行失败"
        return result

    # 3. 读取结果，裁掉重叠，合并
    new_df = pd.read_parquet(result_parquet)
    result_parquet.unlink(missing_ok=True)  # 清理临时文件

    combined = merge_incremental_result(existing_df, new_df, last_date)
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


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser(description="一键全量流水线：全量/增量补算因子（默认本地模式）")
    parser.add_argument("subdir", nargs="?", default=None, help="日期子目录 (如 20260726)，默认当天")
    parser.add_argument("--report", help="指定研报名 (模糊匹配)", default=None)
    parser.add_argument("--force", action="store_true", help="强制重跑（无视状态）")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数 (默认: 1)")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待跑因子，不执行")
    args = parser.parse_args()

    t_start = time.time()

    print("📌 本地模式")

    # ── Step 1: 确定目标目录（指定子目录 或 临时目录 → 完成时重命名） ──
    if args.subdir:
        date_str = args.subdir
        target_base = OUTPUT_BASE / date_str
        if target_base.exists():
            scan_base = target_base
            print(f"📅 扫描子目录: {date_str}")
        else:
            scan_base = OUTPUT_BASE
            print(f"⚠️ 子目录不存在: {target_base}，回退到根目录")
        tmp_base = None
    else:
        # 自动模式：扫描根目录，写入临时目录，完成时重命名为完成日期
        scan_base = OUTPUT_BASE
        tmp_name = f"_tmp_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        tmp_base = OUTPUT_BASE / tmp_name
        print(f"📅 临时目录: {tmp_name}")

    # ── Step 2: 扫描因子 ──
    pending = find_pending_factors(args.report, args.force, base_dir=scan_base)

    if not pending:
        print("\n✅ 无待处理因子")
        return 0

    # 分类统计
    pending_list = [p for p in pending if p["status"] in ("pending", "stale")]
    current_list = [p for p in pending if p["status"] == "current"]
    pending_count = sum(1 for p in pending if p["status"] == "pending")
    stale_count = sum(1 for p in pending if p["status"] == "stale")

    print(f"\n📊 共 {len(pending_list)} 个待处理因子（{pending_count} 全量 + {stale_count} 增量）")
    if current_list:
        print(f"   ✅ 已最新跳过: {len(current_list)} 个")

    if args.dry_run:
        print("\n待处理列表:")
        for p in pending_list:
            if p["status"] == "pending":
                print(f"  [全量] {p['report']}/{p['factor']}")
            else:
                print(f"  [增量] {p['report']}/{p['factor']} "
                      f"({p['last_date'].strftime('%Y-%m-%d')} → {p['latest_date'].strftime('%Y-%m-%d')})")
        return 0

    # 临时目录模式：创建临时目录并重定向输出路径
    if tmp_base is not None:
        tmp_base.mkdir(parents=True, exist_ok=True)
        for item in pending_list:
            report = item["report"]
            factor = item["factor"]
            new_output = tmp_base / report / factor
            new_output.mkdir(parents=True, exist_ok=True)
            # 增量因子：复制现有 parquet 到临时目录作为起点
            if item["status"] == "stale":
                src = item["parquet_path"]
                if src.exists():
                    shutil.copy2(src, new_output / f"{factor}.parquet")
            item["output_dir"] = new_output
            item["parquet_path"] = new_output / f"{factor}.parquet"
            item["meta_path"] = new_output / f"{factor}.meta.json"

    # ── Step 3: 执行 ──
    success_count = 0
    fail_count = 0
    skipped_count = 0

    if args.workers > 1 and len(pending_list) > 1:
        # 并行模式
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fut_map = {}
            for item in pending_list:
                if item["status"] == "pending":
                    fut = pool.submit(run_full_pipeline_for_factor, item)
                else:
                    fut = pool.submit(run_incremental_for_factor, item)
                fut_map[fut] = item

            for fut in as_completed(fut_map):
                r = fut.result()
                if r["status"] == "success":
                    success_count += 1
                elif r["status"] == "skipped":
                    skipped_count += 1
                else:
                    fail_count += 1
    else:
        # 串行模式
        for item in pending_list:
            if item["status"] == "pending":
                r = run_full_pipeline_for_factor(item)
            else:
                r = run_incremental_for_factor(item)
            if r["status"] == "success":
                success_count += 1
            elif r["status"] == "skipped":
                skipped_count += 1
            else:
                fail_count += 1

    # ── 重命名临时目录为完成日期 ──
    if tmp_base is not None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        final_base = OUTPUT_BASE / date_str
        if final_base.exists():
            print(f"📦 合并到已有目录: {date_str}")
            for item in tmp_base.rglob('*'):
                if item.is_file():
                    rel = item.relative_to(tmp_base)
                    dst = final_base / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(dst))
            shutil.rmtree(tmp_base)
        else:
            tmp_base.rename(final_base)
            print(f"📅 重命名为完成日期: {date_str}")

    # ── 汇总 ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏁 完成: {success_count} 成功, {fail_count} 失败, {skipped_count} 跳过 (耗时 {elapsed/60:.1f}min)")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
