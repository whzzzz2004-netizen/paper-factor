#!/usr/bin/env python3
"""
一键全量流水线：挂载 → 同步数据 → 全量/增量补算因子。

流程:
  1. 挂载远程E盘（如未挂载）
  2. 同步最新数据（market_daily_daily_new / market_minute_daily_new → per-stock parquet）
  3. 扫描文献因子_全量/ 下所有因子:
     ├─ 无 .parquet → 全量计算
     ├─ 有 .parquet 但日期落后 → 增量补算（只算新日期，merge 回全量 parquet）
     └─ 已最新 → 跳过

用法:
  python scripts/run_all.py                        # 本地模式，扫描所有因子
  python scripts/run_all.py 20260726               # 只扫描指定目录
  python scripts/run_all.py --report 研报名        # 只跑指定研报
  python scripts/run_all.py --force                # 强制重跑（无视状态）
  python scripts/run_all.py --workers 3            # 并行数
  python scripts/run_all.py --dry-run              # 只打印计划，不执行
  python scripts/run_all.py --remote               # 启用远程挂载+数据同步
"""

import argparse
import os
import subprocess
import sys
import time
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
)

PROJECT_ROOT = Path(__file__).parent.parent

# ── 路径（默认本地，远程挂载成功后再切换，模块级别不访问 CIFS 避免卡死） ──
REMOTE_MOUNT = Path("/mnt/remote_e")
REMOTE_FULL = REMOTE_MOUNT / "paper_factors" / "文献因子_全量"
LOCAL_FULL = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"

# 全量输出 + 数据目录（初始默认本地，确保模块级别不访问 CIFS）
OUTPUT_BASE = LOCAL_FULL
FULL_DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data")))
REMOTE_DATA_DIR = REMOTE_MOUNT / "_paper_factor_unified" / "factor_implementation_source_data"

# sync_data.py 路径
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_data.py"

# ── 远程挂载 ──

def ensure_mounted() -> bool:
    """确保远程E盘已挂载，返回是否成功。超时15秒，不卡死。"""
    global OUTPUT_BASE, FULL_DATA_DIR

    # 用 mountpoint 检查（安全，不卡死）
    try:
        r = subprocess.run(["mountpoint", "-q", str(REMOTE_MOUNT)], capture_output=True, timeout=5)
        if r.returncode == 0:
            print("  ✅ 远程已挂载")
            OUTPUT_BASE = REMOTE_FULL
            if REMOTE_DATA_DIR.exists():
                FULL_DATA_DIR = REMOTE_DATA_DIR
            return True
    except Exception:
        pass

    # 尝试挂载
    print("📌 挂载远程E盘...")
    os.makedirs(str(REMOTE_MOUNT), exist_ok=True)
    uid = os.getuid()
    gid = os.getgid()
    mount_cmd = [
        "mount", "-t", "cifs",
        "//192.168.1.13/E", str(REMOTE_MOUNT),
        "-o", f"user=pc,password=123456,uid={uid},gid={gid},"
              f"file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"
    ]
    if os.geteuid() != 0:
        mount_cmd = ["sudo", "-n"] + mount_cmd

    try:
        subprocess.run(mount_cmd, capture_output=True, timeout=15)
        r = subprocess.run(["mountpoint", "-q", str(REMOTE_MOUNT)], capture_output=True, timeout=5)
        if r.returncode == 0:
            print("  ✅ 已挂载")
            OUTPUT_BASE = REMOTE_FULL
            if REMOTE_DATA_DIR.exists():
                FULL_DATA_DIR = REMOTE_DATA_DIR
            return True
    except subprocess.TimeoutExpired:
        print("  ⚠️ 挂载超时（15s）")
    except Exception as e:
        print(f"  ⚠️ 挂载失败: {e}")

    print("  ⚠️ 使用本地数据")
    return False


# ── 数据同步 ──

def sync_data() -> bool:
    """运行 sync_data.py 同步最新数据，实时显示输出"""
    if not SYNC_SCRIPT.exists():
        print("  ⚠️ sync_data.py 不存在，跳过数据同步")
        return False
    print("📌 同步最新数据...")
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(SYNC_SCRIPT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        line = line.strip()
        if line:
            print(f"  {line}")
    proc.wait(timeout=3600)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  ⚠️ 数据同步异常 (exit={proc.returncode}, {elapsed:.0f}s)，继续执行")
        return False
    print(f"  ✅ 数据同步完成 ({elapsed:.0f}s)")
    return True
    return True


# ── 扫描 ──

def find_pending_factors(report_filter: str | None, force: bool) -> list[dict]:
    """
    返回待处理因子列表，每项含 {report, factor, code_path, output_dir, meta_path, parquet_path, status}
    status: "pending" (无 parquet), "stale" (有 parquet 但日期老), "current" (已最新)
    """
    if not OUTPUT_BASE.exists():
        return []

    factors = []
    trade_dates = None  # 延迟加载

    report_dirs = sorted(d for d in OUTPUT_BASE.iterdir() if d.is_dir())
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

    # 2. 执行子进程
    code_text = code_path.read_text(encoding="utf-8")
    start_date_str = last_date.strftime("%Y-%m-%d")
    result_parquet = run_factor_subprocess(
        code_text, factor_name, FULL_DATA_DIR,
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
    parser.add_argument("--remote", action="store_true", help="启用远程挂载（数据始终先同步，无需此参数）")
    args = parser.parse_args()

    t_start = time.time()

    # ── Step 1: 挂载（仅 --remote 时） ──
    if args.remote:
        ensure_mounted()
    else:
        global OUTPUT_BASE, FULL_DATA_DIR
        OUTPUT_BASE = LOCAL_FULL
        print("📌 本地模式，跳过远程挂载")

    # ── Step 2: 同步数据（始终先更新数据，再算因子） ──
    sync_data()

    # ── Step 3: 确定日期子目录（默认当天） ──
    date_str = args.subdir or datetime.now().strftime("%Y%m%d")
    dated_base = OUTPUT_BASE / date_str
    if dated_base.exists():
        OUTPUT_BASE = dated_base
        print(f"📅 扫描子目录: {date_str}")
    else:
        print(f"⚠️ 子目录不存在: {dated_base}，回退到根目录")

    # ── Step 4: 扫描因子 ──
    pending = find_pending_factors(args.report, args.force)

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

    # ── Step 4: 执行 ──
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

    # ── 汇总 ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"🏁 完成: {success_count} 成功, {fail_count} 失败, {skipped_count} 跳过 (耗时 {elapsed/60:.1f}min)")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
