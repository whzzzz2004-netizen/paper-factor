#!/usr/bin/env python3
"""
因子每日增量更新脚本。

用法:
  python scripts/daily_update.py                  # 更新所有 enabled 因子
  python scripts/daily_update.py --factor idea__0/MorningVolumeRatio  # 单个
  python scripts/daily_update.py --dry-run        # 只检查，不执行
  python scripts/daily_update.py --skip-eval      # 跳过评估/绘图
  python scripts/daily_update.py --skip-sync      # 跳过远程同步
  python scripts/daily_update.py --workers 5       # 并行数（默认3）

机制:
  1. 读增量 parquet → last_date（首次: 从全量目录复制原始 parquet）
  2. 读 trade_dates.json → latest_date
  3. 若 latest_date <= last_date → 跳过（已最新）
  4. 复制 .code.py → 注入日期过滤 patch
  5. 设 FACTOR_INCREMENTAL_START_DATE → subprocess 执行
  6. 裁掉 lookback 重叠行 → concat 到增量 parquet
  7. 评估 + 绘图 + 同步远程
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
)

PROJECT_ROOT = Path(__file__).parent.parent

SMB_HOST = "192.168.1.13"
SMB_SHARE = "E"
SMB_USER = "pc"
SMB_PASS = "123456"
CIFS_MOUNT = Path("/mnt/remote_e")


def _sudo_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """执行 sudo 命令，自动处理 TTY 密码需求（-S piped from stdin）"""
    if "PYTHON_RUN_AS_ROOT" in os.environ:
        return subprocess.run(cmd, **kwargs)
    try:
        return subprocess.run(["sudo", "-n"] + cmd, **kwargs)
    except Exception:
        pass
    kwargs.pop("input", None)
    return subprocess.run(
        ["sudo", "-S"] + cmd,
        input=f"{SMB_PASS}\n".encode(),
        **kwargs,
    )


def _ensure_remote_mounted() -> bool:
    """自动挂载远程 E 盘"""
    if CIFS_MOUNT.exists() and any(CIFS_MOUNT.iterdir()):
        return True
    try:
        CIFS_MOUNT.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    r = _sudo_run(
        ["mount", "-t", "cifs", f"//{SMB_HOST}/{SMB_SHARE}", str(CIFS_MOUNT),
         "-o", f"user={SMB_USER},password={SMB_PASS},uid={os.getuid()},gid={os.getgid()},file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"],
        capture_output=True, text=True, timeout=30,
    )
    return r.returncode == 0


# 优先用远程（CIFS 挂载），让 .code.py 等文件变更即时可见
_REMOTE = Path("/mnt/remote_e/paper_factors/文献因子_全量")
FULL_OUTPUT = _REMOTE if _REMOTE.exists() else PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"
_REMOTE_DAILY = Path("/mnt/remote_e/paper_factors/文献因子_每日更新")
DAILY_UPDATE_DIR = _REMOTE_DAILY if _REMOTE_DAILY.exists() else PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_每日更新"
def _detect_data_dir() -> Path:
    candidates = [
        os.environ.get("FACTOR_DATA_DIR", ""),
        os.environ.get("RDAGENT_FACTOR_DATA_DIR", ""),
        str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"),
        "/mnt/remote_e/_paper_factor_unified/factor_implementation_source_data",
        "E:\\_paper_factor_unified\\factor_implementation_source_data",
        "Z:\\_paper_factor_unified\\factor_implementation_source_data",
        "\\\\192.168.1.13\\E\\_paper_factor_unified\\factor_implementation_source_data",
    ]
    for p in candidates:
        if p and (Path(p) / "stock_data" / "daily").exists():
            return Path(p)
    print("  ⏳ 未找到数据目录，尝试自动挂载远程 E 盘...")
    if _ensure_remote_mounted():
        for p in candidates:
            if p and (Path(p) / "stock_data" / "daily").exists():
                return Path(p)
    return Path(".")
FULL_DATA_DIR = _detect_data_dir()
CONFIG_PATH = PROJECT_ROOT / "git_ignore_folder" / "daily_update_config.json"
STATUS_PATH = PROJECT_ROOT / "git_ignore_folder" / "daily_update_status.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"enabled": [], "history": []}


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def find_factor_source(report: str, factor_name: str, date_subdir: str | None = None) -> tuple[Path | None, Path | None]:
    """
    在全量输出目录中查找因子的 .code.py 和 .parquet。
    """
    full_base = (FULL_OUTPUT / date_subdir) if date_subdir else FULL_OUTPUT
    # 搜索多个可能的位置
    candidates = [
        full_base / report / factor_name / f"{factor_name}.code.py",
        full_base / report / factor_name / f"{factor_name}.parquet",
    ]
    # 也搜索 literature_reports
    lit = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
    if date_subdir:
        lit = lit / date_subdir
    candidates.extend([
        lit / report / factor_name / f"{factor_name}.code.py",
        lit / report / factor_name / f"{factor_name}.parquet",
    ])
    return (
        next((p for p in candidates[::2] if p.exists()), None),   # code.py at 0, 2
        next((p for p in candidates[1::2] if p.exists()), None), # parquet at 1, 3
    )


def update_factor(
    factor_key: str,
    dry_run: bool = False,
    skip_eval: bool = False,
    skip_sync: bool = False,
    date_subdir: str | None = None,
) -> dict:
    """
    更新单个因子。返回状态 dict。
    """
    parts = factor_key.split("/")
    if len(parts) != 2:
        return {"factor": factor_key, "status": "error", "error": "格式错误，应为 report/factor"}

    report, factor_name = parts
    result = {"factor": factor_key, "status": "pending", "report": report, "name": factor_name}

    # 1. 确定增量输出目录
    daily_dir = DAILY_UPDATE_DIR / report / factor_name
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_parquet = daily_dir / f"{factor_name}.parquet"

    # 2. 首次：从全量复制（非 dry-run 时才实际复制）
    if not daily_parquet.exists():
        full_code, full_parquet = find_factor_source(report, factor_name, date_subdir)
        if full_parquet is None:
            result["status"] = "error"
            result["error"] = "全量 parquet 不存在，无法初始化"
            return result
        if dry_run:
            print(f"  [dry-run] 将从全量复制: {full_parquet.name}")
            # dry-run 时直接读全量 parquet 获取 last_date
            read_parquet = full_parquet
        else:
            print(f"  首次初始化：复制全量 parquet → 增量目录")
            shutil.copy2(full_parquet, daily_parquet)
            if full_code:
                shutil.copy2(full_code, daily_dir / f"{factor_name}.code.py")
            read_parquet = daily_parquet
    else:
        read_parquet = daily_parquet

    # 3. 读 parquet → last_date
    try:
        existing_df = pd.read_parquet(read_parquet)
        last_date = pd.Timestamp(existing_df.index.max())
        result["last_date"] = last_date.strftime("%Y-%m-%d")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"读取 parquet 失败: {e}"
        return result

    # 4. 读 trade_dates → latest_date
    try:
        trade_dates = load_trade_dates(FULL_DATA_DIR)
        latest_date = pd.Timestamp(trade_dates[-1])
        result["latest_date"] = latest_date.strftime("%Y-%m-%d")
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"读取 trade_dates 失败: {e}"
        return result

    # 5. 比较
    if latest_date <= last_date:
        result["status"] = "up_to_date"
        print(f"  [{factor_key}] 已最新 ({last_date.strftime('%Y-%m-%d')})")
        return result

    result["needs_update"] = True
    n_new = (latest_date - last_date).days  # approximate
    print(f"  [{factor_key}] 需更新: {last_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')}")

    if dry_run:
        result["status"] = "dry_run"
        return result

    # 6. 找 .code.py
    code_candidates = [
        daily_dir / f"{factor_name}.code.py",
        FULL_OUTPUT / report / factor_name / f"{factor_name}.code.py",
    ]
    lit = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
    code_candidates.append(lit / report / factor_name / f"{factor_name}.code.py")
    code_path = next((p for p in code_candidates if p.exists()), None)
    if code_path is None:
        result["status"] = "error"
        result["error"] = ".code.py 不存在"
        return result

    # 7. 执行子进程
    code_text = code_path.read_text(encoding="utf-8")
    start_date_str = last_date.strftime("%Y-%m-%d")
    result_parquet = run_factor_subprocess(
        code_text, factor_name, FULL_DATA_DIR,
        start_date=start_date_str, n_workers=8, timeout=7200,
    )
    if result_parquet is None:
        result["status"] = "error"
        result["error"] = "子进程执行失败"
        return result

    # 8. 读取结果，裁掉重叠，合并
    result_df = pd.read_parquet(result_parquet)
    result_parquet.unlink(missing_ok=True)

    combined = merge_incremental_result(existing_df, result_df, last_date)
    if combined is existing_df:
        result["status"] = "error"
        result["error"] = "增量结果为空（可能 lookback 不足）"
        return result

    # 备份 + 写回
    backup_parquet(daily_parquet)
    combined.to_parquet(daily_parquet)

    print(f"  合并完成: {combined.shape[0]} 天 ({combined.shape[0] - len(existing_df)} 新增)")

    # 9. 更新 meta.json
    result["status"] = "success"
    result["new_dates"] = combined.shape[0] - len(existing_df)
    result["total_dates"] = combined.shape[0]

    meta_path = daily_dir / f"{factor_name}.meta.json"
    update_factor_meta(meta_path, combined, extra={"pipeline_status": "completed"})

    # 10. 评估 + 绘图（除非跳过）
    if not skip_eval:
        evaluate_factor(daily_parquet, factor_name, daily_dir, FULL_DATA_DIR)

    # 11. 同步远程
    if not skip_sync:
        try:
            from scripts.sync_utils import ensure_remote_writable, upload_tree, REMOTE_BASE_DAILY
            if ensure_remote_writable():
                remote_prefix = f"{REMOTE_BASE_DAILY}\\{report}\\{factor_name}"
                n = upload_tree(daily_dir, remote_prefix)
                print(f"    ✅ 远程同步: {n} 个文件")
            else:
                print(f"    ⚠️ 远程不可用，跳过同步")
        except Exception as e:
            print(f"    ⚠️ 远程同步失败: {e}")

    # 12. 动态清理：删除写回时创建的 .parquet.bak 备份，
    #     让因子目录看着就和全新计算的一样（无中间产物残留）
    cleanup_parquet_backup(daily_parquet)

    print(f"  ✅ [{factor_key}] 更新完成")
    return result


def scan_all_factors(date_subdir: str | None = None) -> list[str]:
    """扫描文献因子_全量/ 下所有已有 .parquet 的因子"""
    factors = []
    scan_base = (FULL_OUTPUT / date_subdir) if date_subdir else FULL_OUTPUT
    if scan_base.exists():
        for report_dir in sorted(scan_base.iterdir()):
            if not report_dir.is_dir():
                continue
            for factor_dir in sorted(report_dir.iterdir()):
                if not factor_dir.is_dir():
                    continue
                parquet = factor_dir / f"{factor_dir.name}.parquet"
                if parquet.exists():
                    prefix = f"{date_subdir}/" if date_subdir else ""
                    factors.append(f"{prefix}{report_dir.name}/{factor_dir.name}")
    return factors


def main():
    parser = argparse.ArgumentParser(description="因子每日增量更新")
    parser.add_argument("--factor", help="单个因子 (格式: report/factor_name)")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不执行")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估和绘图")
    parser.add_argument("--skip-sync", action="store_true", help="跳过远程同步")
    parser.add_argument("--workers", type=int, default=3, help="并行 worker 数（默认3）")
    parser.add_argument("--report", default=None, help="只更新指定研报（模糊匹配）")
    parser.add_argument("--date", type=str, default=None, help="指定日期子目录 (YYYYMMDD)，如 --date 20260726")
    args = parser.parse_args()

    if args.factor:
        factors = [args.factor]
    else:
        factors = scan_all_factors(args.date)
        if args.report:
            factors = [f for f in factors if args.report.lower() in f.lower()]

    if not factors:
        print("未找到任何全量因子。")
        return 0

    print(f"{'='*50}")
    print(f"因子每日增量更新")
    print(f"{'='*50}")
    print(f"因子数: {len(factors)}")
    print(f"模式: {'DRY RUN' if args.dry_run else '执行'}")
    print(f"并行: {args.workers}")
    print()

    results = []
    start_time = datetime.now()

    def _write_status(current, total, msg=""):
        status = {
            "running": True,
            "current": current,
            "total": total,
            "message": msg,
            "started_at": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")

    if args.workers <= 1 or args.dry_run:
        for i, f in enumerate(factors):
            _write_status(i + 1, len(factors), f"处理 {f}")
            r = update_factor(f, dry_run=args.dry_run, skip_eval=args.skip_eval, skip_sync=args.skip_sync, date_subdir=args.date)
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, f in enumerate(factors):
                _write_status(i + 1, len(factors), f"提交 {f}")
                fut = executor.submit(
                    update_factor, f, args.dry_run, args.skip_eval, args.skip_sync, args.date
                )
                futures[fut] = f

            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                except Exception as e:
                    results.append({"factor": fname, "status": "error", "error": str(e)})

    # 写入最终状态
    status = {
        "running": False,
        "results": results,
        "started_at": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")

    # 汇总
    print(f"\n{'='*50}")
    print(f"更新汇总")
    print(f"{'='*50}")
    success = sum(1 for r in results if r["status"] == "success")
    up_to_date = sum(1 for r in results if r["status"] == "up_to_date")
    error = sum(1 for r in results if r["status"] == "error")
    print(f"  成功: {success}  已最新: {up_to_date}  失败: {error}")
    for r in results:
        if r["status"] == "error":
            print(f"  ❌ {r['factor']}: {r.get('error', 'unknown')}")

    return 1 if error > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
