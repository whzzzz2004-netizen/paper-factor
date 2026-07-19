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
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
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


def load_trade_dates() -> list[str]:
    """从全量数据读取 trade_dates.json"""
    # 优先尝试 daily 目录，fallback 到 minute_by_date
    for p in [
        FULL_DATA_DIR / "stock_data" / "daily" / "trade_dates.json",
        FULL_DATA_DIR / "stock_data" / "minute_by_date" / "trade_dates.json",
    ]:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("trade_dates.json not found")


def detect_factor_type(code_text: str) -> str:
    if any(k in code_text for k in ('MINUTE_BY_DATE_DIR', 'minute_pv', 'calc_factors_one_day')):
        if 'cross_section' in code_text.lower() or 'calc_factor_minute_raw' in code_text:
            return "minute_cross_section"
        return "minute"
    if 'cross_section' in code_text.lower() or 'calc_factor_cross_section' in code_text:
        return "cross_section"
    return "daily"


def inject_incremental_patch(code_text: str) -> str:
    """
    在 .code.py 中注入增量日期过滤代码。
    在 TRADE_DATES = json.load(...) 行之后插入 patch。
    """
    # Pattern: LOOKBACK_DAYS = ...  (LOOKBACK_DAYS 总在 TRADE_DATES 之后)
    # 在 LOOKBACK_DAYS 赋值行之后注入 patch，确保 LOOKBACK_DAYS 已可用
    pattern = re.compile(r'^(LOOKBACK_DAYS\s*=\s*.+)', re.MULTILINE)

    match = pattern.search(code_text)
    if not match:
        print("    ⚠️ 未找到 LOOKBACK_DAYS 赋值，跳过 patch 注入")
        return code_text

    patch = """

# ── 增量更新 patch（由 daily_update.py 自动注入）──
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _start_dt = pd.Timestamp(_INC_START)
    _td_idx = pd.DatetimeIndex(TRADE_DATES)
    _start_pos = max(0, _td_idx.searchsorted(_start_dt) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_start_pos:]
# ── patch end ──
"""

    pos = match.end()
    return code_text[:pos] + patch + code_text[pos:]


def find_factor_source(report: str, factor_name: str) -> tuple[Path | None, Path | None]:
    """
    在全量输出目录中查找因子的 .code.py 和 .parquet。
    """
    # 搜索多个可能的位置
    candidates = [
        FULL_OUTPUT / report / factor_name / f"{factor_name}.code.py",
        FULL_OUTPUT / report / factor_name / f"{factor_name}.parquet",
    ]
    # 也搜索 literature_reports
    lit = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
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
        full_code, full_parquet = find_factor_source(report, factor_name)
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
        trade_dates = load_trade_dates()
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

    # 7. 注入 patch → 写入临时文件 → 执行
    code_text = code_path.read_text(encoding="utf-8")
    patched_code = inject_incremental_patch(code_text)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_code = tmpdir / f"{factor_name}.py"
        tmp_code.write_text(patched_code, encoding="utf-8")

        env = {k: str(v) for k, v in os.environ.items()}
        env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        env["FACTOR_INCREMENTAL_START_DATE"] = last_date.strftime("%Y-%m-%d")
        factor_type = detect_factor_type(code_text)
        env.setdefault("FACTOR_N_WORKERS", "8")

        print(f"  执行中... (start={last_date.strftime('%Y-%m-%d')}, type={factor_type})")
        try:
            proc = subprocess.run(
                [sys.executable, f"{factor_name}.py"],
                cwd=tmpdir,
                capture_output=True, text=True, timeout=7200,
                env=env,
            )
            for line in proc.stdout.split("\n"):
                line = line.strip()
                if line:
                    print(f"    {line}")
            if proc.returncode != 0:
                stderr = proc.stderr[-500:] if len(proc.stderr) > 500 else proc.stderr
                result["status"] = "error"
                result["error"] = f"执行失败: {stderr.strip()}"
                return result
        except subprocess.TimeoutExpired:
            result["status"] = "error"
            result["error"] = "执行超时（2h）"
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            return result

        # 8. 读取 .code.py 输出的 parquet（文件名 = 因子名），裁掉 lookback，合并
        result_parquet = tmpdir / f"{factor_name}.parquet"
        if not result_parquet.exists():
            result["status"] = "error"
            result["error"] = f"未生成 {factor_name}.parquet"
            return result

        new_df = pd.read_parquet(result_parquet)
        # 统一 index 为 DatetimeIndex（.code.py 可能输出字符串索引）
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        if not isinstance(existing_df.index, pd.DatetimeIndex):
            existing_df.index = pd.to_datetime(existing_df.index)
        # 只保留 date > last_date 的行
        new_df = new_df[new_df.index > last_date]
        if new_df.empty:
            result["status"] = "error"
            result["error"] = "增量结果为空（可能 lookback 不足）"
            return result

        # 合并：已有 + 新增
        # 先备份
        bak_path = daily_parquet.with_suffix(f".parquet.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(daily_parquet, bak_path)

        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        combined.to_parquet(daily_parquet)

        print(f"  合并完成: {combined.shape[0]} 天 ({len(new_df)} 新增)")

    # 9. 更新 meta.json
    result["status"] = "success"
    result["new_dates"] = len(new_df)
    result["total_dates"] = combined.shape[0]

    meta_path = daily_dir / f"{factor_name}.meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta["date_range"] = f"{combined.index.min().strftime('%Y-%m-%d')} ~ {combined.index.max().strftime('%Y-%m-%d')}"
    meta["rows"] = combined.shape[0]
    meta["stock_count"] = combined.shape[1]
    meta["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    meta["daily_update"] = True
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 10. 评估 + 绘图（除非跳过）
    if not skip_eval:
        eval_script = PROJECT_ROOT / "scripts" / "evaluate_factor.py"
        if eval_script.exists():
            try:
                eval_result = subprocess.run(
                    [sys.executable, str(eval_script), str(daily_parquet),
                     "--data-dir", str(FULL_DATA_DIR)],
                    capture_output=True, text=True, timeout=600,
                )
                if eval_result.returncode == 0:
                    # evaluate_factor.py 会更新 meta.json
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    for line in eval_result.stdout.split("\n"):
                        if any(k in line for k in ("IC (Pearson)", "Rank IC", "Sharpe")):
                            print(f"    {line.strip()}")
                else:
                    print(f"    ⚠️ 评估失败")
            except Exception as e:
                print(f"    ⚠️ 评估异常: {e}")

        plot_script = PROJECT_ROOT / "scripts" / "plot_decile.py"
        plot_output = daily_dir / f"{factor_name}.decile.png"
        if plot_script.exists():
            try:
                plot_result = subprocess.run(
                    [sys.executable, str(plot_script), str(daily_parquet),
                     "--data-dir", str(FULL_DATA_DIR), "--output", str(plot_output)],
                    capture_output=True, text=True, timeout=600,
                )
                if plot_result.returncode == 0:
                    print(f"    ✅ 图表已更新")
                else:
                    print(f"    ⚠️ 图表生成失败")
            except Exception as e:
                print(f"    ⚠️ 图表异常: {e}")

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

    print(f"  ✅ [{factor_key}] 更新完成")
    return result


def scan_all_factors() -> list[str]:
    """扫描文献因子_全量/ 下所有已有 .parquet 的因子"""
    factors = []
    if FULL_OUTPUT.exists():
        for report_dir in sorted(FULL_OUTPUT.iterdir()):
            if not report_dir.is_dir():
                continue
            for factor_dir in sorted(report_dir.iterdir()):
                if not factor_dir.is_dir():
                    continue
                parquet = factor_dir / f"{factor_dir.name}.parquet"
                if parquet.exists():
                    factors.append(f"{report_dir.name}/{factor_dir.name}")
    return factors


def main():
    parser = argparse.ArgumentParser(description="因子每日增量更新")
    parser.add_argument("--factor", help="单个因子 (格式: report/factor_name)")
    parser.add_argument("--dry-run", action="store_true", help="只检查，不执行")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估和绘图")
    parser.add_argument("--skip-sync", action="store_true", help="跳过远程同步")
    parser.add_argument("--workers", type=int, default=3, help="并行 worker 数（默认3）")
    parser.add_argument("--report", default=None, help="只更新指定研报（模糊匹配）")
    args = parser.parse_args()

    if args.factor:
        factors = [args.factor]
    else:
        factors = scan_all_factors()
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
            r = update_factor(f, dry_run=args.dry_run, skip_eval=args.skip_eval, skip_sync=args.skip_sync)
            results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, f in enumerate(factors):
                _write_status(i + 1, len(factors), f"提交 {f}")
                fut = executor.submit(
                    update_factor, f, args.dry_run, args.skip_eval, args.skip_sync
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
