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
  python scripts/run_all.py                        # 默认流程
  python scripts/run_all.py --report 研报名        # 只跑指定研报
  python scripts/run_all.py --force                # 强制重跑（无视状态）
  python scripts/run_all.py --workers 3            # 并行数
  python scripts/run_all.py --dry-run              # 只打印计划，不执行
  python scripts/run_all.py --skip-sync            # 跳过数据同步
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# ── 路径 ──
REMOTE_MOUNT = Path("/mnt/remote_e")
REMOTE_FULL = REMOTE_MOUNT / "paper_factors" / "文献因子_全量"
LOCAL_FULL = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"
FULL_BASE = REMOTE_FULL if REMOTE_FULL.exists() else LOCAL_FULL

# 全量输出 + 数据目录
OUTPUT_BASE = FULL_BASE
FULL_DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data")))

# sync_data.py 路径
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "sync_data.py"

# ── 远程挂载 ──

def ensure_mounted() -> bool:
    """确保远程E盘已挂载，返回是否成功"""
    if REMOTE_MOUNT.exists() and any(REMOTE_MOUNT.iterdir()):
        return True
    print("📌 挂载远程E盘...")
    os.makedirs(str(REMOTE_MOUNT), exist_ok=True)
    uid = os.getuid()
    gid = os.getgid()
    ret = os.system(
        f"sudo mount -t cifs //192.168.1.13/E {REMOTE_MOUNT} "
        f"-o user=pc,password=123456,uid={uid},gid={gid},"
        f"file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"
    )
    if ret != 0:
        print("  ⚠️ 挂载失败，使用本地数据")
        return False
    print("  ✅ 已挂载")
    return True


# ── 数据同步 ──

def sync_data() -> bool:
    """运行 sync_data.py 同步最新数据"""
    if not SYNC_SCRIPT.exists():
        print("  ⚠️ sync_data.py 不存在，跳过数据同步")
        return False
    print("📌 同步最新数据...")
    ret = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT)],
        capture_output=True, text=True, timeout=3600,
    )
    for line in ret.stdout.split("\n"):
        line = line.strip()
        if line:
            print(f"  {line}")
    if ret.returncode != 0:
        print(f"  ⚠️ 数据同步异常，继续执行")
        return False
    print("  ✅ 数据同步完成")
    return True


# ── 工具函数 ──

def load_trade_dates() -> list[str]:
    """从数据目录读取交易日列表"""
    for p in [
        FULL_DATA_DIR / "stock_data" / "daily" / "trade_dates.json",
        FULL_DATA_DIR / "stock_data" / "minute_by_date" / "trade_dates.json",
    ]:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"trade_dates.json 未找到 (搜索: {FULL_DATA_DIR})")


def detect_factor_type(code_text: str) -> str:
    if any(k in code_text for k in ('MINUTE_BY_DATE_DIR', 'minute_pv', 'calc_factors_one_day')):
        if 'cross_section' in code_text.lower() or 'calc_factor_minute_raw' in code_text:
            return "minute_cross_section"
        return "minute"
    if 'cross_section' in code_text.lower() or 'calc_factor_cross_section' in code_text:
        return "cross_section"
    return "daily"


def inject_incremental_patch(code_text: str, last_date: str) -> str:
    """在 .code.py 中注入增量日期过滤代码"""
    pattern = re.compile(r'^(LOOKBACK_DAYS\s*=\s*.+)', re.MULTILINE)
    match = pattern.search(code_text)
    if not match:
        print("    ⚠️ 未找到 LOOKBACK_DAYS，跳过 patch 注入")
        return code_text

    patch = f"""

# ── 增量更新 patch（由 run_all.py 自动注入）──
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
                    trade_dates = load_trade_dates()
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

    # 2. 注入 patch → 写入临时文件 → 执行
    code_text = code_path.read_text(encoding="utf-8")
    patched_code = inject_incremental_patch(code_text, last_date.strftime("%Y-%m-%d"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_code = tmpdir / "factor.py"
        tmp_code.write_text(patched_code, encoding="utf-8")

        env = {k: str(v) for k, v in os.environ.items()}
        env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        env["FACTOR_INCREMENTAL_START_DATE"] = last_date.strftime("%Y-%m-%d")

        factor_type = detect_factor_type(code_text)
        if factor_type == "daily":
            env.setdefault("FACTOR_N_WORKERS", "4")
        else:
            env.setdefault("FACTOR_N_WORKERS", "8")

        print(f"  执行中... (start={last_date.strftime('%Y-%m-%d')}, type={factor_type})")
        try:
            proc = subprocess.run(
                [sys.executable, "factor.py"],
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
                print(f"  ❌ 执行失败: {stderr.strip()}")
                result["status"] = "failed"
                result["error"] = stderr.strip()
                return result
        except subprocess.TimeoutExpired:
            print("  ❌ 执行超时（2h）")
            result["status"] = "failed"
            result["error"] = "timeout"
            return result
        except Exception as e:
            print(f"  ❌ 执行异常: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

        # 3. 读取 result.parquet，裁掉重叠，合并
        result_parquet = tmpdir / "result.parquet"
        if not result_parquet.exists():
            print("  ❌ 未生成 result.parquet")
            result["status"] = "failed"
            result["error"] = "no result.parquet"
            return result

        new_df = pd.read_parquet(result_parquet)
        # 只保留 date > last_date
        new_df = new_df[new_df.index > last_date]
        if new_df.empty:
            print("  ⚠️ 增量结果为空")
            result["status"] = "skipped"
            return result

        # 备份 + 合并
        bak_path = parquet_path.with_suffix(f".parquet.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(parquet_path, bak_path)

        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        combined.to_parquet(parquet_path)

        print(f"  ✅ 合并完成: {combined.shape[0]} 行 ({len(new_df)} 新增)")

    # 4. 评估 + 绘图
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.evaluate_factor import evaluate_factor as _eval_fn, load_full_data_label
        from scripts.plot_decile import plot_decile_returns as _plot_fn

        print(f"  评估中...", flush=True)
        _label_df = load_full_data_label(FULL_DATA_DIR)
        _eval_result = _eval_fn(combined, FULL_DATA_DIR, label_df=_label_df)

        if _eval_result and "error" not in _eval_result:
            ic = _eval_result.get("ic_mean", float("nan"))
            ir = _eval_result.get("ic_ir", "N/A")
            ric = _eval_result.get("rank_ic_mean", float("nan"))
            ls = _eval_result.get("long_short_mean", None)
            lss = _eval_result.get("long_short_sharpe", None)
            print(f"    IC={ic:.6f}  IR={ir}  RankIC={ric:.6f}"
                  f"{'' if ls is None else f'  多空={ls:.4%}'}"
                  f"{'' if lss is None else f'  Sharpe={lss:.2f}'}")

            print(f"  生成图表...", flush=True)
            _png_path = output_dir / f"{factor_name}.decile.png"
            _plot_fn(combined, _label_df, factor_name, str(_png_path))
            print(f"  图表已保存: {_png_path}")
        else:
            err = _eval_result.get("error", "未知错误") if _eval_result else "评估返回空"
            print(f"  ⚠️ 评估异常: {err}")
            _eval_result = {}
    except Exception as e:
        print(f"  ⚠️ 评估/绘图失败: {e}")
        _eval_result = {}
        import traceback
        traceback.print_exc()

    # 5. 更新 meta.json
    meta = {}
    meta_path = output_dir / f"{factor_name}.meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta["date_range"] = f"{combined.index.min().strftime('%Y-%m-%d')} ~ {combined.index.max().strftime('%Y-%m-%d')}"
    meta["rows"] = combined.shape[0]
    meta["stock_count"] = combined.shape[1]
    meta["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if _eval_result and "error" not in _eval_result:
        meta["evaluation"] = _eval_result
    meta["daily_update"] = True
    meta["pipeline_status"] = "completed"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  ✅ [增量] {report_name}/{factor_name} 完成")
    result["status"] = "success"
    result["new_dates"] = len(new_df)
    return result


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser(description="一键全量流水线：挂载 → 同步数据 → 全量/增量补算因子")
    parser.add_argument("--report", help="指定研报名 (模糊匹配)", default=None)
    parser.add_argument("--force", action="store_true", help="强制重跑（无视状态）")
    parser.add_argument("--workers", type=int, default=1, help="并行 worker 数 (默认: 1)")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待跑因子，不执行")
    parser.add_argument("--skip-sync", action="store_true", help="跳过数据同步")
    parser.add_argument("--skip-mount", action="store_true", help="跳过挂载检查")
    args = parser.parse_args()

    t_start = time.time()

    # ── Step 1: 挂载 ──
    if not args.skip_mount:
        ensure_mounted()

    # ── Step 2: 同步数据 ──
    if not args.skip_sync:
        sync_data()

    # ── Step 3: 扫描因子 ──
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
