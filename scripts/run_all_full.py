#!/usr/bin/env python3
"""
统一入口：同步数据 → 全量计算 + 增量更新。

步骤:
  1. 自动同步最新原始数据（日线/分钟线从远程 E 盘下载，转为因子所需格式）
  2. 逐个扫描因子：
     - 已有 parquet → 增量更新（只算新天数，合并到原文件）
     - 无 parquet   → 全量计算（从零开始）

用法:
  python scripts/run_all_full.py [--report <name>] [--force]

说明:
  --report    只跑指定研报下的因子 (模糊匹配)
  --force     强制全量重跑（即使已有 parquet）
  --dry-run   只列出因子，不执行
  --skip-sync 跳过数据同步步骤
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
_REMOTE_OUTPUTS = [
    Path("/mnt/remote_e/paper_factors/文献因子_全量"),
    Path("E:\\paper_factors\\文献因子_全量"),
    Path("Z:\\paper_factors\\文献因子_全量"),
]
FULL_BASE = next((p for p in _REMOTE_OUTPUTS if p.exists()),
                 PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量")

SMB_HOST = "192.168.1.13"
SMB_SHARE = "E"
SMB_USER = "pc"
SMB_PASS = "123456"
CIFS_MOUNT = Path("/mnt/remote_e")


def _sudo_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
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
    if CIFS_MOUNT.exists() and any(CIFS_MOUNT.iterdir()):
        return True
    print(f"  ⏳ 自动挂载远程 E 盘 {SMB_HOST}/{SMB_SHARE} → {CIFS_MOUNT} ...")
    try:
        CIFS_MOUNT.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"  ❌ 无法创建挂载点 {CIFS_MOUNT}: {e}")
        return False
    _sudo_run(["modprobe", "cifs"], capture_output=True, timeout=10)
    _sudo_run(["apt", "install", "-y", "cifs-utils"], capture_output=True, timeout=120)
    _base_opts = f"user={SMB_USER},password={SMB_PASS},uid={os.getuid()},gid={os.getgid()},file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"
    for _vers in ("3.0", "2.1", "2.0", "1.0"):
        r = _sudo_run(
            ["mount", "-t", "cifs", f"//{SMB_HOST}/{SMB_SHARE}", str(CIFS_MOUNT), "-o", f"vers={_vers},{_base_opts}"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"  ✅ 已挂载远程 E 盘 (vers={_vers})")
            return True
        _err = (r.stderr or r.stdout).strip()
        if _err:
            print(f"  ⚠️ vers={_vers} 失败: {_err[:200]}")
    print(f"  ❌ 所有版本均挂载失败。")
    print(f"  💡 手工命令:")
    print(f"    sudo mkdir -p /mnt/remote_e && sudo mount -t cifs //{SMB_HOST}/{SMB_SHARE} /mnt/remote_e -o vers=3.0,{_base_opts}")
    return False


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
if FULL_DATA_DIR != Path("."):
    os.environ["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
    os.environ["RDAGENT_FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)


# ── 工具函数 ──

def load_trade_dates() -> list[str]:
    for p in [
        FULL_DATA_DIR / "stock_data" / "daily" / "trade_dates.json",
        FULL_DATA_DIR / "stock_data" / "minute_by_date" / "trade_dates.json",
    ]:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("trade_dates.json not found")


def inject_incremental_patch(code_text: str) -> str:
    """在 LOOKBACK_DAYS 赋值后注入增量日期过滤 patch"""
    pattern = re.compile(r'^(LOOKBACK_DAYS\s*=\s*.+)', re.MULTILINE)
    match = pattern.search(code_text)
    if not match:
        print("    ⚠️ 未找到 LOOKBACK_DAYS 赋值，跳过 patch 注入")
        return code_text
    patch = """

# ── 增量更新 patch ──
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


def detect_factor_type(code_text: str) -> str:
    if any(k in code_text for k in ('MINUTE_BY_DATE_DIR', 'minute_pv', 'calc_factors_one_day')):
        return "minute" if 'cross_section' not in code_text.lower() else "minute_cross_section"
    if 'cross_section' in code_text.lower() or 'calc_factor_cross_section' in code_text:
        return "cross_section"
    return "daily"


# ── 扫描因子 ──

def scan_all_factors(report_filter: str | None) -> list[dict]:
    """扫描所有有 .code.py 的因子"""
    if not FULL_BASE.exists():
        return []
    factors = []
    report_dirs = sorted(d for d in FULL_BASE.iterdir() if d.is_dir())
    if report_filter:
        report_dirs = [d for d in report_dirs if report_filter in d.name]
    for report_dir in report_dirs:
        report_name = report_dir.name
        for factor_dir in sorted(report_dir.iterdir()):
            if not factor_dir.is_dir():
                continue
            factor_name = factor_dir.name
            code_path = factor_dir / f"{factor_name}.code.py"
            if not code_path.exists():
                continue
            factors.append({
                "report": report_name,
                "factor": factor_name,
                "code_path": code_path,
                "output_dir": factor_dir,
                "parquet_path": factor_dir / f"{factor_name}.parquet",
            })
    return factors


# ── 增量更新单个因子 ──

def run_incremental(item: dict) -> bool:
    """已有 parquet → 只算新天数，合并到原文件"""
    factor_name = item["factor"]
    report_name = item["report"]
    code_path = item["code_path"]
    factor_dir = item["output_dir"]
    parquet_path = item["parquet_path"]

    if not parquet_path.exists():
        print(f"  ⚠️ 无已有 parquet，跳过增量")
        return False

    # 读已有 parquet → last_date
    try:
        existing_df = pd.read_parquet(parquet_path)
        last_date = pd.Timestamp(existing_df.index.max())
    except Exception as e:
        print(f"  ❌ 读取已有 parquet 失败: {e}")
        return False

    # 读 trade_dates → latest_date
    try:
        trade_dates = load_trade_dates()
        latest_date = pd.Timestamp(trade_dates[-1])
    except Exception as e:
        print(f"  ❌ 读取 trade_dates 失败: {e}")
        return False

    if latest_date <= last_date:
        print(f"  ✅ 已最新 ({last_date.strftime('%Y-%m-%d')})")
        return True

    n_new = (latest_date - last_date).days
    print(f"  📈 增量更新: {last_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')} (~{n_new}天)")

    # 准备 patched 代码 → temp → 执行
    code_text = code_path.read_text(encoding="utf-8")
    patched = inject_incremental_patch(code_text)

    with tempfile.TemporaryDirectory(prefix=f"{factor_name}_", dir="/tmp") as _tmp:
        tmpdir = Path(_tmp)
        tmp_code = tmpdir / f"{factor_name}.py"
        tmp_code.write_text(patched, encoding="utf-8")

        env = os.environ.copy()
        env["FACTOR_INCREMENTAL_START_DATE"] = last_date.strftime("%Y-%m-%d")
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        env["PYTHONWARNINGS"] = "ignore"
        env["PYTHONUNBUFFERED"] = "1"
        factor_type = detect_factor_type(code_text)
        env.setdefault("FACTOR_N_WORKERS", "4")

        print(f"  执行中... (start={last_date.strftime('%Y-%m-%d')}, type={factor_type})", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_code)],
                cwd=tmpdir, capture_output=True, text=True, timeout=7200, env=env,
            )
            for line in proc.stdout.split("\n"):
                line = line.strip()
                if line:
                    print(f"    {line}")
            if proc.returncode != 0:
                stderr = proc.stderr[-500:] if len(proc.stderr) > 500 else proc.stderr
                print(f"  ❌ 增量计算失败: {stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ❌ 增量计算超时")
            return False
        except Exception as e:
            print(f"  ❌ 增量计算异常: {e}")
            return False

        # 读新数据 → 裁掉 lookback 重叠 → 合并
        result_parquet = tmpdir / f"{factor_name}.parquet"
        if not result_parquet.exists():
            print(f"  ❌ 未生成 parquet")
            return False

        new_df = pd.read_parquet(result_parquet)
        if not isinstance(new_df.index, pd.DatetimeIndex):
            new_df.index = pd.to_datetime(new_df.index)
        if not isinstance(existing_df.index, pd.DatetimeIndex):
            existing_df.index = pd.to_datetime(existing_df.index)
        new_df = new_df[new_df.index > last_date]
        if new_df.empty:
            print(f"  ⚠️ 增量结果为空（可能 lookback 不足）")
            return False

        # 备份 → 合并 → 写回
        bak = parquet_path.with_suffix(f".parquet.bak.{_time.strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(parquet_path, bak)
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        combined.to_parquet(parquet_path)
        print(f"  ✅ 合并完成: {combined.shape[0]} 天 ({len(new_df)} 新增)")

    return True


# ── 全量计算单个因子 ──

def run_full(item: dict) -> bool:
    """无 parquet → 全量计算"""
    factor_name = item["factor"]
    code_path = item["code_path"]
    output_dir = item["output_dir"]

    sys.path.insert(0, str(PROJECT_ROOT))
    from rdagent.app.qlib_rd_loop.factor_full_pipeline import run_full_pipeline

    return run_full_pipeline(
        factor_name=factor_name,
        code_path=code_path,
        output_dir=output_dir,
        factor_type=None,
        test_meta=None,
        source_excerpt="",
    )


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser(description="统一入口：同步数据 + 全量计算 + 增量更新")
    parser.add_argument("--report", help="指定研报名 (模糊匹配)", default=None)
    parser.add_argument("--force", action="store_true", help="强制全量重跑（跳过增量）")
    parser.add_argument("--dry-run", action="store_true", help="仅列出因子，不执行")
    parser.add_argument("--skip-sync", action="store_true", help="跳过数据同步步骤")
    args = parser.parse_args()

    # ── 步骤1: 同步最新原始数据（每日/分钟线从远程 E 盘下载） ──
    if not args.dry_run and not args.skip_sync:
        print(f"{'='*60}")
        print("步骤1: 同步最新原始数据 → 因子计算所需格式")
        print(f"{'='*60}")
        sync_script = PROJECT_ROOT / "scripts" / "sync_data.py"
        if sync_script.exists():
            try:
                r = subprocess.run(
                    [sys.executable, str(sync_script)],
                    timeout=7200,
                )
                if r.returncode != 0:
                    print("  ⚠️ 数据同步返回非零退出码，继续执行因子计算...")
            except subprocess.TimeoutExpired:
                print("  ⚠️ 数据同步超时，继续执行因子计算...")
            except Exception as e:
                print(f"  ⚠️ 数据同步异常: {e}，继续执行因子计算...")
        else:
            print(f"  ⚠️ 未找到 {sync_script}，跳过数据同步")
    else:
        print("⏭️ 跳过数据同步")

    # ── 步骤2: 扫描并处理因子 ──
    if not FULL_DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {FULL_DATA_DIR}")
        return 1 if not args.dry_run else 0

    factors = scan_all_factors(args.report)
    if not factors:
        print("未找到因子。")
        return 0

    if args.dry_run:
        print(f"共 {len(factors)} 个因子:\n")
        for item in factors:
            mode = "全量" if not item["parquet_path"].exists() else "增量"
            if args.force and item["parquet_path"].exists():
                mode = "强制全量"
            print(f"  [{mode}] {item['report']} / {item['factor']}")
        return 0

    print(f"{'='*60}")
    print(f"因子处理: {len(factors)} 个")
    print(f"模式: {'强制全量' if args.force else '自动(有parquet→增量, 无→全量)'}")
    print(f"{'='*60}\n")

    success, fail = 0, 0
    for item in factors:
        factor_name = item["factor"]
        report_name = item["report"]
        has_parquet = item["parquet_path"].exists()

        print(f"\n{'='*60}")
        if args.force or not has_parquet:
            mode = "全量" if not has_parquet else "强制全量"
            print(f"▶ [{mode}] {report_name} / {factor_name}")
            print(f"{'='*60}\n")
            ok = run_full(item)
        else:
            print(f"▶ [增量] {report_name} / {factor_name}")
            print(f"{'='*60}\n")
            ok = run_incremental(item)

        if ok:
            success += 1
            print(f"  ✅ {report_name}/{factor_name}")
        else:
            fail += 1
            print(f"  ❌ {report_name}/{factor_name}")

    print(f"\n{'='*60}")
    print(f"完成: 成功 {success}, 失败 {fail}")
    print(f"{'='*60}")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
