#!/usr/bin/env python3
"""
一键运行全量目录下所有未完成的因子。
扫描 文献因子_全量/，对没有 .parquet 产出的因子逐个执行 run-full（含评估、图、Barra）。

用法:
  python scripts/run_all_pending_full.py                          # 全部待运行因子
  python scripts/run_all_pending_full.py --report "报告名"        # 只跑某份报告
  python scripts/run_all_pending_full.py --dry-run                # 只列出不运行
  python scripts/run_all_pending_full.py --no-mount               # 不自动挂载远程E盘
  python scripts/run_all_pending_full.py --no-mount --remote-path Z:\\paper_factors\\文献因子_全量  # Windows: 手动映射网络驱动器后指定路径

首次运行会自动尝试挂载远程 E 盘（192.168.1.13），挂载成功则自动设置数据路径。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"

# ── SMB 远程挂载配置 ──
REMOTE_SERVER = "192.168.1.13"
REMOTE_SHARE = "E"
MOUNT_POINT = "/mnt/remote_e"
REMOTE_DATA_PATH = "_paper_factor_unified/factor_implementation_source_data"
REMOTE_FACTOR_DIR = "paper_factors"  # 远程 E 盘上的因子目录


def ensure_mount(max_retries=2):
    """尝试挂载远程 E 盘（SMB）。返回是否挂载成功。"""
    # 已挂载？
    try:
        r = subprocess.run(["mountpoint", "-q", MOUNT_POINT], capture_output=True)
        if r.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # 从 .env 或环境变量读密码
    smb_pass = os.environ.get("SMB_PASS", "123456")
    smb_user = os.environ.get("SMB_USER", "pc")

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(
                ["sudo", "-n", "mount", "-t", "cifs",
                 f"//{REMOTE_SERVER}/{REMOTE_SHARE}", MOUNT_POINT,
                 "-o", (
                     f"user={smb_user},password={smb_pass},"
                     f"uid={os.getuid()},gid=os.getgid(),"
                     f"file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"
                 )],
                capture_output=True, timeout=10,
            )
            # 验证
            r = subprocess.run(["mountpoint", "-q", MOUNT_POINT], capture_output=True)
            if r.returncode == 0:
                return True
        except Exception:
            pass
        if attempt < max_retries:
            print(f"  ⏳ 重试挂载... ({attempt}/{max_retries})", flush=True)
            time.sleep(1)

    return False


def find_pending_factors(report_filter=None):
    """扫描全量目录，返回所有待运行因子列表。"""
    pending = []
    for report_dir in sorted(FULL_BASE.iterdir()):
        if not report_dir.is_dir():
            continue
        if report_filter and report_dir.name != report_filter:
            continue

        for factor_dir in sorted(report_dir.iterdir()):
            if not factor_dir.is_dir():
                continue

            factor_name = factor_dir.name
            code_path = factor_dir / f"{factor_name}.code.py"
            parquet_path = factor_dir / f"{factor_name}.parquet"

            if not code_path.exists():
                continue
            if parquet_path.exists():
                continue

            pending.append({
                "report": report_dir.name,
                "factor": factor_name,
                "code": code_path,
                "output_dir": factor_dir,
            })

    return pending


def run_one(factor_info, idx, total, dry_run=False, output_base=None):
    """对单个因子执行 run-full"""
    report = factor_info["report"]
    factor = factor_info["factor"]
    code = factor_info["code"]

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "claude_factor_helper.py"),
        "run-full",
        "--code", str(code),
        "--factor-name", factor,
        "--report-name", report,
    ]

    # 如果指定了远程输出目录，写到远程
    if output_base:
        output_dir = output_base / report / factor
        cmd += ["--output", str(output_dir)]

    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True

    print(f"\n{'='*60}")
    print(f"[{idx+1}/{total}] {report}/{factor}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✅ 完成 ({elapsed:.0f}s)")
        return True
    else:
        print(f"  ❌ 失败 (exit={result.returncode}, {elapsed:.0f}s)")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="一键运行全量目录下所有未完成的因子"
    )
    parser.add_argument("--report", default=None, help="只跑指定报告（可选）")
    parser.add_argument("--dry-run", action="store_true", help="只列出待运行因子，不实际执行")
    parser.add_argument("--no-mount", action="store_true", help="不自动挂载远程 E 盘")
    parser.add_argument("--remote-path", default=None,
                        help="远程 文献因子_全量 目录路径（Windows: 映射网络驱动器后指定）")
    args = parser.parse_args()

    # ── 第一步：挂载远程 E 盘 ──
    remote_full_base = None
    if not args.no_mount and "FACTOR_DATA_DIR" not in os.environ:
        print("挂载远程 E 盘...", end=" ", flush=True)
        subprocess.run(["mkdir", "-p", MOUNT_POINT], capture_output=True)
        if ensure_mount():
            os.environ["FACTOR_DATA_DIR"] = str(Path(MOUNT_POINT) / REMOTE_DATA_PATH)
            remote_full_base = Path(MOUNT_POINT) / REMOTE_FACTOR_DIR / "文献因子_全量"
            print(f"✅ 远程 E 盘已挂载，扫描: {remote_full_base}")
        else:
            print("⚠️ 远程 E 盘挂载失败，使用本地数据路径")
    elif "FACTOR_DATA_DIR" in os.environ:
        print(f"数据路径: FACTOR_DATA_DIR={os.environ['FACTOR_DATA_DIR']}")
    else:
        print("数据路径: 使用默认路径")

    # ── 第二步：扫描待运行因子（远程优先，本地回退） ──
    # (FULL_BASE 是模块级变量，在 __main__ 块中直接赋值即可)

    if remote_full_base and remote_full_base.exists():
        FULL_BASE = remote_full_base
    elif args.remote_path:
        rp = Path(args.remote_path)
        if rp.exists():
            FULL_BASE = rp
            remote_full_base = rp  # 让 output_base 也指向远程
            print(f"使用远程路径: {FULL_BASE}")
        else:
            print(f"⚠️ 指定的远程路径不存在: {rp}")
    pending = find_pending_factors(report_filter=args.report)

    if not pending:
        print("\n全量目录下所有因子已完成，没有待运行项")
        sys.exit(0)

    print(f"\n发现 {len(pending)} 个待运行因子：\n")
    for p in pending:
        print(f"  {p['report']} / {p['factor']}")

    if args.dry_run:
        print("\n[DRY-RUN] 列表如上，未执行任何计算")
        sys.exit(0)

    # ── 第三步：逐个运行 ──
    total = len(pending)
    successes = 0
    failures = 0

    for i, p in enumerate(pending):
        ok = run_one(p, i, total, output_base=remote_full_base)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"\n{'='*60}")
    print(f"全部完成: {successes} 成功, {failures} 失败, 共 {total} 个因子")
    sys.exit(0 if failures == 0 else 1)