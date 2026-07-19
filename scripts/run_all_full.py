#!/usr/bin/env python3
"""
全量因子批量运行入口。

扫描 文献因子_全量/ 下所有已部署（pipeline_status != "completed"）的因子，
逐个调用 run_full_pipeline() 完成：计算 → IC/IR评估 → 十分组图 → Barra暴露分析
→ LLM审查 → 同步远程 → 标记完成。

用法:
  python scripts/run_all_full.py [--report <report_name>] [--force]

说明:
  --report      只跑指定研报下的因子 (默认: 所有)
  --force       强制重跑 (即使 pipeline_status=completed)
  --dry-run     只打印将要跑的因子，不实际运行
  --watch <秒>  持续监听模式，每 N 秒扫描一次新部署的因子
"""

import argparse
import json
import os
import subprocess
import sys
import time as _time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
_REMOTE_OUTPUTS = [
    Path("/mnt/remote_e/paper_factors/文献因子_全量"),
    Path("E:\\paper_factors\\文献因子_全量"),
    Path("Z:\\paper_factors\\文献因子_全量"),
]
FULL_BASE = next((p for p in _REMOTE_OUTPUTS if p.exists()), PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量")

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
    """自动挂载远程 E 盘（modprobe + 多版本协商 + 自动装依赖）"""
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


# 自动检测数据目录
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
    # 自动挂载远程再试
    print("  ⏳ 未找到数据目录，尝试自动挂载远程 E 盘...")
    if _ensure_remote_mounted():
        for p in candidates:
            if p and (Path(p) / "stock_data" / "daily").exists():
                return Path(p)
    return Path(".")

FULL_DATA_DIR = _detect_data_dir()
# 设置环境变量，使子进程继承正确的数据路径（只有找到时才设，避免覆盖子进程自身检测）
if FULL_DATA_DIR != Path("."):
    os.environ["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
    os.environ["RDAGENT_FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)


def find_pending_factors(report_filter: str | None, force: bool) -> list[dict]:
    """返回 pending 因子列表，每项含 {report, factor, code_path, output_dir}"""
    if not FULL_BASE.exists():
        return []

    pending = []
    report_dirs = sorted(d for d in FULL_BASE.iterdir() if d.is_dir())
    if report_filter:
        report_dirs = [d for d in report_dirs if report_filter in d.name]

    for report_dir in report_dirs:
        report_name = report_dir.name
        factor_dirs = sorted(d for d in report_dir.iterdir() if d.is_dir())
        for factor_dir in factor_dirs:
            factor_name = factor_dir.name
            code_path = factor_dir / f"{factor_name}.code.py"
            meta_path = factor_dir / f"{factor_name}.meta.json"

            if not code_path.exists():
                continue

            # 检查 pipeline_status
            if not force and meta_path.exists():
                try:
                    m = json.loads(meta_path.read_text())
                    if m.get("pipeline_status") == "completed":
                        continue  # 已完成，跳过
                except Exception:
                    pass  # meta 异常 → 尝试重跑

            pending.append({
                "report": report_name,
                "factor": factor_name,
                "code_path": code_path,
                "output_dir": factor_dir,
                "meta_path": meta_path,
            })

    return pending


def run_one_factor(item: dict) -> dict:
    """跑单个因子的全量流水线"""
    factor_name = item["factor"]
    report_name = item["report"]
    code_path = item["code_path"]
    output_dir = item["output_dir"]

    print(f"\n{'='*60}")
    print(f"▶ {report_name} / {factor_name}")
    print(f"  code: {code_path}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from rdagent.app.qlib_rd_loop.factor_full_pipeline import run_full_pipeline

        ok = run_full_pipeline(
            factor_name=factor_name,
            code_path=code_path,
            output_dir=output_dir,
            factor_type=None,  # 自动检测
            test_meta=None,    # meta 已在 output_dir 中
            source_excerpt="",
        )

        if ok:
            print(f"  ✅ {report_name}/{factor_name} 完成")
        else:
            print(f"  ❌ {report_name}/{factor_name} 失败")

        return {"report": report_name, "factor": factor_name, "success": ok}

    except Exception as e:
        print(f"  ❌ {report_name}/{factor_name} 异常: {e}")
        return {"report": report_name, "factor": factor_name, "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="批量跑全量因子（从 文献因子_全量 扫描）")
    parser.add_argument("--report", help="指定研报名 (模糊匹配)", default=None)
    parser.add_argument("--force", action="store_true", help="强制重跑已完成因子")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待跑因子，不执行")
    parser.add_argument("--watch", type=int, metavar="SECONDS", default=0,
                        help="持续监听模式，每 N 秒扫描一次新部署的因子")
    args = parser.parse_args()

    if not FULL_DATA_DIR.exists():
        print(f"❌ 全量数据目录不存在: {FULL_DATA_DIR}")
        return 1

    pending = find_pending_factors(args.report, args.force)

    if not pending and not args.watch:
        print("✅ 所有因子已完成全量，无需执行。")
        return 0

    if args.dry_run:
        print(f"共发现 {len(pending)} 个待跑因子:\n")
        for item in pending:
            flag = "重跑" if args.force and item["output_dir"].exists() else "待跑"
            print(f"  [{flag}] {item['report']} / {item['factor']}")
        return 0

    if args.watch:
        filt = f" (筛选: {args.report})" if args.report else ""
        print(f"持续监听模式{filt} — 每 {args.watch}s 扫描，Ctrl+C 停止\n")

    success_count = 0
    fail_count = 0
    seen = set()

    while True:
        if not pending:
            if not args.watch:
                break
            print(f"\r⏳ 已完成: {success_count}, 失败: {fail_count} — 无新因子，等待部署...   ",
                  end="", flush=True)
            _time.sleep(args.watch)
            pending = find_pending_factors(args.report, args.force)
            pending = [p for p in pending if f"{p['report']}/{p['factor']}" not in seen]
            continue

        print(f"\n本轮 {len(pending)} 个待跑因子:")
        for p in pending:
            key = f"{p['report']}/{p['factor']}"
            if key not in seen:
                seen.add(key)
                print(f"  ▶ {key}")

        for p in pending:
            r = run_one_factor(p)
            if r["success"]:
                success_count += 1
            else:
                fail_count += 1

        if not args.watch:
            break

        pending = find_pending_factors(args.report, args.force)
        pending = [p for p in pending if f"{p['report']}/{p['factor']}" not in seen]
        print(f"  ✅ 本轮完成 — 累计成功: {success_count}, 失败: {fail_count}, 新待跑: {len(pending)}")

    print(f"\n{'='*60}")
    print(f"全部完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"{'='*60}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
