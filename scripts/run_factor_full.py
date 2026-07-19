#!/usr/bin/env python3
"""
将测试通过的因子代码接到全量数据运行，包含完整评估流程。

用法:
  python scripts/run_factor_full.py <factor_code.py> [--output-dir <dir>]

输出 (每个因子一个子目录):
  文献因子_全量/<report>/<factor_name>/
    ├── factor_name.parquet       # 全量因子值
    ├── factor_name.code.py       # 因子代码
    ├── factor_name.meta.json     # 元数据 (含评估指标 + LLM审查)
    ├── factor_name.decile.png    # 十分组收益图
    └── factor_name.report.md     # 原始研报信息
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

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


# 自动检测数据目录（同 run_all_full.py）
def _ensure_remote_mounted() -> bool:
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

def _detect_data_dir() -> Path:
    candidates = [
        os.environ.get("FACTOR_DATA_DIR", ""),
        os.environ.get("RDAGENT_FACTOR_DATA_DIR", ""),
        str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"),
        "/mnt/remote_e/_paper_factor_unified/factor_implementation_source_data",
        "E:\\_paper_factor_unified\\factor_implementation_source_data",
        "Z:\\_paper_factor_unified\\factor_implementation_source_data",
        "\\\\192.168.1.13\\E\\_paper_factor_unified\\factor_implementation_source_data",
        str(PROJECT_ROOT / "factor_implementation_source_data"),
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
_REMOTE_OUTPUTS = [
    Path("/mnt/remote_e/paper_factors/文献因子_全量"),
    Path("E:\\paper_factors\\文献因子_全量"),
    Path("Z:\\paper_factors\\文献因子_全量"),
]
OUTPUT_BASE = next((p for p in _REMOTE_OUTPUTS if p.exists()), PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量")

try:
    from scripts.sync_utils import ensure_remote_mounted, REMOTE_BASE_FULL as REMOTE_BASE
except ModuleNotFoundError:
    from sync_utils import ensure_remote_mounted, REMOTE_BASE_FULL as REMOTE_BASE  # type: ignore[import-unverified]
def detect_factor_type(code_path: Path) -> str:
    code = code_path.read_text()
    if any(k in code for k in ('MINUTE_DATA_DIR', 'MINUTE_BY_DATE_DIR', 'minute_pv', 'calc_factors_one_day')):
        return "minute"
    return "daily"


def run_locally(code_path: Path, workspace: Path, data_dir: Path, timeout: int = 7200, factor_type: str = "daily") -> Path | None:
    """本地运行因子代码（默认后端）"""
    shutil.copy(code_path, workspace / "factor.py")
    env = {k: str(v) for k, v in __import__('os').environ.items()}
    env["FACTOR_DATA_DIR"] = str(data_dir)
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # 日线因子限制4核，避免多因子并行时loky进程过多导致OOM
    if factor_type == "daily":
        env.setdefault("FACTOR_N_WORKERS", "4")
    print(f"  本地运行中... (数据: {data_dir})")
    result = subprocess.run(
        [sys.executable, "factor.py"],
        cwd=workspace,
        capture_output=True, text=True, timeout=timeout,
        env=env
    )
    # 打印 stdout（因子代码自己的输出）
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line:
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  ❌ 本地运行失败:")
        stderr = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        for line in stderr.split("\n"):
            line = line.strip()
            if line:
                print(f"    {line}")
        return None
    output = workspace / "result.parquet"
    if not output.exists():
        print(f"  ❌ 未找到输出文件 result.parquet")
        return None
    return output


def run_in_docker(code_path: Path, workspace: Path, data_dir: Path, timeout: int = 7200) -> Path | None:
    """Docker运行因子代码（--docker 时使用）"""
    shutil.copy(code_path, workspace / "factor.py")
    cmd = [
        "docker", "run", "--rm",
        "--shm-size=16g",
        "--gpus", "all",
        f"-v{workspace}:/workspace/factor_workspace",
        f"-v{data_dir}:/workspace/factor_data:ro",
        "-e", "FACTOR_DATA_DIR=/workspace/factor_data",
        "-e", "HDF5_USE_FILE_LOCKING=FALSE",
        "-w", "/workspace/factor_workspace",
        "local_factor_exec:latest",
        "python", "factor.py"
    ]
    print(f"  Docker运行中...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"  ❌ Docker运行失败:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return None
    output = workspace / "result.parquet"
    if not output.exists():
        print(f"  ❌ 未找到输出文件 result.parquet")
        return None
    return output


def sync_to_remote(factor_dir: Path, factor_name: str):
    """同步因子目录到远程E盘（自动挂载）"""
    # 输出已在远程路径上 → 无需同步
    if any(str(factor_dir).startswith(str(p)) for p in _REMOTE_OUTPUTS if p.exists()):
        print(f"  ✓ 已在远程路径，跳过同步")
        return True
    if not ensure_remote_mounted():
        print(f"  ⚠️ 远程E盘不可用，跳过同步")
        return False
    remote_parent = REMOTE_BASE / factor_dir.parent.name
    remote_dir = remote_parent / factor_name
    remote_dir.mkdir(parents=True, exist_ok=True)
    for f in factor_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, remote_dir / f.name)
            print(f"  已同步: {remote_dir / f.name}")
    return True


def create_source_report_md(source_meta: dict, factor_name: str) -> str:
    """生成原始研报信息markdown"""
    lines = [
        f"# {source_meta.get('source_report_title', '未知研报')}",
        f"",
        f"## 因子: {factor_name}",
        f"",
        f"### 因子描述",
        f"{source_meta.get('factor_description', '无')}",
        f"",
        f"### 因子公式",
        f"{source_meta.get('factor_formulation', '无')}",
        f"",
    ]
    variables = source_meta.get("variables", {})
    if variables:
        lines.append("### 变量说明")
        for k, v in variables.items():
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def create_metadata(factor_name: str, factor_type: str, parquet_path: Path, code_path: Path, source_meta: dict = None) -> dict:
    df = pd.read_parquet(parquet_path)
    meta = {
        "factor_name": factor_name,
        "display_name": factor_name,
        "factor_description": source_meta.get("factor_description", "") if source_meta else "",
        "factor_formulation": source_meta.get("factor_formulation", "") if source_meta else "",
        "variables": source_meta.get("variables", {}) if source_meta else {},
        "rows": df.shape[0],
        "non_null": int(df.notna().sum().sum()),
        "time_granularity": "daily",
        "dataset": "full",
        "stock_count": df.shape[1],
        "date_range": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "tags": ["literature_factor", factor_type, "full_dataset"],
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_report_title": source_meta.get("source_report_title", "") if source_meta else "",
        "source_type": "literature_report"
    }
    return meta


def main():
    parser = argparse.ArgumentParser(description="将因子代码接到全量数据运行")
    parser.add_argument("code", help="因子代码文件路径 (.code.py)")
    parser.add_argument("--output-dir", "-o", help="输出目录 (默认: 文献因子_全量)", default=None)
    parser.add_argument("--source-meta", help="原始元数据文件路径 (可选)", default=None)
    parser.add_argument("--timeout", type=int, default=7200, help="运行超时秒数 (默认: 7200)")
    parser.add_argument("--docker", action="store_true", help="使用Docker运行 (默认: 本地直接运行)")
    parser.add_argument("--skip-run", action="store_true", help="跳过运行 (已有parquet时使用)")
    parser.add_argument("--parquet", help="已有parquet文件路径 (与--skip-run配合)")
    args = parser.parse_args()

    code_path = Path(args.code).resolve()
    if not code_path.exists():
        print(f"❌ 代码文件不存在: {code_path}")
        return 1

    factor_type = detect_factor_type(code_path)
    print(f"因子类型: {factor_type}")

    data_dir = FULL_DATA_DIR
    if not data_dir.exists():
        print(f"❌ 全量数据目录不存在: {data_dir}")
        return 1

    # 读取原始元数据
    source_meta = None
    factor_name = code_path.stem.replace(".code", "")
    if args.source_meta:
        source_meta = json.loads(Path(args.source_meta).read_text())
    else:
        # per-factor 子目录结构: <factor>/<factor>.meta.json
        meta_in_dir = code_path.parent / f"{factor_name}.meta.json"
        if meta_in_dir.exists():
            source_meta = json.loads(meta_in_dir.read_text())
        else:
            # 旧结构 fallback: 与 code.py 同目录
            meta_path = code_path.with_suffix("").with_suffix(".meta.json")
            if meta_path.exists():
                source_meta = json.loads(meta_path.read_text())

    # 确定输出目录 → 使用 per-factor 子目录
    if args.output_dir:
        factor_dir = Path(args.output_dir)
    else:
        # per-factor 结构: literature_reports/<reoport>/<factor>/<factor>.code.py
        if code_path.parent.name == factor_name:
            report_dir = code_path.parent.parent.name
        else:
            report_dir = code_path.parent.name  # 旧结构 fallback
        factor_dir = OUTPUT_BASE / report_dir / factor_name

    factor_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {factor_dir}")

    # 运行因子（本地默认 / Docker可选）
    if args.skip_run and args.parquet:
        result_parquet = Path(args.parquet)
        if not result_parquet.exists():
            print(f"❌ 指定的parquet文件不存在: {result_parquet}")
            return 1
        print(f"使用已有parquet: {result_parquet}")
    elif args.skip_run:
        # 检查是否已有结果
        existing = factor_dir / f"{factor_name}.parquet"
        if existing.exists():
            result_parquet = existing
            print(f"使用已有结果: {result_parquet}")
            # 确保code.py也复制到因子目录
            target_code = factor_dir / f"{factor_name}.code.py"
            if not target_code.exists():
                shutil.copy(code_path, target_code)
                print(f"  已复制代码: {target_code}")
        else:
            print(f"❌ --skip-run但未找到已有parquet")
            return 1
    else:
        workspace = factor_dir  # 直接在因子目录运行，checkpoint实时可见
        print(f"运行因子: {factor_name} (输出目录: {workspace})")
        if args.docker:
            result_parquet = run_in_docker(code_path, workspace, data_dir, args.timeout)
        else:
            result_parquet = run_locally(code_path, workspace, data_dir, args.timeout, factor_type)
        if result_parquet is None:
            return 1

        # 确保 result.parquet 以因子名命名
        dst_parquet = factor_dir / f"{factor_name}.parquet"
        if result_parquet != dst_parquet:
            shutil.copy(result_parquet, dst_parquet)
            result_parquet = dst_parquet
        # 复制代码文件（可能已在目标目录）
        dst_code = factor_dir / f"{factor_name}.code.py"
        if code_path.resolve() != dst_code.resolve():
            shutil.copy(code_path, dst_code)

    # 创建元数据
    meta = create_metadata(factor_name, factor_type, result_parquet, code_path, source_meta)
    meta_path = factor_dir / f"{factor_name}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ 因子计算完成!")
    print(f"   规模: {meta['rows']}天 × {meta['stock_count']}只股票")

    # --- 阶段1: 回测评估 (IC/IR/十分组) ---
    print(f"\n{'='*50}")
    print(f"阶段1: 因子回测评估")
    print(f"{'='*50}")
    eval_script = PROJECT_ROOT / "scripts" / "evaluate_factor.py"
    eval_result = subprocess.run(
        [sys.executable, str(eval_script), str(result_parquet),
         "--data-dir", str(data_dir)],
        capture_output=True, text=True, timeout=600
    )
    if eval_result.returncode == 0:
        for line in eval_result.stdout.split("\n"):
            if any(k in line for k in ("IC (Pearson)", "Rank IC", "D1:", "D2:", "D3:", "多空收益", "Sharpe", "Max DD")):
                print(f"  {line.strip()}")
        # 从meta.json读取评估结果（evaluate_factor.py会自动更新meta.json）
        meta_path = factor_dir / f"{factor_name}.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    else:
        print(f"  ⚠️ 评估失败: {eval_result.stderr[-300:] if eval_result.stderr else 'unknown'}")

    # --- 阶段2: 十分组收益图 ---
    print(f"\n{'='*50}")
    print(f"阶段2: 生成十分组收益图")
    print(f"{'='*50}")
    plot_script = PROJECT_ROOT / "scripts" / "plot_decile.py"
    plot_output = factor_dir / f"{factor_name}.decile.png"
    plot_result = subprocess.run(
        [sys.executable, str(plot_script), str(result_parquet),
         "--data-dir", str(data_dir), "--output", str(plot_output)],
        capture_output=True, text=True, timeout=600
    )
    if plot_result.returncode == 0:
        print(f"  ✅ 图表已生成: {plot_output}")
    else:
        print(f"  ⚠️ 图表生成失败: {plot_result.stderr[-200:] if plot_result.stderr else 'unknown'}")

    # --- 阶段3: Barra 暴露分析 ---
    print(f"\n{'='*50}")
    print(f"阶段3: Barra 暴露分析")
    print(f"{'='*50}")
    barra_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "barra_evaluate.py"),
         str(result_parquet), "--data-dir", str(data_dir)],
        capture_output=True, text=True, timeout=300
    )
    if barra_result.returncode == 0:
        for line in barra_result.stdout.split("\n"):
            if any(k in line for k in ("Alpha", "显著因子", "tstat", "R²")):
                print(f"  {line.strip()}")
        # barra_evaluate.py 会自动更新 meta.json
        meta_path = factor_dir / f"{factor_name}.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    else:
        print(f"  ⚠️ Barra分析失败: {barra_result.stderr[-200:] if barra_result.stderr else 'unknown'}")

    # --- 阶段4: 原始研报信息 ---
    if source_meta:
        report_md = create_source_report_md(source_meta, factor_name)
        report_path = factor_dir / f"{factor_name}.report.md"
        report_path.write_text(report_md)
        print(f"\n研报信息已保存: {report_path}")

    # --- 阶段4: LLM逻辑正确性审查（测试阶段已做过则跳过） ---
    print(f"\n{'='*50}")
    print(f"阶段3: LLM逻辑正确性审查")
    print(f"{'='*50}")

    # 检查meta.json是否已有LLM审查结果
    existing_meta = json.loads(meta_path.read_text())
    if existing_meta.get("llm_review") and existing_meta["llm_review"].get("verdict"):
        print(f"  复用测试阶段LLM审查结果: {existing_meta['llm_review']['verdict']}")
        print(f"  {existing_meta['llm_review']['summary']}")
    else:
        llm_script = PROJECT_ROOT / "scripts" / "llm_review_factor.py"
        code_file = factor_dir / f"{factor_name}.code.py"
        llm_result = subprocess.run(
            [sys.executable, str(llm_script), str(meta_path), str(code_file)],
            capture_output=True, text=True, timeout=120
        )
        if llm_result.returncode == 0:
            for line in llm_result.stdout.split("\n"):
                if any(k in line for k in ("判定", "总结")):
                    print(f"  {line.strip()}")
        else:
            print(f"  ⚠️ LLM审查失败: {llm_result.stderr[-200:] if llm_result.stderr else 'unknown'}")
        # 重新读取meta（llm_review_factor.py已更新meta.json）
        meta = json.loads(meta_path.read_text())

    # 更新meta.json (含评估+LLM审查)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- 同步到远程 ---
    print(f"\n同步到远程E盘...")
    try:
        sync_to_remote(factor_dir, factor_name)
    except Exception as e:
        print(f"  ⚠️ 远程同步失败: {e}")

    print(f"\n{'='*50}")
    print(f"✅ 全部完成! 因子: {factor_name}")
    print(f"   {factor_dir}/")
    print(f"   - {factor_name}.parquet")
    print(f"   - {factor_name}.code.py")
    print(f"   - {factor_name}.meta.json  (含评估指标 + LLM审查)")
    print(f"   - {factor_name}.decile.png")
    print(f"   - {factor_name}.report.md")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
