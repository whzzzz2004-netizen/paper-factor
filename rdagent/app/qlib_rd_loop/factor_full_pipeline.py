"""
全量因子流水线执行器 — 测试通过后自动触发。

设计：
  - FullPipelineExecutor 是单例，内部用 ThreadPoolExecutor(max_workers=1)
  - submit() 非阻塞，调用后立即返回
  - 主进程退出前调用 wait_for_completion() 等待所有任务完成
"""

import ast
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── 路径常量（与 scripts/full.py 保持一致） ──
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # rdagent/app/qlib_rd_loop/ → project root

SMB_HOST = "192.168.1.13"
SMB_SHARE = "E"
SMB_USER = "pc"
SMB_PASS = "123456"
CIFS_MOUNT = Path("/mnt/remote_e")


def _ensure_remote_mounted() -> bool:
    """自动挂载远程 E 盘（不需要用户手动操作）"""
    if CIFS_MOUNT.exists() and any(CIFS_MOUNT.iterdir()):
        return True
    try:
        CIFS_MOUNT.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["sudo", "mount", "-t", "cifs", f"//{SMB_HOST}/{SMB_SHARE}", str(CIFS_MOUNT),
             "-o", f"user={SMB_USER},password={SMB_PASS},uid={os.getuid()},gid={os.getgid()},file_mode=0644,dir_mode=0755,iocharset=utf8,noperm"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"  📡 已自动挂载远程 E 盘 → {CIFS_MOUNT}", flush=True)
            return True
    except Exception:
        pass
    return False


# 数据目录自动检测（多路径降级 + 自动挂载）
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
    # 全都没找到 → 尝试自动挂载远程再重试
    print("  ⏳ 未找到数据目录，尝试自动挂载远程 E 盘...", flush=True)
    if _ensure_remote_mounted():
        for p in candidates:
            if p and (Path(p) / "stock_data" / "daily").exists():
                return Path(p)
    return Path(".")

# 输出目录优先远程
_REMOTE_OUTPUTS = [
    Path("/mnt/remote_e/paper_factors/文献因子_全量"),
    Path("E:\\paper_factors\\文献因子_全量"),
    Path("Z:\\paper_factors\\文献因子_全量"),
]
FULL_OUTPUT_BASE = next((p for p in _REMOTE_OUTPUTS if p.exists()), PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量")
FULL_DATA_DIR = _detect_data_dir()
BARRA_DIR = Path(os.environ.get("PAPER_FACTOR_BARRA_DIR",
                                 str(PROJECT_ROOT / "git_ignore_folder" / "barra_model")))

EXPECTED_ROWS = 2027
DEFAULT_N_WORKERS = 4  # 日线/截面因子4核

# ── 工具函数 ──


def detect_factor_type_from_code(code: str) -> str:
    """检测因子类型: minute / daily / cross_section / minute_cs / deep_learning"""
    if "calc_factors_one_day" in code:
        return "minute"
    if "calc_factor_minute_raw" in code and "cross_section_transform" in code:
        return "minute_cs"
    if "calc_factor_cross_section" in code:
        return "cross_section"
    if "calc_factor_single_stock" in code:
        return "daily"
    if "train_model" in code and "predict" in code:
        return "deep_learning"
    if "def calc_factors_one_day" in code:
        return "minute"
    if "def _compute_day_raw" in code or "_compute_chunk_full" in code:
        return "minute_cs"
    return "unknown"


def detect_factor_type(code_path: Path) -> str:
    return detect_factor_type_from_code(code_path.read_text())


def cleanup_workers(factor_name: str | None = None):
    """清除可能残留的worker进程"""
    subprocess.run(["pkill", "-9", "-f", "python3 -c  import os"], capture_output=True, timeout=5)
    # 注意：不能用 -f "{factor_name}.code.py"，父进程命令行包含 code.py 路径会误杀自己
    gc.collect()


# ── 全量计算 ──


def _find_remote_code(factor_name: str, report_name: str) -> Path | None:
    """从远程 E 盘查找已更新的 .code.py

    数据目录与产出目录在远程盘上是同级关系：
      {base}/_paper_factor_unified/factor_implementation_source_data/   ← 数据
      {base}/paper_factors/文献因子_全量/{report}/{factor}/{factor}.code.py  ← 产出
    """
    if FULL_DATA_DIR is None or FULL_DATA_DIR == Path("."):
        return None
    # 从数据目录推导产出目录基路径
    remote_base = FULL_DATA_DIR.parent.parent / "paper_factors" / "文献因子_全量"
    p = remote_base / report_name / factor_name / f"{factor_name}.code.py"
    if p.exists():
        return p
    # 试原始硬编码路径（兼容旧挂载）
    for prefix in ("/mnt/remote_e", "E:", "Z:"):
        alt = Path(f"{prefix}/paper_factors/文献因子_全量") / report_name / factor_name / f"{factor_name}.code.py"
        if alt.exists():
            return alt
    return None


def _patch_old_code_data_dir(code_dst: Path):
    """修补旧 .code.py: DATA_DIR = Path("硬编码") → DATA_DIR = Path(env_var or "硬编码")"""
    code = code_dst.read_text()
    if 'os.environ.get("FACTOR_DATA_DIR")' in code:
        return  # 已有 env var 检查
    code = re.sub(
        r'^(DATA_DIR\s*=\s*Path\()([^)]+)(\))',
        r'\1os.environ.get("FACTOR_DATA_DIR") or \2\3',
        code,
        count=1,
        flags=re.MULTILINE,
    )
    code_dst.write_text(code)
    print(f"  🔧 已修补旧 .code.py: DATA_DIR 优先读 FACTOR_DATA_DIR 环境变量", flush=True)


def run_other_factor(factor_name: str, factor_dir: Path, code_path: Path) -> bool:
    """本地运行日线/截面因子（直接执行现有的 .code.py）"""
    factor_dir.mkdir(parents=True, exist_ok=True)

    # 复制 .code.py（优先用远程已更新的版本）
    code_dst = factor_dir / f"{factor_name}.code.py"
    report_name = factor_dir.parent.name
    remote_code = _find_remote_code(factor_name, report_name)
    src_code = remote_code if remote_code else code_path
    if src_code.resolve() != code_dst.resolve():
        shutil.copy(src_code, code_dst)
    if remote_code:
        print(f"  📡 使用远程已更新的 .code.py", flush=True)
    else:
        _patch_old_code_data_dir(code_dst)  # 保底：修补旧代码

    # 清除旧结果
    for p in [factor_dir / f"{factor_name}.parquet", factor_dir / "result.parquet"]:
        if p.exists():
            p.unlink()

    for attempt in range(2):  # 最多2次，防间歇性 loky 崩溃
        if attempt > 0:
            print(f"  🔄 重试第 {attempt + 1} 次...", flush=True)
            time.sleep(2)

        # 运行
        env = os.environ.copy()
        env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
        env["RDAGENT_FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
        env["FACTOR_N_WORKERS"] = str(DEFAULT_N_WORKERS)
        env["PYTHONWARNINGS"] = "ignore"
        env["PYTHONUNBUFFERED"] = "1"

        t0 = time.time()
        log_path = Path(f"/tmp/{factor_name}.run.log")
        last_lines = []
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable, str(code_dst)],
                stdout=log_f, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=str(factor_dir),
                preexec_fn=os.setpgrp,
            )
            last_pos = 0
            while proc.poll() is None:
                time.sleep(0.5)
                with open(log_path) as rf:
                    rf.seek(last_pos)
                    for line in rf:
                        print(line, end="", flush=True)
                        last_lines.append(line)
                        if len(last_lines) > 20:
                            last_lines.pop(0)
                    last_pos = rf.tell()
        # 读退出前的剩余输出
        try:
            with open(log_path) as rf:
                rf.seek(last_pos)
                for line in rf:
                    print(line, end="", flush=True)
                    last_lines.append(line)
                    if len(last_lines) > 20:
                        last_lines.pop(0)
        except OSError:
            pass
        elapsed = time.time() - t0
        print(f"  subprocess returncode={proc.returncode}, elapsed={elapsed:.0f}s", flush=True)

        if proc.returncode != 0:
            print(f"  ⚠️ 运行退出码非零 (code={proc.returncode})", flush=True)
            for line in last_lines[-10:]:
                print(f"    {line}", end="", flush=True)

        result_parquet = factor_dir / "result.parquet"
        if result_parquet.exists():
            df = pd_read_parquet(result_parquet)
            print(f"  ✅ 完成: {df.shape[0]}天 x {df.shape[1]}只, {elapsed:.0f}s", flush=True)

            dst_parquet = factor_dir / f"{factor_name}.parquet"
            if dst_parquet.exists():
                dst_parquet.unlink()
            shutil.move(result_parquet, dst_parquet)

            cleanup_workers(factor_name)
            return True

        cleanup_workers(factor_name)
        if attempt == 0:
            print(f"  ❌ result.parquet 未生成", flush=True)
        else:
            print(f"  ❌ result.parquet 未生成（重试后仍失败）", flush=True)

    return False


def run_minute_factor(factor_name: str, factor_dir: Path, code_path: Path) -> bool:
    """用本地模板运行分钟因子（直接复用 .code.py，已含 per-stock 模板）"""
    factor_dir.mkdir(parents=True, exist_ok=True)

    # 复制 .code.py（优先用远程已更新的版本）
    code_dst = factor_dir / f"{factor_name}.code.py"
    report_name = factor_dir.parent.name
    remote_code = _find_remote_code(factor_name, report_name)
    src_code = remote_code if remote_code else code_path
    if src_code.resolve() != code_dst.resolve():
        shutil.copy(src_code, code_dst)
    if remote_code:
        print(f"  📡 使用远程已更新的 .code.py", flush=True)
    else:
        _patch_old_code_data_dir(code_dst)

    # 修补旧模板的硬编码 CHUNK_SIZE → 环境变量可配置（防OOM）
    _old_code = code_dst.read_text()
    _patched = re.sub(
        r'CHUNK_SIZE\s*=\s*\d+',
        'CHUNK_SIZE = int(os.environ.get("FACTOR_CHUNK_SIZE", "15"))',
        _old_code,
    )
    if _patched != _old_code:
        code_dst.write_text(_patched)
        print(f"  🔧 CHUNK_SIZE 已修补为环境变量可配置", flush=True)

    # 清除旧结果
    chk_dir = factor_dir / "checkpoints"
    if chk_dir.exists():
        for f in chk_dir.glob("chk_*.parquet"):
            f.unlink()
        try:
            chk_dir.rmdir()
        except Exception:
            pass
    for p in [factor_dir / f"{factor_name}.parquet", factor_dir / "result.parquet"]:
        if p.exists():
            p.unlink()

    env = os.environ.copy()

    for attempt in range(2):
        if attempt > 0:
            print(f"  🔄 重试第 {attempt + 1} 次... (FACTOR_CHUNK_SIZE=15, FACTOR_N_WORKERS=8)", flush=True)
            time.sleep(2)
            # 重试时缩小 chunk 和并行度减小内存压力
            env["FACTOR_CHUNK_SIZE"] = "15"
            env["FACTOR_N_WORKERS"] = "8"

        env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
        env["PYTHONWARNINGS"] = "ignore"
        env["PYTHONUNBUFFERED"] = "1"

        t0 = time.time()
        log_path = Path(f"/tmp/{factor_name}.run.log")
        last_lines = []
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable, str(code_dst)],
                stdout=log_f, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=str(factor_dir),
                preexec_fn=os.setpgrp,
            )
            last_pos = 0
            while proc.poll() is None:
                time.sleep(0.5)
                with open(log_path) as rf:
                    rf.seek(last_pos)
                    for line in rf:
                        print(line, end="", flush=True)
                        last_lines.append(line)
                        if len(last_lines) > 20:
                            last_lines.pop(0)
                    last_pos = rf.tell()
        # 读退出前的剩余输出（cleanup 之前，避免 pkill 干扰）
        try:
            with open(log_path) as rf:
                rf.seek(last_pos)
                for line in rf:
                    print(line, end="", flush=True)
                    last_lines.append(line)
                    if len(last_lines) > 20:
                        last_lines.pop(0)
        except OSError:
            pass  # 子进程可能删了日志，不影响
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  ⚠️ 运行异常退出 (code={proc.returncode})", flush=True)
            chk_files = sorted(chk_dir.glob("chk_*.parquet")) if chk_dir.exists() else []
            if chk_files:
                n_chk = len(chk_files)
                print(f"  发现 {n_chk} 个checkpoint（部分完成），不合并为最终结果", flush=True)
                # 清理checkpoint，让重试从零开始
                for _f in chk_files:
                    _f.unlink()
                try:
                    chk_dir.rmdir()
                except Exception:
                    pass
            else:
                for line in last_lines[-10:]:
                    print(f"    {line}", end="", flush=True)
            if attempt == 0:
                print(f"  🔄 准备重试...", flush=True)
                continue
            print(f"  ❌ 重试仍失败", flush=True)
            cleanup_workers(factor_name)
            return False
        else:
            result_parquet = factor_dir / "result.parquet"

        if result_parquet.exists():
            df = pd_read_parquet(result_parquet)
            print(f"  ✅ 完成: {df.shape[0]}天 x {df.shape[1]}只, {elapsed:.0f}s", flush=True)

            dst_parquet = factor_dir / f"{factor_name}.parquet"
            if dst_parquet.exists():
                dst_parquet.unlink()
            shutil.move(result_parquet, dst_parquet)

            cleanup_workers(factor_name)
            return True

        cleanup_workers(factor_name)
        if attempt == 0:
            print(f"  ❌ result.parquet 未生成", flush=True)
        else:
            print(f"  ❌ result.parquet 未生成（重试后仍失败）", flush=True)

    return False


def pd_read_parquet(path):
    """安全的 parquet 读取，避免 pandas 未来警告"""
    import pandas as pd
    return pd.read_parquet(path)


# ── 后处理 ──


def post_process(factor_name: str, factor_dir: Path, factor_type: str,
                 test_meta: dict | None = None, skip_eval: bool = False) -> bool:
    """评估 + 绘图 + meta.json + 复制报告"""
    dst_parquet = factor_dir / f"{factor_name}.parquet"
    if not dst_parquet.exists():
        print(f"  ❌ 找不到 parquet 文件", flush=True)
        return False

    import pandas as pd
    import numpy as np

    df = pd_read_parquet(dst_parquet)
    print(f"  因子: {df.shape[0]}天 x {df.shape[1]}只, 非空={int(df.notna().sum().sum())}", flush=True)

    # 转换宽表（日期=行，股票=列）为评估函数可识别的格式
    _eval_df = df.copy()
    _eval_df.index = pd.to_datetime(_eval_df.index)
    _eval_df.index.name = "Date"
    _eval_df.columns = _eval_df.columns.astype(str)
    _eval_df.columns.name = "Code"

    # index 可能是 datetime 或 string（截面因子输出）
    _idx_min = df.index.min()
    _idx_max = df.index.max()
    if hasattr(_idx_min, 'strftime'):
        date_range = f"{_idx_min.strftime('%Y-%m-%d')} ~ {_idx_max.strftime('%Y-%m-%d')}"
    else:
        date_range = f"{_idx_min} ~ {_idx_max}"

    meta_path = factor_dir / f"{factor_name}.meta.json"
    if not meta_path.exists():
        basic_meta = {
            "factor_name": factor_name,
            "factor_type": factor_type,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path.write_text(json.dumps(basic_meta, indent=2, ensure_ascii=False))

    eval_result = None
    label_df = None
    _plot_fn = None
    if not skip_eval:
        print(f"  评估中...", flush=True)
        # 确保 scripts/ 可导入
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from scripts.evaluate_factor import evaluate_factor as _eval_fn, load_full_data_label
            from scripts.plot_decile import plot_decile_returns as _plot_fn
        except Exception as e:
            print(f"  ⚠️ 导入评估模块失败: {e}", flush=True)
            _eval_fn = None
            _plot_fn = None
            load_full_data_label = None

        if _eval_fn is not None:
            try:
                label_df = load_full_data_label(FULL_DATA_DIR)
                eval_result = _eval_fn(_eval_df, FULL_DATA_DIR, label_df=label_df)
                if eval_result and "error" not in eval_result:
                    ic = eval_result.get("ic_mean", float("nan"))
                    ir = eval_result.get("ic_ir", "N/A")
                    ric = eval_result.get("rank_ic_mean", float("nan"))
                    ls = eval_result.get("long_short_mean", None)
                    sharpe = eval_result.get("long_short_sharpe", None)
                    print(f"    IC={ic:.6f}  IR={ir}  RankIC={ric:.6f}"
                          f"{f'  多空={ls:.4%}' if isinstance(ls, float) else ''}"
                          f"{f'  Sharpe={sharpe:.2f}' if isinstance(sharpe, float) else ''}",
                          flush=True)
                elif eval_result and "error" in eval_result:
                    print(f"    ⚠️ {eval_result['error']}", flush=True)
            except Exception as e:
                print(f"  ⚠️ 评估失败: {e}", flush=True)
                eval_result = None

        if _plot_fn is not None and label_df is not None:
            print(f"  生成图表...", flush=True)
            try:
                _plot_fn(_eval_df, label_df, factor_name, str(factor_dir / f"{factor_name}.decile.png"))
            except Exception as e:
                print(f"  ⚠️ 图表生成失败: {e}", flush=True)

    # Barra 暴露分析
    _barra_result = None
    barra_factor_returns = BARRA_DIR / "因子收益率表(Trading Model).csv"
    if barra_factor_returns.exists() and not skip_eval:
        print(f"  Barra 暴露分析...", flush=True)
        try:
            from scripts.barra_evaluate import evaluate_barra
            barra_result = evaluate_barra(_eval_df, FULL_DATA_DIR, BARRA_DIR, model="Trading Model")
            if "error" not in barra_result:
                alpha = barra_result["exposures"]["alpha"]
                sig_factors = [
                    f"{n}({e['coef']:+.4f})"
                    for n, e in barra_result["exposures"].items()
                    if n != "alpha" and abs(e["tstat"]) > 2
                ]
                print(f"    Alpha: {alpha['coef']:.6f} (t={alpha['tstat']:.2f})  "
                      f"R²={barra_result['r_squared']:.4f}  "
                      f"显著因子: {', '.join(sig_factors[:5])}{'...' if len(sig_factors) > 5 else ''}",
                      flush=True)
                _barra_result = barra_result
            else:
                print(f"    ⚠️ {barra_result['error']}", flush=True)
        except Exception as e:
            print(f"    ⚠️ Barra 分析失败: {e}", flush=True)

    # meta.json
    test_meta = test_meta or {}
    meta = {
        "factor_name": factor_name,
        "display_name": factor_name,
        "factor_type": factor_type,
        "factor_description": test_meta.get("factor_description") or test_meta.get("source_excerpt", ""),
        "factor_formulation": test_meta.get("factor_formulation") or test_meta.get("source_excerpt", ""),
        "variables": test_meta.get("variables", {}),
        "source_report_title": test_meta.get("source_report_title", ""),
        "source_report_path": test_meta.get("source_report_path", ""),
        "source_excerpt": test_meta.get("source_excerpt", ""),
        "rows": df.shape[0],
        "stock_count": df.shape[1],
        "non_null": int(df.notna().sum().sum()),
        "non_null_ratio": round(float(df.notna().mean().mean()), 4),
        "time_granularity": "daily",
        "dataset": "full",
        "date_range": date_range,
        "tags": ["literature_factor", factor_type, "full_dataset"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_status": "running",
    }
    if _barra_result is not None:
        meta["barra_analysis"] = _barra_result
    if eval_result:
        meta["evaluation"] = eval_result
    if meta_path.exists():
        try:
            eval_meta = json.loads(meta_path.read_text())
            for k, v in eval_meta.items():
                if k not in meta:
                    meta[k] = v
        except Exception:
            pass
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  ✅ meta: {meta_path.name}", flush=True)

    # 复制原始报告
    src_report = factor_dir.parent.parent / "literature_reports" / factor_dir.parent.name / factor_name / f"{factor_name}.report.md"
    # 回退：在测试输出目录查找
    if not src_report.exists():
        alt = (PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
               / factor_dir.parent.name / factor_name / f"{factor_name}.report.md")
        if alt.exists():
            src_report = alt
    if src_report.exists():
        shutil.copy(src_report, factor_dir / f"{factor_name}.report.md")
        print(f"  ✅ report: {factor_name}.report.md", flush=True)

    return True


# ── LLM 审查 ──


def llm_review_and_decide(factor_name: str, factor_dir: Path, meta: dict,
                          source_excerpt: str) -> bool:
    """LLM 审查因子：对比研报原文 vs 代码实现。

    Returns:
        True = 接受因子；False = 需要重新生成。
    """
    meta_path = factor_dir / f"{factor_name}.meta.json"
    code_path = factor_dir / f"{factor_name}.code.py"

    evaluation = meta.get("evaluation", {})
    ic_mean = evaluation.get("ic_mean")
    ls_sharpe = evaluation.get("long_short_sharpe")

    if not evaluation or ic_mean is None:
        print(f"  ⚠️ 无有效的评价数据，跳过 LLM 审查", flush=True)
        return True

    # 检查是否需要触发审查
    needs_review = False
    review_reasons = []
    if abs(ic_mean) < 0.005:
        needs_review = True
        review_reasons.append(f"IC={ic_mean:.4f} < 0.01")
    if ls_sharpe is not None and ls_sharpe < -1.0:
        needs_review = True
        review_reasons.append(f"多空Sharpe={ls_sharpe:.2f} < -1.0")

    # Barra 极度偏离检查
    barra = meta.get("barra_analysis")
    if barra and "error" not in barra:
        r2 = barra.get("r_squared", 0)
        if r2 > 0.25:
            strong_exposures = []
            for name, exp in barra.get("exposures", {}).items():
                if name != "alpha" and abs(exp.get("tstat", 0)) > 3.0:
                    strong_exposures.append(f"{name}(t={exp['tstat']:.1f})")
            if len(strong_exposures) >= 3:
                needs_review = True
                review_reasons.append(
                    f"Barra R²={r2:.2%}, {len(strong_exposures)}个因子|t|>3.0"
                )

    if not needs_review:
        print(f"  因子效果达标 (IC={ic_mean:.4f}, Sharpe={ls_sharpe:.2f})，跳过 LLM 审查", flush=True)
        return True

    print(f"  触发审查: {'; '.join(review_reasons)}", flush=True)

    if not source_excerpt:
        print(f"  ⚠️ 无 source_excerpt，使用 description+formulation 做 LLM 审查", flush=True)

    from rdagent.oai.llm_utils import APIBackend
    from scripts.llm_review_factor import review_factor_with_llm

    code = code_path.read_text() if code_path.exists() else ""
    if not code:
        print(f"  ⚠️ 无代码文件，跳过 LLM 审查", flush=True)
        return True

    result = review_factor_with_llm(
        report_title=meta.get("source_report_title", ""),
        factor_description=meta.get("factor_description", ""),
        factor_formulation=meta.get("factor_formulation", ""),
        variables=meta.get("variables", {}),
        code=code,
        factor_name=factor_name,
        source_excerpt=source_excerpt,
    )

    meta["llm_review"] = result
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  LLM 审查: {result.get('verdict')} — {result.get('summary')}", flush=True)
    # 始终接受，不触发重跑
    print(f"  ✅ 接受因子", flush=True)
    return True


def regenerate_and_rerun(factor_name: str, factor_dir: Path, factor_type: str,
                         source_excerpt: str) -> bool:
    """用 source_excerpt 重新生成因子代码并重新运行全量。

    重跑后直接接受（不再审查）。如果重跑失败，恢复备份并接受原结果。
    """
    if not source_excerpt:
        print(f"  ⚠️ 无 source_excerpt，跳过重新生成", flush=True)
        return False

    code_path = factor_dir / f"{factor_name}.code.py"

    if not code_path.exists():
        print(f"  ❌ 找不到 .code.py，无法重新生成", flush=True)
        return False

    code = code_path.read_text()

    user_func_name = None
    for func_name in ("calc_factor_single_stock", "calc_factors_one_day",
                      "calc_factor_cross_section", "calc_factor_minute_raw",
                      "train_model"):
        if f"def {func_name}" in code:
            user_func_name = func_name
            break

    if not user_func_name:
        print(f"  ❌ 无法识别用户函数", flush=True)
        return False

    tree = ast.parse(code)
    func_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == user_func_name), None)
    if func_node is None:
        print(f"  ❌ AST 解析找不到函数 {user_func_name}", flush=True)
        return False

    lines = code.splitlines()
    old_func_source = "\n".join(lines[func_node.lineno - 1: func_node.end_lineno])

    from rdagent.oai.llm_utils import APIBackend

    rewrite_prompt = f"""你是一个量化因子代码专家。请根据研报原文重写因子的计算函数。

## 因子名称
{factor_name}

## 研报原文（最高优先级，以此为准）
{source_excerpt}

## 当前函数
```python
{old_func_source}
```

## 任务
重写 `{user_func_name}` 函数，使其与研报原文完全一致。
- 函数签名保持不变
- 只返回函数定义本身，不要额外代码
- 确保语法正确

请只输出重写后的函数，用 ```python ... ``` 包裹。"""

    try:
        api = APIBackend()
        response = api.build_messages_and_create_chat_completion(
            user_prompt=rewrite_prompt,
            system_prompt="你是一个量化因子代码专家。只输出重写后的函数定义。",
            json_mode=False,
        )
    except Exception as e:
        print(f"  ❌ LLM 重写失败: {e}", flush=True)
        return False

    m = re.search(r"```python\s*\n(.*?)\n```", response, re.DOTALL)
    new_func_source = m.group(1).strip() if m else response.strip()

    try:
        ast.parse(new_func_source)
    except SyntaxError as e:
        print(f"  ❌ LLM 生成的代码语法错误: {e}", flush=True)
        return False

    backup_path = code_path.with_suffix(".code.py.bak")
    code_path.rename(backup_path)

    old_lines = code.splitlines()
    idx_start = func_node.lineno - 1
    idx_end = func_node.end_lineno - 1
    new_lines = new_func_source.splitlines()
    new_code = "\n".join(old_lines[:idx_start] + new_lines + old_lines[idx_end:]) + "\n"

    try:
        ast.parse(new_code)
    except SyntaxError as e:
        print(f"  ❌ 注入后整体语法错误，恢复备份: {e}", flush=True)
        code_path.write_text(backup_path.read_text())
        return False

    code_path.write_text(new_code)
    print(f"  ✅ 函数 {user_func_name} 已重写（备份: {backup_path.name}）", flush=True)

    # 重跑全量 — 失败则恢复备份
    print(f"  🔄 重新运行全量...", flush=True)
    try:
        if factor_type in ("minute", "minute_cs"):
            ok = run_minute_factor(factor_name, factor_dir, code_path)
        else:
            ok = run_other_factor(factor_name, factor_dir, code_path)

        if not ok:
            print(f"  ❌ 重新运行失败，恢复备份", flush=True)
            if backup_path.exists():
                code_path.write_text(backup_path.read_text())
            return False

        ok = post_process(factor_name, factor_dir, factor_type, skip_eval=False)
        if ok:
            print(f"  ✅ 重新运行完成，直接接受（不再审查）", flush=True)
        return ok
    except Exception:
        print(f"  ❌ 重新运行异常，恢复备份代码", flush=True)
        traceback.print_exc()
        if backup_path.exists():
            code_path.write_text(backup_path.read_text())
        return False


# ── 远程同步 ──


def sync_to_remote(factor_dir: Path) -> bool:
    """通过 SMB 同步到远程E盘"""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from scripts.sync_utils import upload_file, ensure_remote_writable, REMOTE_BASE_FULL
    except ModuleNotFoundError:
        try:
            from sync_utils import upload_file, ensure_remote_writable, REMOTE_BASE_FULL
        except ModuleNotFoundError:
            print(f"  ⚠️ 无法导入 sync_utils，跳过同步", flush=True)
            return False

    if not ensure_remote_writable():
        print(f"  ⚠️ 远程不可用，跳过同步", flush=True)
        return False

    remote_prefix = f"{REMOTE_BASE_FULL}\\{factor_dir.parent.name}\\{factor_dir.name}"

    count = 0
    for f in factor_dir.iterdir():
        if f.is_file() and f.suffix in (".parquet", ".py", ".json", ".png", ".md"):
            remote_path = f"{remote_prefix}\\{f.name}"
            if upload_file(f, remote_path):
                count += 1

    print(f"  ✅ 已同步 {count} 个文件到远程: {remote_prefix}", flush=True)
    return count > 0


# ── 独立全量流水线 ──


def run_full_pipeline(
    factor_name: str,
    code_path: Path,
    output_dir: Path,
    factor_type: str | None = None,
    test_meta: dict | None = None,
    source_excerpt: str = "",
) -> bool:
    """独立全量流水线：计算 → 评估 → LLM审查 → 同步。

    可用于不依赖测试输出目录结构的场景，只需 .code.py + 元数据即可运行。

    Args:
        factor_name: 因子名
        code_path: .code.py 文件路径
        output_dir: 输出目录（计算结果、meta、图表等均写入此目录）
        factor_type: 因子类型（None=自动检测）
        test_meta: 测试阶段元数据（description/formulation/variables/source_excerpt等）
        source_excerpt: 研报原文摘录（单独提供时覆盖 test_meta 中的值）

    Returns:
        True=流水线成功完成；False=任一阶段失败
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"[FullPipeline] {output_dir.parent.name}/{factor_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 确定因子类型
    if factor_type is None:
        factor_type = detect_factor_type(code_path)
    print(f"  类型: {factor_type}", flush=True)

    # 2. 读取 test_meta
    test_meta = test_meta or {}

    # 3. 清理 checkpoints
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"  🧹 清理遗留 checkpoints", flush=True)

    try:
        # 4. 计算全量
        print(f"  计算全量...", flush=True)
        if factor_type in ("minute", "minute_cs"):
            ok = run_minute_factor(factor_name, output_dir, code_path)
        else:
            ok = run_other_factor(factor_name, output_dir, code_path)

        if not ok:
            print(f"  ❌ 全量计算失败", flush=True)
            return False

        # 5. 后处理
        print(f"  后处理...", flush=True)
        src_excerpt = source_excerpt or test_meta.get("source_excerpt", "")
        ok = post_process(factor_name, output_dir, factor_type, test_meta=test_meta)
        if not ok:
            print(f"  ❌ 后处理失败", flush=True)
            return False

        # 6. 重新读取 meta（post_process 可能已更新）
        meta_path_full = output_dir / f"{factor_name}.meta.json"
        try:
            meta = json.loads(meta_path_full.read_text())
        except Exception:
            meta = {}

        # 7. 跳过 LLM 审查，直接接受（用户要求保留所有因子）
        print(f"  ✅ 跳过 LLM 审查，直接接受因子", flush=True)

        # 8. 标记完成
        if meta_path_full.exists():
            try:
                meta = json.loads(meta_path_full.read_text())
            except Exception:
                meta = {}
            meta["pipeline_status"] = "completed"
            meta_path_full.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"  ✅ pipeline_status: completed", flush=True)

        # 9. 同步远程
        print(f"  同步远程...", flush=True)
        try:
            sync_to_remote(output_dir)
        except Exception as e:
            print(f"  ⚠️ 同步失败: {e}", flush=True)

        print(f"  ✅ [FullPipeline] {factor_name} 全量流水线完成", flush=True)
        return True

    except Exception as e:
        print(f"  ❌ [FullPipeline] {factor_name} 异常: {e}", flush=True)
        traceback.print_exc()
        # 标记失败
        meta_path_full = output_dir / f"{factor_name}.meta.json"
        if meta_path_full.exists():
            try:
                meta = json.loads(meta_path_full.read_text())
            except Exception:
                meta = {}
            meta["pipeline_status"] = "failed"
            meta["pipeline_error"] = str(e)
            meta_path_full.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return False


# ── FullPipelineExecutor ──


class FullPipelineExecutor:
    """全量因子流水线执行器（单例，后台线程池）。

    流程:
      1. 检查全量结果是否已存在 → 跳过
      2. 计算全量
      3. 后处理评估
      4. LLM 审查 → 如果"错误" → regenerate_and_rerun（最多1次，直接接受）
      5. 同步远程
      6. 标记完成
    """

    _instance = None

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[object] = []

    @classmethod
    def get_instance(cls, max_workers: int = 1) -> "FullPipelineExecutor":
        if cls._instance is None:
            cls._instance = cls(max_workers=max_workers)
        return cls._instance

    def submit(self, factor_name: str, report_name: str, code_path: Path,
               meta_path: Path, factor_type: str | None = None) -> object:
        """提交因子到全量流水线（非阻塞）。

        Args:
            factor_name: 因子名（slug）
            report_name: 研报名（slug）
            code_path: 测试阶段的 .code.py 路径
            meta_path: 测试阶段的 .meta.json 路径
            factor_type: 因子类型（None=自动检测）
        """
        future = self._executor.submit(
            self._run_pipeline, factor_name, report_name,
            code_path, meta_path, factor_type,
        )
        self._futures.append(future)
        return future

    def wait_for_completion(self, timeout: int | None = None):
        """等待所有已提交的任务完成。

        Args:
            timeout: 总超时秒数，None=无限等待
        """
        if not self._futures:
            return
        from concurrent.futures import as_completed, TimeoutError as CTimeoutError

        t_start = time.time()
        remaining = self._futures.copy()
        self._futures = []

        try:
            for future in as_completed(remaining, timeout=timeout):
                try:
                    future.result()
                except Exception as e:
                    print(f"  ⚠️ FullPipeline 任务异常: {e}", flush=True)
        except CTimeoutError:
            print(f"  ⚠️ FullPipeline 等待超时 ({timeout}s)，有任务未完成", flush=True)

        elapsed = time.time() - t_start
        if elapsed > 1:
            print(f"  FullPipeline 完成，耗时 {elapsed:.0f}s", flush=True)

    def _run_pipeline(self, factor_name: str, report_name: str,
                      code_path: Path, meta_path: Path,
                      factor_type: str | None):
        """内部：运行单个因子的全量流水线（在后台线程中执行）"""
        print(f"\n{'=' * 60}", flush=True)
        print(f"[FullPipeline] {report_name}/{factor_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        factor_dir = FULL_OUTPUT_BASE / report_name / factor_name

        # 检查是否已存在完整结果
        full_parquet = factor_dir / f"{factor_name}.parquet"
        if full_parquet.exists():
            print(f"  全量结果已存在，跳过计算", flush=True)
            return

        # 读取测试阶段 meta 信息
        test_meta = {}
        if meta_path.exists():
            try:
                test_meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        source_excerpt = test_meta.get("source_excerpt", "")

        # 委托给独立函数
        run_full_pipeline(
            factor_name=factor_name,
            code_path=code_path,
            output_dir=factor_dir,
            factor_type=factor_type,
            test_meta=test_meta,
            source_excerpt=source_excerpt,
        )
