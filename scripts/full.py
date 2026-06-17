#!/usr/bin/env python3
"""
全量因子统一运行命令 — 扫描测试集 + 重跑不完整因子。

一条指令搞定所有全量计算：
  1. 扫描 literature_reports/ 下已通过测试但未跑全量的因子
  2. 扫描 文献因子_全量/ 下日期不完整的因子并重跑
  3. 分钟因子 → 本地并行 (32 workers, 列过滤, checkpoint, pool重启)
  4. 日线/截面因子 → 本地运行现有 .code.py
  5. 评估(IC/IR) + 十分组图 + 同步远程

用法:
  python3 scripts/full.py                     # 扫所有待跑因子并运行
  python3 scripts/full.py --force             # 强制重跑所有（不管是否完整）
  python3 scripts/full.py --dry-run           # 只打印不执行
  python3 scripts/full.py --minute-only       # 只跑分钟线因子
  python3 scripts/full.py --daily-only        # 只跑日线/截面因子
  python3 scripts/full.py --report <name>     # 只跑指定研报(模糊匹配)
  python3 scripts/full.py --workers 32        # 分钟因子 worker 数
  python3 scripts/full.py --skip-sync         # 跳过远程同步
  python3 scripts/full.py --skip-eval         # 跳过评估/绘图
  python3 scripts/full.py --llm-review        # 全量运行后对效果差的因子做LLM审查
  python3 scripts/full.py --ic-threshold 0.01 --sharpe-threshold -0.5  # LLM审查触发阈值（默认IC<0.01 或 多空Sharpe<-1）
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LIT_REPORTS_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
FULL_OUTPUT_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"
FULL_DATA_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"
BARRA_DIR = Path(os.environ.get("PAPER_FACTOR_BARRA_DIR",
                                 str(PROJECT_ROOT / "git_ignore_folder" / "barra_model")))

DEFAULT_N_WORKERS = 32
EXPECTED_ROWS = 2027  # 完整日期数

# ── 导入分钟因子模板和工具 ──
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.batch_minute_full_local import (
    NEW_TEMPLATE,
    extract_user_code,
    wrap_with_new_template,
    sync_to_remote,
    cleanup_workers,
)


def detect_factor_type(code_path: Path) -> str:
    """检测因子类型: minute / daily / cross_section / minute_cs / deep_learning"""
    code = code_path.read_text()
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
    # fallback: check function defs
    if "def calc_factors_one_day" in code:
        return "minute"
    if "def _compute_day_raw" in code:
        return "minute_cs"
    return "unknown"


def scan_pending_factors(report_filter: str | None = None, force: bool = False,
                         minute_only: bool = False, daily_only: bool = False) -> list[dict]:
    """扫描所有待跑因子。

    返回 list[dict]:
      - name, report_name, factor_dir_name: 标识
      - code_path, meta_path: 测试输出中的路径
      - factor_type: minute / daily / cross_section / minute_cs
      - full_dir: 全量输出目录
      - status: "new" / "incomplete" / "force"
      - existing_rows: 已有行数
    """
    pending = []

    for report_dir in sorted(LIT_REPORTS_DIR.iterdir()):
        if not report_dir.is_dir():
            continue
        if report_filter and report_filter not in report_dir.name:
            continue

        for factor_dir in sorted(report_dir.iterdir()):
            if not factor_dir.is_dir():
                continue

            factor_name = factor_dir.name
            code_path = factor_dir / f"{factor_name}.code.py"
            meta_path = factor_dir / f"{factor_name}.meta.json"

            if not code_path.exists():
                continue

            # 只处理测试通过的因子
            if meta_path.exists():
                try:
                    m = json.loads(meta_path.read_text())
                    if not m.get("accepted", False):
                        continue
                except Exception:
                    continue
            else:
                continue

            ftype = detect_factor_type(code_path)

            # 类型过滤
            if minute_only and ftype not in ("minute", "minute_cs"):
                continue
            if daily_only and ftype not in ("daily", "cross_section", "deep_learning"):
                continue

            # 检查全量结果
            full_dir = FULL_OUTPUT_BASE / report_dir.name / factor_name
            full_meta = full_dir / f"{factor_name}.meta.json"
            full_parquet = full_dir / f"{factor_name}.parquet"

            status = "new"
            existing_rows = 0

            if full_parquet.exists() and full_meta.exists():
                try:
                    fm = json.loads(full_meta.read_text())
                    existing_rows = fm.get("rows", 0)
                except Exception:
                    existing_rows = 0

                if force:
                    status = "force"
                elif existing_rows >= EXPECTED_ROWS and not (full_dir / "checkpoints").exists():
                    continue  # 完整且无残留checkpoint, 跳过
                elif existing_rows > 0:
                    status = "incomplete"

            pending.append({
                "name": factor_name,
                "report_name": report_dir.name,
                "factor_dir_name": factor_dir.name,
                "code_path": code_path,
                "meta_path": meta_path,
                "factor_type": ftype,
                "full_dir": full_dir,
                "status": status,
                "existing_rows": existing_rows,
            })

    return pending


def run_minute_factor(factor: dict, n_workers: int) -> bool:
    """用本地模板运行分钟因子 (32 workers, 列过滤, checkpoint, pool重启)"""
    factor_name = factor["name"]
    factor_dir = factor["full_dir"]
    factor_dir.mkdir(parents=True, exist_ok=True)

    # 提取用户代码
    user_code, lookback = extract_user_code(factor["code_path"])
    if not user_code:
        print(f"  ❌ 无法提取 calc_factors_one_day", flush=True)
        return False

    # 重新包装
    wrapped = wrap_with_new_template(user_code, lookback)
    code_dst = factor_dir / f"{factor_name}.code.py"
    code_dst.write_text(wrapped)
    print(f"  代码已重新包装: {code_dst.name}", flush=True)

    # 如果有旧checkpoint, 清除 (重新包装后必须从头跑)
    chk_dir = factor_dir / "checkpoints"
    if chk_dir.exists():
        for f in chk_dir.glob("chk_*.parquet"):
            f.unlink()
        try:
            chk_dir.rmdir()
        except Exception:
            pass

    # 如果有旧result.parquet, 清除
    old_parquet = factor_dir / f"{factor_name}.parquet"
    if old_parquet.exists():
        old_parquet.unlink()
    old_result = factor_dir / "result.parquet"
    if old_result.exists():
        old_result.unlink()

    # 运行
    env = os.environ.copy()
    env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
    env["FACTOR_N_WORKERS"] = str(n_workers)
    env["PYTHONWARNINGS"] = "ignore"

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(code_dst)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd=str(factor_dir),
    )
    last_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        last_lines.append(line)
        if len(last_lines) > 20:
            last_lines.pop(0)
    proc.wait()
    elapsed = time.time() - t0

    # 杀残留worker
    cleanup_workers(factor_name)

    if proc.returncode != 0:
        print(f"  ⚠️ 运行异常退出 (code={proc.returncode})", flush=True)
        # 尝试合并checkpoint
        chk_dir = factor_dir / "checkpoints"
        chk_files = sorted(chk_dir.glob("chk_*.parquet")) if chk_dir.exists() else []
        if chk_files:
            print(f"  发现 {len(chk_files)} 个checkpoint, 尝试合并...", flush=True)
            parts = [pd.read_parquet(f) for f in chk_files]
            long_df = pd.concat(parts, ignore_index=True)
            for f in chk_files:
                f.unlink()
            try:
                chk_dir.rmdir()
            except Exception:
                pass
            long_df["datetime"] = pd.to_datetime(long_df["datetime"])
            fname = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
            wide = long_df.pivot(index="datetime", columns="instrument", values=fname)
            wide = wide.sort_index().sort_index(axis=1)
            wide.index.name = "Date"
            wide.columns.name = "Code"
            wide = wide.replace([np.inf, -np.inf], np.nan)
            wide.to_parquet(factor_dir / "result.parquet")
            nn = int(wide.notna().sum().sum())
            print(f"  ✅ checkpoint合并完成: {wide.shape[0]}天 x {wide.shape[1]}只, "
                  f"非空={nn}/{wide.size}={nn/wide.size*100:.1f}%", flush=True)
            result_parquet = factor_dir / "result.parquet"
        else:
            for line in last_lines[-10:]:
                print(f"    {line}", end="", flush=True)
            print(f"  ❌ 运行失败, 无checkpoint可恢复", flush=True)
            return False
    else:
        result_parquet = factor_dir / "result.parquet"

    if not result_parquet.exists():
        print(f"  ❌ result.parquet 未生成", flush=True)
        return False

    df = pd.read_parquet(result_parquet)
    print(f"  ✅ 完成: {df.shape[0]}天 x {df.shape[1]}只, {elapsed:.0f}s", flush=True)

    # 重命名
    dst_parquet = factor_dir / f"{factor_name}.parquet"
    if dst_parquet.exists():
        dst_parquet.unlink()
    shutil.move(result_parquet, dst_parquet)
    return True


def run_other_factor(factor: dict) -> bool:
    """本地运行日线/截面因子 (直接执行现有的 .code.py)"""
    factor_name = factor["name"]
    factor_dir = factor["full_dir"]
    factor_dir.mkdir(parents=True, exist_ok=True)

    # 复制 .code.py
    code_dst = factor_dir / f"{factor_name}.code.py"
    if factor["code_path"].resolve() != code_dst.resolve():
        shutil.copy(factor["code_path"], code_dst)
    print(f"  代码已复制: {code_dst.name}", flush=True)

    # 清除旧结果
    old_parquet = factor_dir / f"{factor_name}.parquet"
    if old_parquet.exists():
        old_parquet.unlink()
    old_result = factor_dir / "result.parquet"
    if old_result.exists():
        old_result.unlink()

    # 运行
    env = os.environ.copy()
    env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
    env["RDAGENT_FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
    env["PYTHONWARNINGS"] = "ignore"

    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(code_dst)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd=str(factor_dir),
    )
    last_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        last_lines.append(line)
        if len(last_lines) > 20:
            last_lines.pop(0)
    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ❌ 运行失败 (code={proc.returncode})", flush=True)
        for line in last_lines[-10:]:
            print(f"    {line}", end="", flush=True)
        return False

    result_parquet = factor_dir / "result.parquet"
    if not result_parquet.exists():
        print(f"  ❌ result.parquet 未生成", flush=True)
        return False

    df = pd.read_parquet(result_parquet)
    print(f"  ✅ 完成: {df.shape[0]}天 x {df.shape[1]}只, {elapsed:.0f}s", flush=True)

    # 重命名
    dst_parquet = factor_dir / f"{factor_name}.parquet"
    if dst_parquet.exists():
        dst_parquet.unlink()
    shutil.move(result_parquet, dst_parquet)
    return True


def post_process(factor: dict, skip_eval: bool = False) -> bool:
    """评估 + 绘图 + meta.json + 复制报告"""
    factor_name = factor["name"]
    factor_dir = factor["full_dir"]
    dst_parquet = factor_dir / f"{factor_name}.parquet"

    if not dst_parquet.exists():
        print(f"  ❌ 找不到 parquet 文件", flush=True)
        return False

    df = pd.read_parquet(dst_parquet)
    print(f"  因子: {df.shape[0]}天 x {df.shape[1]}只, 非空={int(df.notna().sum().sum())}", flush=True)

    if not skip_eval:
        # 评估
        print(f"  评估中...", flush=True)
        eval_script = PROJECT_ROOT / "scripts" / "evaluate_factor.py"
        eval_result = subprocess.run(
            [sys.executable, str(eval_script), str(dst_parquet),
             "--data-dir", str(FULL_DATA_DIR)],
            capture_output=True, text=True, timeout=600,
        )
        if eval_result.returncode == 0:
            for line in eval_result.stdout.splitlines():
                if any(k in line for k in ("IC (Pearson)", "Rank IC", "D1:", "D2:", "D3:", "多空收益", "Sharpe", "Max DD")):
                    print(f"    {line.strip()}", flush=True)

        # 十分组图
        print(f"  生成图表...", flush=True)
        plot_script = PROJECT_ROOT / "scripts" / "plot_decile.py"
        plot_output = factor_dir / f"{factor_name}.decile.png"
        subprocess.run(
            [sys.executable, str(plot_script), str(dst_parquet),
             "--data-dir", str(FULL_DATA_DIR), "--output", str(plot_output)],
            capture_output=True, text=True, timeout=600,
        )
        if plot_output.exists():
            print(f"  ✅ 图表: {plot_output.name}", flush=True)
    else:
        # 没有评估数据时也生成基础meta
        pass

    # ── Barra 暴露分析 ──
    barra_factor_returns = BARRA_DIR / "因子收益率表(Trading Model).csv"
    if barra_factor_returns.exists() and not skip_eval:
        print(f"  Barra 暴露分析...", flush=True)
        try:
            from scripts.barra_evaluate import evaluate_barra
            barra_result = evaluate_barra(df, FULL_DATA_DIR, BARRA_DIR, model="Trading Model")
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
                barra_results_key = f"barra_analysis"
                # 暂存到全局，等 meta 写入时 merge
                _barra_result = barra_result
            else:
                print(f"    ⚠️ {barra_result['error']}", flush=True)
                _barra_result = None
        except Exception as e:
            print(f"    ⚠️ Barra 分析失败: {e}", flush=True)
            _barra_result = None
    else:
        _barra_result = None

    # meta.json — 从测试集meta复制描述信息
    meta_path = factor_dir / f"{factor_name}.meta.json"
    test_meta = {}
    if factor["meta_path"].exists():
        try:
            test_meta = json.loads(factor["meta_path"].read_text())
        except Exception:
            pass
    meta = {
        "factor_name": factor_name,
        "display_name": factor_name,
        "factor_type": factor["factor_type"],
        "factor_description": test_meta.get("factor_description", ""),
        "factor_formulation": test_meta.get("factor_formulation", ""),
        "variables": test_meta.get("variables", {}),
        "source_report_title": test_meta.get("source_report_title", ""),
        "source_report_path": test_meta.get("source_report_path", ""),
        "source_excerpt": test_meta.get("source_excerpt", ""),
        "rows": df.shape[0],
        "stock_count": df.shape[1],
        "non_null": int(df.notna().sum().sum()),
        "time_granularity": "daily",
        "dataset": "full",
        "date_range": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "tags": ["literature_factor", factor["factor_type"], "full_dataset"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if _barra_result is not None:
        meta["barra_analysis"] = _barra_result
    # 如果已有meta文件, 追加保留额外字段
    if meta_path.exists():
        try:
            eval_meta = json.loads(meta_path.read_text())
            # 只保留 meta 中已有的额外字段（如 llm_review, evaluation 等），
            # 不覆盖本次新计算的指标
            for k, v in eval_meta.items():
                if k not in meta:
                    meta[k] = v
        except Exception:
            pass
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✅ meta: {meta_path.name}", flush=True)

    # 复制原始报告
    src_report = factor["code_path"].parent / f"{factor_name}.report.md"
    if src_report.exists():
        shutil.copy(src_report, factor_dir / f"{factor_name}.report.md")
        print(f"  ✅ report: {factor_name}.report.md", flush=True)

    return True


def llm_review_and_decide(factor: dict, meta: dict, ic_threshold: float = 0.01,
                          sharpe_threshold: float = -1.0) -> bool:
    """LLM 审查因子：对比研报原文 vs 代码实现。

    Args:
        factor: 因子信息 dict。
        meta: 已写入的 full meta dict。
        ic_threshold: IC 均值低于此值时触发审查。
        sharpe_threshold: 多空 Sharpe 低于此值时触发审查（负值表示反指）。

    Returns:
        True = 接受因子（代码与原文一致，或原文不可用）；False = 需要人工介入。
    """
    factor_name = factor["name"]
    factor_dir = factor["full_dir"]
    meta_path = factor_dir / f"{factor_name}.meta.json"
    code_path = factor_dir / f"{factor_name}.code.py"

    evaluation = meta.get("evaluation", {})
    ic_mean = evaluation.get("ic_mean")
    ls_sharpe = evaluation.get("long_short_sharpe")

    # 检查是否有评价数据
    if not evaluation or ic_mean is None:
        print(f"  ⚠️ 无有效的评价数据 (IC=ic_mean)，跳过 LLM 审查", flush=True)
        return True

    # source_excerpt 已由 post_process 写入 meta，直接取
    source_excerpt = meta.get("source_excerpt", "")

    # 检查是否需要触发审查
    needs_review = False
    if abs(ic_mean) < ic_threshold:
        needs_review = True
    if ls_sharpe is not None and ls_sharpe < sharpe_threshold:
        needs_review = True

    if not needs_review:
        print(f"  因子效果达标 (IC={ic_mean:.4f}, Sharpe={ls_sharpe:.2f})，跳过 LLM 审查", flush=True)
        return True

    if not source_excerpt:
        print(f"  ⚠️ 无 source_excerpt，无法做 LLM 审查", flush=True)
        return True

    # 执行 LLM 审查
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

    # 写入 meta
    meta["llm_review"] = result
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  LLM 审查: {result.get('verdict')} — {result.get('summary')}", flush=True)

    if result.get("verdict") in ("正确", "部分正确"):
        print(f"  ✅ 代码与原文一致，因子本身效果弱，接受", flush=True)
        return True
    else:
        print(f"  🔄 代码与原文不一致（verdict={result.get('verdict')}），准备重新生成...", flush=True)
        return regenerate_and_rerun(factor, source_excerpt)


def regenerate_and_rerun(factor: dict, source_excerpt: str) -> bool:
    """用 source_excerpt 重新生成因子代码。

    只提取用户函数发给 LLM，避免模板被改 + 节省 token。
    """
    import ast

    factor_name = factor["name"]
    full_dir = factor["full_dir"]
    code_path = factor["full_dir"] / f"{factor_name}.code.py"

    # 如果 full_dir 没有 .code.py，从测试目录复制
    if not code_path.exists():
        test_code = factor["code_path"]
        if test_code.exists():
            import shutil
            shutil.copy(test_code, code_path)

    if not code_path.exists():
        print(f"  ❌ 找不到 .code.py，无法重新生成", flush=True)
        return False

    code = code_path.read_text()

    # 检测因子类型对应的用户函数名
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

    # AST 提取用户函数源码
    tree = ast.parse(code)
    func_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == user_func_name), None)
    if func_node is None:
        print(f"  ❌ AST 解析找不到函数 {user_func_name}", flush=True)
        return False

    lines = code.splitlines()
    old_func_source = "\n".join(lines[func_node.lineno - 1 : func_node.end_lineno])

    # 只发送用户函数给 LLM
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

    # 提取 LLM 返回的函数
    m = re.search(r"```python\s*\n(.*?)\n```", response, re.DOTALL)
    new_func_source = m.group(1).strip() if m else response.strip()

    # 校验 AST
    try:
        ast.parse(new_func_source)
    except SyntaxError as e:
        print(f"  ❌ LLM 生成的代码语法错误: {e}", flush=True)
        return False

    # 注入回原文件
    backup_path = code_path.with_suffix(".code.py.bak")
    code_path.rename(backup_path)

    old_lines = code.splitlines()
    idx_start = func_node.lineno - 1
    idx_end = func_node.end_lineno - 1
    new_lines = new_func_source.splitlines()
    new_code = "\n".join(old_lines[:idx_start] + new_lines + old_lines[idx_end:]) + "\n"

    # 校验整体 AST
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        print(f"  ❌ 注入后整体语法错误，恢复备份: {e}", flush=True)
        code_path.write_text(backup_path.read_text())
        return False

    code_path.write_text(new_code)
    print(f"  ✅ 函数 {user_func_name} 已重写（备份: {backup_path.name}）", flush=True)

    # 重新跑全量
    print(f"  🔄 重新运行全量...", flush=True)
    try:
        if factor["factor_type"] in ("minute", "minute_cs"):
            ok = run_minute_factor(factor, DEFAULT_N_WORKERS)
        else:
            ok = run_other_factor(factor)

        if not ok:
            print(f"  ❌ 重新运行失败，恢复备份", flush=True)
            if backup_path.exists():
                code_path.write_text(backup_path.read_text())
            return False

        ok = post_process(factor, skip_eval=False)
        if ok:
            print(f"  ✅ 重新运行完成", flush=True)
        return ok
    except Exception:
        print(f"  ❌ 重新运行异常，恢复备份代码", flush=True)
        if backup_path.exists():
            code_path.write_text(backup_path.read_text())
        raise


def dry_run_print(pending: list[dict]):
    """打印待跑因子清单"""
    if not pending:
        print("没有因子需要处理。")
        return

    # 按类型分组
    by_type: dict[str, list[dict]] = {}
    for p in pending:
        by_type.setdefault(p["factor_type"], []).append(p)

    type_names = {
        "minute": "分钟线因子",
        "minute_cs": "分钟截面因子",
        "daily": "日线因子",
        "cross_section": "截面因子",
        "deep_learning": "深度学习因子",
    }

    total = len(pending)
    print(f"\n共发现 {total} 个待跑因子:\n")

    for ftype, factors in sorted(by_type.items()):
        label = type_names.get(ftype, ftype)
        print(f"  [{label}] ({len(factors)}个)")
        for f in sorted(factors, key=lambda x: x["report_name"] + "/" + x["name"]):
            status_tag = {
                "new": "新因子",
                "incomplete": f"补跑({f['existing_rows']}/{EXPECTED_ROWS})",
                "force": "强制重跑",
            }.get(f["status"], f["status"])
            print(f"    - {f['report_name']}/{f['name']} [{status_tag}]")
        print()

    print("---")
    print(f"分钟因子: 本地 {DEFAULT_N_WORKERS} workers 并行(列过滤+checkpoint+pool重启)")
    print("日线/截面因子: 本地运行现有 .code.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="全量因子统一运行命令")
    parser.add_argument("--force", action="store_true", help="强制重跑所有因子")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待跑因子，不执行")
    parser.add_argument("--minute-only", action="store_true", help="只跑分钟线因子")
    parser.add_argument("--daily-only", action="store_true", help="只跑日线/截面因子")
    parser.add_argument("--report", default=None, help="只跑指定研报(模糊匹配)")
    parser.add_argument("--workers", type=int, default=DEFAULT_N_WORKERS, help="分钟因子 worker 数")
    parser.add_argument("--skip-sync", action="store_true", help="跳过远程同步")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估/绘图")
    parser.add_argument("--llm-review", action="store_true", help="全量运行后对效果差的因子做LLM审查")
    parser.add_argument("--ic-threshold", type=float, default=0.01, help="IC均值阈值（低于此值触发审查，默认0.01）")
    parser.add_argument("--sharpe-threshold", type=float, default=-1.0, help="多空Sharpe阈值（低于此值触发审查，默认-1.0）")
    args = parser.parse_args()

    t_start = time.time()

    # 预先清理残留worker
    cleanup_workers()

    # 清理测试阶段缓存
    import shutil
    ws = Path.cwd() / "git_ignore_folder" / "RD-Agent_workspace"
    if ws.exists():
        shutil.rmtree(ws)
        print(f"已清理测试缓存: {ws}", flush=True)

    # 扫描
    pending = scan_pending_factors(
        report_filter=args.report,
        force=args.force,
        minute_only=args.minute_only,
        daily_only=args.daily_only,
    )

    # 如果是 dry-run，只打印不执行
    if args.dry_run:
        dry_run_print(pending)
        return 0

    if not pending:
        print("✅ 所有因子已完整，无需处理。")
        return 0

    # 分组
    minute_factors = [f for f in pending if f["factor_type"] in ("minute", "minute_cs")]
    other_factors = [f for f in pending if f["factor_type"] not in ("minute", "minute_cs")]

    print(f"\n{'='*60}")
    print(f"全量因子运行开始")
    print(f"  分钟因子: {len(minute_factors)} 个 ({args.workers} workers)")
    print(f"  日线/截面因子: {len(other_factors)} 个 (本地 joblib)")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0

    # 先跑因子的计算部分
    all_factors = minute_factors + other_factors
    for i, factor in enumerate(all_factors, 1):
        factor_name = factor["name"]
        factor_type = factor["factor_type"]
        status_tag = factor["status"]

        print(f"\n[{i}/{len(all_factors)}] {factor_name}", flush=True)
        print(f"  [{factor_type}] {factor['report_name']} ({status_tag})", flush=True)

        factor_dir = factor["full_dir"]
        factor_dir.mkdir(parents=True, exist_ok=True)

        # 计算
        if factor_type in ("minute", "minute_cs"):
            ok = run_minute_factor(factor, args.workers)
        else:
            ok = run_other_factor(factor)

        if not ok:
            fail_count += 1
            print(f"  ❌ {factor_name} 计算失败", flush=True)
            cleanup_workers(factor_name)
            continue

        # 后处理
        ok = post_process(factor, skip_eval=args.skip_eval)
        if ok:
            success_count += 1
            print(f"  ✅ {factor_name} 完成", flush=True)
        else:
            fail_count += 1
            print(f"  ❌ {factor_name} 后处理失败", flush=True)
            cleanup_workers(factor_name)
            continue

        # LLM 审查（可选，只对后处理成功的因子做）
        if args.llm_review and ok:
            factor_dir = factor["full_dir"]
            meta_path = factor_dir / f"{factor_name}.meta.json"
            if meta_path.exists():
                try:
                    full_meta = json.loads(meta_path.read_text())
                    review_ok = llm_review_and_decide(
                        factor, full_meta,
                        ic_threshold=args.ic_threshold,
                        sharpe_threshold=args.sharpe_threshold,
                    )
                    if not review_ok:
                        print(f"  ⚠️ {factor_name} LLM 审查建议人工复核", flush=True)
                except Exception as e:
                    print(f"  ⚠️ LLM 审查出错: {e}", flush=True)

        # 同步
        if not args.skip_sync:
            print(f"  同步中...", flush=True)
            try:
                sync_to_remote(factor_dir)
            except Exception as e:
                print(f"  ⚠️ 同步失败: {e}", flush=True)

        elapsed_total = time.time() - t_start
        print(f"  ({elapsed_total:.0f}s 累计)", flush=True)

        # 因子间清理
        cleanup_workers(factor_name)
        gc.collect()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"全部完成! {total/60:.1f} 分钟")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
