#!/usr/bin/env python3
"""
因子工具函数：run_all.py 使用的实用函数。

提供统一的数据加载、代码注入、子进程执行、合并、评估等操作。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def load_trade_dates(data_dir: Path) -> list[str]:
    """从数据目录读取交易日列表"""
    for p in [
        data_dir / "stock_data" / "daily" / "trade_dates.json",
        data_dir / "stock_data" / "minute_by_date" / "trade_dates.json",
    ]:
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"trade_dates.json not found (search: {data_dir})")


# 分钟模板无条件注入的标记行。实测 61 个分钟因子 61 命中、日线/截面因子 0 命中，
# 是唯一零误判的分钟判据（cross_section 模板含 "minute_by_date" 字面量但**不含**这行赋值）。
_MINUTE_MARKER = re.compile(r"^MINUTE_BY_DATE_DIR\s*=\s*DATA_DIR", re.M)


def detect_factor_type(code_text: str) -> str:
    """从代码文本判断因子类型（仅作兜底，权威类型见 resolve_factor_type）。

    两条铁律：
    1. 必须用 `def ` 锚定的正则：模板注入的 _COLLECT_COLS_SRC 样板里含
       `calc_factors_one_day` / `MINUTE_BY_DATE_DIR` 等**字面量**（注释 + 元组），
       裸子串会把 daily / cross_section 因子误判成 minute（实测 985/1155 误判）。
    2. **分钟标记必须排在 `calc_factor_series` 分支之前**。分钟因子常只写向量化的
       `calc_factor_series`（phase2_code.md 明确要求 lookback 大的分钟因子走向量化），
       不写 `calc_factors_one_day`；若 series 分支先返回 daily，这类分钟因子会被
       误判成 daily，run_all 于是用日线目录跑分钟因子 →
       FileNotFoundError: stock_data/minute_by_date/stock_list.json。
       曾实测 17 个因子因此失败（2026-10-02 / 2026-10-03）。
    """
    if re.search(r"\bdef\s+calc_factor_cross_section\s*\(", code_text):
        return "cross_section"
    if re.search(r"\bdef\s+train_model\s*\(", code_text) or re.search(r"\bdef\s+predict\s*\(", code_text):
        return "deep_learning"
    # ── 分钟标记先于 series 分支 ──
    # 分钟截面钩子（cross_section_transform）与旧入口 calc_factor_minute_raw 都是分钟信号：
    # 它们只在分钟模板里有意义，且向量化分钟因子可能不带 MINUTE_BY_DATE_DIR 字面量。
    if (_MINUTE_MARKER.search(code_text)
            or re.search(r"\bdef\s+cross_section_transform\s*\(", code_text)
            or re.search(r"\bdef\s+calc_factor_minute_raw\s*\(", code_text)):
        return "minute"
    if re.search(r"\bdef\s+calc_factors_one_day\s*\(", code_text):
        return "minute"
    if re.search(r"\bdef\s+calc_factor_series\s*\(", code_text) or \
       re.search(r"\bdef\s+calc_factor_single_stock\s*\(", code_text):
        return "daily"
    return "daily"


# ── 权威类型解析：定义阶段 → 部署 meta → 代码兜底 ──

EXTRACTED_REPORTS_DIR = (
    Path("/mnt/d/paper-factor-data") / "数据仓库" / "因子产出" / "extracted_reports"
)

# 类型取值 → 内部类型。
# ⚠️ minute_cs / minute_cross_section 是**历史遗留值**，只用于兼容旧 meta.json：
# 该类型已废除（分钟截面统一由 minute 模板的可选 cross_section_transform 钩子承担），
# 定义阶段不会再产出这两个值，仅解析历史产物时映射回 minute。
_TYPE_ALIASES = {
    "daily": "daily",
    "daily_single": "daily",
    "minute": "minute",
    "minute_cs": "minute",
    "minute_cross_section": "minute",
    "cross_section": "cross_section",
    "deep_learning": "deep_learning",
}


def load_authoritative_types(date_str: str) -> dict:
    """读 extracted_reports/{date}/*.extracted.json，返回 {报告名: {因子名: 类型}}。

    这是定义 agent 判定的权威类型。测试与全量阶段都应以它为准，
    而不是各自从代码文本重新猜（曾因此出现「测试用参数、全量靠猜」的分裂）。
    """
    out: dict[str, dict[str, str]] = {}
    base = EXTRACTED_REPORTS_DIR / date_str
    if not base.is_dir():
        return out
    for p in base.glob("*.extracted.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        report = d.get("report_name") or p.name[: -len(".extracted.json")]
        per = {}
        for f in d.get("factors", []):
            name, t = f.get("name"), f.get("type")
            if name and t:
                per[name] = t
        out[report] = per
    return out


def resolve_factor_type(
    code_text: str,
    factor_name: str = "",
    report_name: str = "",
    meta: dict | None = None,
    auth_types: dict | None = None,
) -> str:
    """解析因子的权威类型，优先级从高到低：

      1. 部署 meta.json 里的 `factor_type`（测试阶段验证时记录的类型，最可信）
      2. 代码文本检测（反映 worker **最终采用**的实现）
      3. extracted_reports 的定义阶段类型（最初的语义判断）

    ⚠️ 第 2、3 的顺序很关键：encode 阶段的 worker 可能修正定义阶段的类型
    （例如维度上的定义被实现成分钟截面）。历史 extracted.json 里这类修正**没有回写**，
    于是 extracted 与代码分叉（实测 IndRankLateVol：extracted=cross_section，
    代码/实际=minute）。代码反映的是真正跑出来的东西，因此排在 extracted 之前；
    两者冲突时打印告警，便于回查定义阶段是否需要纠正。

    返回内部类型（daily / minute / cross_section / deep_learning）。
    """
    code_t = detect_factor_type(code_text)

    # 1. meta
    if meta:
        mt = (meta.get("factor_type") or "").strip()
        if mt in _TYPE_ALIASES:
            m = _TYPE_ALIASES[mt]
            if m != code_t:
                print(f"  ⚠️ 类型分歧 [{report_name}/{factor_name}]："
                      f"meta={m} 但代码检测={code_t}（以 meta 为准）", flush=True)
            return m

    # 2. 代码
    if auth_types and report_name:
        ext = (auth_types.get(report_name) or {}).get(factor_name, "")
        if ext in _TYPE_ALIASES:
            e = _TYPE_ALIASES[ext]
            if e != code_t:
                print(f"  ⚠️ 类型分歧 [{report_name}/{factor_name}]："
                      f"extracted={e} 但代码检测={code_t}（以代码为准；"
                      f"若定义阶段判错，请回写 extracted.json）", flush=True)
    return code_t


def run_factor_subprocess(
    code_text: str,
    factor_name: str,
    data_dir: Path,
    start_date: str | None = None,
    n_workers: int | None = None,
    timeout: int = 7200,
) -> Path | None:
    """
    子进程执行 .code.py。

    将 code_text 写入临时目录的 {factor_name}.py，设环境变量后执行。
    模板已内置 FACTOR_INCREMENTAL_START_DATE 过滤逻辑，设环境变量即可增量。
    返回生成的 {factor_name}.parquet 路径，失败返回 None。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_code = tmpdir / f"{factor_name}.py"

        # 增量更新优化：分钟因子用独立 chunk 目录，避免重算全部 2000+ 天数据
        if start_date:
            code_text = re.sub(
                r'''_CHUNK_DIR\s*=\s*MINUTE_BY_DATE_DIR\s*/\s*"_minute_chunks"''',
                '''_CHUNK_DIR = MINUTE_BY_DATE_DIR / ("_minute_chunks_incr" if os.environ.get("FACTOR_INCREMENTAL_START_DATE") else "_minute_chunks")''',
                code_text,
            )

        tmp_code.write_text(code_text, encoding="utf-8")

        env = {k: str(v) for k, v in os.environ.items()}
        env["FACTOR_DATA_DIR"] = str(data_dir)
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        if start_date:
            env["FACTOR_INCREMENTAL_START_DATE"] = start_date
        if n_workers is not None:
            env["FACTOR_N_WORKERS"] = str(n_workers)
        else:
            env.setdefault("FACTOR_N_WORKERS", "4")
        env["FACTOR_LOOKBACK_CAP"] = "99999"

        factor_type = detect_factor_type(code_text)
        print(f"  执行中... (type={factor_type})"
              f"{f', start={start_date}' if start_date else ''}")

        try:
            proc = subprocess.run(
                [sys.executable, f"{factor_name}.py"],
                cwd=tmpdir,
                capture_output=True, text=True, timeout=timeout,
                env=env,
            )
            # 子进程完整日志留在 /tmp/{factor}.incr.log，不全量回显：
            # 分钟因子日志实测可刷到数千行，逐行回显会灌进 agent 上下文。
            try:
                Path(f"/tmp/{factor_name}.incr.log").write_text(
                    (proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
            except OSError:
                pass
            if proc.returncode != 0:
                stderr = proc.stderr[-500:] if len(proc.stderr) > 500 else proc.stderr
                print(f"  ❌ 执行失败: {stderr.strip()}")
                return None
        except subprocess.TimeoutExpired:
            print(f"  ❌ 执行超时（{timeout}s）")
            return None
        except Exception as e:
            print(f"  ❌ 执行异常: {e}")
            return None

        result_parquet = tmpdir / f"{factor_name}.parquet"
        if not result_parquet.exists():
            print(f"  ❌ 未生成 {factor_name}.parquet")
            return None

        # 读取 parquet（tmpdir 会在上下文退出时被清理，所以先读入内存并保存到固定路径）
        # 需要把 parquet 读到内存再写到外部临时路径
        out_path = Path(tempfile.mktemp(suffix=".parquet"))
        shutil.copy2(result_parquet, out_path)
        return out_path


def merge_incremental_result(existing_df: pd.DataFrame, new_df: pd.DataFrame, last_date: pd.Timestamp | None = None,
                             trade_dates: list | None = None) -> pd.DataFrame:
    """
    合并增量结果：裁掉重叠，concat + 去重 + sort。

    如果 last_date 不为 None，先只保留 new_df 中 date > last_date 的行。
    如果 trade_dates 提供（权威交易日历），合并后裁掉不在日历内的行（清周末幽灵行）：
    - new_df 里非交易日（周末/非交易日）行直接丢弃
    - existing_df 里同样保留日历内行（旧数据含周末脏行时一并清除）
    """
    # 统一 index 为 DatetimeIndex
    if not isinstance(new_df.index, pd.DatetimeIndex):
        new_df.index = pd.to_datetime(new_df.index)
    if not isinstance(existing_df.index, pd.DatetimeIndex):
        existing_df.index = pd.to_datetime(existing_df.index)

    if last_date is not None:
        new_df = new_df[new_df.index > last_date]

    if not new_df.empty and trade_dates:
        _cal_set = set(pd.DatetimeIndex(trade_dates))
        new_df = new_df[new_df.index.isin(_cal_set)]
        existing_df = existing_df[existing_df.index.isin(_cal_set)]

    if new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)

    if trade_dates:
        _cal = pd.DatetimeIndex(sorted(trade_dates))
        _extra = combined.index.difference(_cal)
        if len(_extra):
            combined = combined.drop(index=_extra)

    return combined


def backup_parquet(parquet_path: Path) -> Path:
    """创建 .parquet.bak 备份，返回备份路径。

    备份只是写回 parquet 时的临时安全网（写盘失败可恢复）。
    增量更新全部成功后必须调用 cleanup_parquet_backup() 删除，
    否则因子目录会残留 .parquet.bak.*，看着像重复文件。
    """
    bak_path = parquet_path.with_suffix(
        f".parquet.bak.{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    )
    shutil.copy2(parquet_path, bak_path)
    return bak_path


def cleanup_parquet_backup(parquet_path: Path) -> None:
    """删除因子目录下该因子的所有 .parquet.bak.* 备份文件。

    增量更新成功后调用，让因子目录"看着就和原来一样"：
    只保留 .code.py / .parquet / .meta.json / .decile.png / .report.md，
    不残留任何中间产物（包括之前崩溃运行遗留的旧备份）。
    """
    for bak in parquet_path.parent.glob(f"{parquet_path.stem}.parquet.bak.*"):
        try:
            bak.unlink()
        except OSError:
            pass


def update_factor_meta(meta_path: Path, df: pd.DataFrame, extra: dict | None = None) -> dict:
    """更新因子的 meta.json，返回 meta dict"""
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    meta["date_range"] = f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}"
    meta["rows"] = df.shape[0]
    meta["stock_count"] = df.shape[1]
    meta["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if extra:
        meta.update(extra)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def evaluate_factor(parquet_path: Path, factor_name: str, output_dir: Path, data_dir: Path) -> dict | None:
    """
    评估因子 + 生成分位数图。

    调用 evaluate_factor.py 和 plot_decile.py 作为子进程。
    返回评估结果 dict，失败返回 None。
    """
    eval_script = PROJECT_ROOT / "scripts" / "evaluate_factor.py"
    plot_script = PROJECT_ROOT / "scripts" / "plot_decile.py"

    result = {}

    # 评估
    if eval_script.exists():
        print(f"  评估中...", flush=True)
        try:
            eval_result = subprocess.run(
                [sys.executable, str(eval_script), str(parquet_path),
                 "--data-dir", str(data_dir)],
                capture_output=True, text=True, timeout=600,
            )
            if eval_result.returncode == 0:
                for line in eval_result.stdout.split("\n"):
                    line = line.strip()
                    if line and any(k in line for k in ("IC (Pearson)", "Rank IC", "Sharpe", "IC=")):
                        print(f"    {line}")
                    # 尝试解析 evaluation.json 输出
                eval_json = parquet_path.with_name(f"{factor_name}.meta.json")
                if eval_json.exists():
                    try:
                        meta = json.loads(eval_json.read_text(encoding="utf-8"))
                        result = meta.get("evaluation", {})
                    except Exception:
                        pass
            else:
                print(f"    ⚠️ 评估脚本失败 (exit={eval_result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ 评估超时")
        except Exception as e:
            print(f"    ⚠️ 评估异常: {e}")

    # 绘图
    if plot_script.exists():
        plot_output = output_dir / f"{factor_name}.decile.png"
        print(f"  生成图表...", flush=True)
        try:
            plot_result = subprocess.run(
                [sys.executable, str(plot_script), str(parquet_path),
                 "--data-dir", str(data_dir), "--output", str(plot_output)],
                capture_output=True, text=True, timeout=600,
            )
            if plot_result.returncode == 0:
                print(f"  图表已保存: {plot_output}")
            else:
                print(f"    ⚠️ 图表生成失败 (exit={plot_result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ 绘图超时")
        except Exception as e:
            print(f"    ⚠️ 绘图异常: {e}")

    return result if result else None
