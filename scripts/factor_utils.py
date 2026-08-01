#!/usr/bin/env python3
"""
因子工具函数：run_all.py 和 daily_update.py 共享的实用函数。

提供统一的数据加载、代码注入、子进程执行、合并、评估等操作。
"""

import json
import os
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


def detect_factor_type(code_text: str) -> str:
    """从代码文本判断因子类型"""
    if any(k in code_text for k in ('MINUTE_BY_DATE_DIR', 'minute_pv', 'calc_factors_one_day')):
        if 'cross_section' in code_text.lower() or 'calc_factor_minute_raw' in code_text:
            return "minute_cross_section"
        return "minute"
    if 'cross_section' in code_text.lower() or 'calc_factor_cross_section' in code_text:
        return "cross_section"
    return "daily"


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
            for line in proc.stdout.split("\n"):
                line = line.strip()
                if line:
                    print(f"    {line}")
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


def merge_incremental_result(existing_df: pd.DataFrame, new_df: pd.DataFrame, last_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    合并增量结果：裁掉重叠，concat + 去重 + sort。

    如果 last_date 不为 None，先只保留 new_df 中 date > last_date 的行。
    """
    # 统一 index 为 DatetimeIndex
    if not isinstance(new_df.index, pd.DatetimeIndex):
        new_df.index = pd.to_datetime(new_df.index)
    if not isinstance(existing_df.index, pd.DatetimeIndex):
        existing_df.index = pd.to_datetime(existing_df.index)

    if last_date is not None:
        new_df = new_df[new_df.index > last_date]

    if new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.sort_index(inplace=True)
    return combined


def backup_parquet(parquet_path: Path) -> Path:
    """创建 .parquet.bak 备份，返回备份路径。

    备份只是写回 parquet 时的临时安全网（写盘失败可恢复）。
    增量更新全部成功后必须调用 cleanup_parquet_backup() 删除，
    否则因子目录会残留 .parquet.bak.*，看着像重复文件。
    """
    bak_path = parquet_path.with_suffix(
        f".parquet.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
    meta["daily_update"] = True

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
