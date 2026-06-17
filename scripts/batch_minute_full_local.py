#!/usr/bin/env python3
"""批量分钟因子全量计算（本地16核并行 + 评估/绘图 + 同步远程）

用法:
  python3 scripts/batch_minute_full_local.py

功能:
  1. 扫描 literature_reports/ 下所有分钟因子（含 calc_factors_one_day）
  2. 提取用户代码，用最新模板重新包装
  3. FACTOR_N_WORKERS=16 本地全量运行
  4. 评估(IC/IR) + 十分组图
  5. 同步到远程E盘
"""
import ast, gc, json, os, shutil, subprocess, sys, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
LIT_REPORTS_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
FULL_OUTPUT_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"
FULL_DATA_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"

N_WORKERS = 32

# ── 新模板 ──
NEW_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os, gc, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from concurrent.futures import BrokenProcessPool
except ImportError:
    BrokenProcessPool = OSError  # Python <3.11 兼容

warnings.filterwarnings("ignore")

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
MINUTE_BY_DATE_DIR = DATA_DIR / "stock_data" / "minute_by_date"
STOCK_LIST = json.load(open(MINUTE_BY_DATE_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(MINUTE_BY_DATE_DIR / "trade_dates.json"))
LOOKBACK_DAYS = max(1, {lookback_days})

N_WORKERS = int(os.environ.get("FACTOR_N_WORKERS", "1"))

_LOAD_COLS = {_LOAD_COLS_DEF}

def load_day(td):
    return pd.read_parquet(MINUTE_BY_DATE_DIR / f"{td}.parquet", columns=_LOAD_COLS)

_DAILY_DATA_DIR = DATA_DIR / "stock_data" / "daily"
_INDUSTRY_FILE = _DAILY_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {{}}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    import hashlib as _hashlib
    _cache_key = f"jq_{data_type}_{_hashlib.md5(symbol.encode()).hexdigest()[:8]}"
    _cache_path = _DAILY_DATA_DIR / f"{_cache_key}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
    import filelock as _fl
    _lock_path = _DAILY_DATA_DIR / f"{_cache_key}.parquet.lock"
    with _fl.FileLock(str(_lock_path), timeout=120):
        if _cache_path.exists():
            return pd.read_parquet(_cache_path)
        _jq_user = os.environ.get("JQ_USER", "")
        _jq_pass = os.environ.get("JQ_PASS", "")
        if not _jq_user or not _jq_pass:
            raise RuntimeError("JQ_USER/JQ_PASS not set")
        import jqdatasdk as jq
        jq.auth(_jq_user, _jq_pass)
        try:
            if data_type == 'price':
                df = jq.get_price(symbol, start_date=start_date, end_date=end_date, frequency='daily', skip_paused=False, fq='pre')
            elif data_type == 'index_components':
                stocks = jq.get_index_stocks(symbol)
                df = pd.DataFrame({{'stock': stocks}})
            else:
                raise ValueError(f"unsupported data_type: {data_type}")
            if df is not None and not df.empty:
                try:
                    df.to_parquet(_cache_path)
                except OSError:
                    pass
            return df
        finally:
            jq.logout()

{user_code}

def _worker_chunk(chunk_args):
    td_str, stock_list = chunk_args
    results = []
    for stock, df in stock_list:
        try:
            r = calc_factors_one_day(df, stock)
            if r:
                r["datetime"] = str(td_str)
                r["instrument"] = stock
                results.append(r)
        except Exception:
            continue  # 单只股票异常跳过，不影响整批
    return results

def _compute_day_sequential(td):
    idx = TRADE_DATES.index(td)
    start_idx = max(0, idx - LOOKBACK_DAYS + 1)
    lookback_dates = TRADE_DATES[start_idx:idx + 1]
    day_all = pd.concat([load_day(d) for d in lookback_dates])
    results = []
    for stock in day_all.index.get_level_values("instrument").unique():
        day_df = day_all.xs(stock, level="instrument")
        if day_df.empty:
            continue
        r = calc_factors_one_day(day_df, stock)
        if r:
            results.append({{"datetime": str(td), "instrument": stock, **r}})
    return results

if __name__ == '__main__':
    t0 = time.time()
    total_dates = len(TRADE_DATES)
    _CHK_DIR = Path("checkpoints")
    _CHK_INTERVAL = 50

    # ── 断点续跑检测 ──
    _resume_from = 0
    _existing_chks = sorted(_CHK_DIR.glob("chk_*.parquet"))
    if _existing_chks:
        # 从最后一个checkpoint的文件名解析天数
        _last_chk = int(_existing_chks[-1].stem.split("_")[1])
        _resume_from = _last_chk  # 从这天的下一天开始
        print(f"发现已有checkpoint, 从第 {_resume_from+1}/{total_dates} 天续跑", flush=True)

    if N_WORKERS <= 1:
        print("顺序处理（Docker测试模式）...", flush=True)
        all_records = []
        for i, td in enumerate(TRADE_DATES):
            if i < _resume_from:
                continue
            day_records = _compute_day_sequential(td)
            all_records.extend(day_records)
            if (i + 1) % 200 == 0 or i == total_dates - 1:
                rss = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
                print(f"  进度: {i+1}/{total_dates} 天, {len(all_records)} 条, "
                      f"{time.time()-t0:.0f}s, RSS={rss}MB", flush=True)
            # checkpoint
            if (i + 1) % _CHK_INTERVAL == 0 and all_records:
                _cp = _CHK_DIR / f"chk_{i+1:04d}.parquet"
                _CHK_DIR.mkdir(exist_ok=True)
                pd.DataFrame(all_records).to_parquet(_cp)
                print(f"  💾 Chk {i+1}: {len(all_records)} 条 → {_cp.name}", flush=True)
                all_records.clear()
    else:
        print(f"多进程模式: {N_WORKERS} workers, {total_dates} 天", flush=True)
        _all_stocks = STOCK_LIST
        _stock_chunks = [_all_stocks[i::N_WORKERS] for i in range(N_WORKERS)]
        print(f"  共 {len(_all_stocks)} 只股票, "
              f"每chunk ~{min(len(c) for c in _stock_chunks)}-{max(len(c) for c in _stock_chunks)} 只",
              flush=True)

        _CHK_DIR.mkdir(exist_ok=True)
        pool = ProcessPoolExecutor(max_workers=N_WORKERS)
        all_records = []
        try:
            # 初始化滑动缓存
            # 非续跑: 载入前 LOOKBACK_DAYS 天
            # 续跑: 载入续跑点前 LOOKBACK_DAYS 天起到续跑当天 + LOOKBACK_DAYS 天前（模拟正常运行时 i+LOOKBACK_DAYS 的预加载）
            if _resume_from == 0:
                _cache_n = min(LOOKBACK_DAYS, total_dates)
                cache = {TRADE_DATES[j]: load_day(TRADE_DATES[j]) for j in range(_cache_n)}
            else:
                _cache_start = max(0, _resume_from - LOOKBACK_DAYS)
                _cache_end = min(_resume_from + LOOKBACK_DAYS, total_dates - 1)
                cache = {TRADE_DATES[j]: load_day(TRADE_DATES[j])
                         for j in range(_cache_start, _cache_end + 1)}

            for i, td in enumerate(TRADE_DATES):
                if i < _resume_from:
                    continue
                day_start = time.time()
                start_idx = max(0, i - LOOKBACK_DAYS + 1)
                lb_dates = TRADE_DATES[start_idx:i + 1]
                all_data = pd.concat([cache[d] for d in lb_dates])
                _stock_map = {s: g.droplevel("instrument")
                              for s, g in all_data.groupby(level="instrument")}
                del all_data
                chunks = []
                for w_idx, w_stocks in enumerate(_stock_chunks):
                    chunk = [(s, _stock_map[s]) for s in w_stocks if s in _stock_map]
                    if chunk:
                        chunks.append((str(td), chunk))
                del _stock_map
                # 若pool损坏则重建重试
                for _retry in range(3):
                    try:
                        futures = [pool.submit(_worker_chunk, c) for c in chunks]
                        break
                    except (BrokenProcessPool, RuntimeError):
                        print(f"  ⚠️ pool损坏，重建后重试...", flush=True)
                        pool.shutdown(wait=False, cancel_futures=True)
                        pool = ProcessPoolExecutor(max_workers=N_WORKERS)
                        gc.collect()
                else:
                    print(f"  ❌ pool连续损坏3次，跳过当前天", flush=True)
                    continue
                day_count = 0
                for fut in as_completed(futures):
                    try:
                        for r in fut.result():
                            all_records.append(r)
                            day_count += 1
                    except Exception:
                        continue  # 某个chunk整体异常也不崩
                next_idx = i + LOOKBACK_DAYS
                if next_idx < total_dates:
                    cache[TRADE_DATES[next_idx]] = load_day(TRADE_DATES[next_idx])
                if i >= LOOKBACK_DAYS:
                    del cache[TRADE_DATES[i - LOOKBACK_DAYS]]
                rss = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
                print(f"  [{i+1}/{total_dates}] {td}: {day_count} 只, "
                      f"{time.time()-day_start:.1f}s, RSS={rss}MB", flush=True)
                # checkpoint
                if (i + 1) % _CHK_INTERVAL == 0 and all_records:
                    _cp = _CHK_DIR / f"chk_{i+1:04d}.parquet"
                    pd.DataFrame(all_records).assign(
                        datetime=lambda x: pd.to_datetime(x["datetime"])
                    ).to_parquet(_cp)
                    print(f"  💾 Chk {i+1}: {len(all_records)} 条 → {_cp.name}, RSS={rss}MB", flush=True)
                    all_records.clear()
                    # 每50天重启pool，避免fork内存损坏（double free / BrokenProcessPool）
                    pool.shutdown(wait=True)
                    pool = ProcessPoolExecutor(max_workers=N_WORKERS)
                    gc.collect()
                elif (i + 1) % 100 == 0:
                    gc.collect()
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # ── 合并所有checkpoint + 剩余all_records ──
    _chk_files = sorted(_CHK_DIR.glob("chk_*.parquet"))
    if _chk_files:
        _parts = [pd.read_parquet(f) for f in _chk_files]
        if all_records:
            _parts.append(pd.DataFrame(all_records))
        long_df = pd.concat(_parts, ignore_index=True)
        for f in _chk_files:
            f.unlink()
        try:
            _CHK_DIR.rmdir()
        except Exception:
            pass
    elif all_records:
        long_df = pd.DataFrame(all_records)
    else:
        print("警告：没有产生任何因子值！", flush=True)
        long_df = None

    if long_df is not None and not long_df.empty:
        long_df["datetime"] = pd.to_datetime(long_df["datetime"])
        factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
        wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "Date"
        wide.columns.name = "Code"
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide.attrs["factor_name"] = factor_name
        wide.to_parquet("result.parquet")
        print(f"完成！{wide.shape[0]} 天 x {wide.shape[1]} 只股票, "
              f"{time.time()-t0:.0f}s", flush=True)
    os._exit(0)
"""


def extract_user_code(code_path: Path) -> tuple[str, int]:
    """从现有 .code.py 中提取 calc_factors_one_day 函数。
    返回: (函数代码字符串, lookback_days)
    """
    code = code_path.read_text()
    lines = code.splitlines()

    # 提取 lookback_days
    lookback = 1
    for line in lines:
        if "LOOKBACK_DAYS" in line and "max" in line:
            import re
            m = re.search(r'max\(1,\s*(\d+)\)', line)
            if m:
                lookback = int(m.group(1))
            break

    # 用AST提取 calc_factors_one_day
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print(f"  ⚠️ AST解析失败: {code_path}")
        return "", lookback

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "calc_factors_one_day":
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else len(lines)
            func_code = "\n".join(lines[start:end])
            return func_code, lookback

    return "", lookback


def detect_minute_factors() -> list[dict]:
    """扫描 literature_reports/ 下所有分钟因子"""
    factors = []
    for code_path in sorted(LIT_REPORTS_DIR.rglob("*.code.py")):
        code = code_path.read_text()
        if "calc_factors_one_day" not in code:
            continue
        factor_name = code_path.stem.replace(".code", "")
        lookback = 1
        import re
        for line in code.splitlines():
            if "LOOKBACK_DAYS" in line and "max" in line:
                m = re.search(r'max\(1,\s*(\d+)\)', line)
                if m:
                    lookback = int(m.group(1))
                break
        factors.append({
            "name": factor_name,
            "code_path": code_path,
            "report_dir": code_path.parent.parent.name,
            "factor_dir": code_path.parent.name,
            "lookback": lookback,
            "meta_path": code_path.parent / f"{factor_name}.meta.json",
        })
    return factors


def wrap_with_new_template(user_code: str, lookback: int) -> str:
    """用新模板包裹用户代码，自动推断需要的列"""
    import re, pyarrow.parquet as pq
    # 从分钟数据文件读取实际列名，排除索引列
    _MINUTE_BY_DATE_DIR = FULL_DATA_DIR / "stock_data" / "minute_by_date"
    _sample_file = sorted(_MINUTE_BY_DATE_DIR.glob("*.parquet"))[0]
    _all_cols = set(pq.read_schema(_sample_file).names) - {'datetime', 'instrument'}
    _col_pattern = re.compile(r"""\[\s*['"](\w+)['"]\s*\]""")
    _found = {m.group(1) for m in _col_pattern.finditer(user_code)}
    _needed = sorted(_found & _all_cols)
    if _needed:
        cols_def = str(_needed)
    else:
        cols_def = "None  # 未识别到列，加载全部"
    result = (NEW_TEMPLATE
              .replace("{_LOAD_COLS_DEF}", cols_def)
              .replace("{lookback_days}", str(lookback))
              .replace("{user_code}", user_code))
    result = result.replace("{{", "{").replace("}}", "}")
    return result


def _evaluate_and_finish(factor_dir: Path, factor_name: str):
    """对已有结果进行评估/绘图/meta（不重新计算）"""
    dst_parquet = factor_dir / f"{factor_name}.parquet"
    if not dst_parquet.exists():
        rp = factor_dir / "result.parquet"
        if rp.exists():
            shutil.move(rp, dst_parquet)
        else:
            print(f"  ❌ 找不到因子结果文件", flush=True)
            return False

    df = pd.read_parquet(dst_parquet)
    print(f"  因子: {df.shape[0]}天 x {df.shape[1]}只, 非空={int(df.notna().sum().sum())}", flush=True)

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

    # meta.json
    meta_path = factor_dir / f"{factor_name}.meta.json"
    meta = {
        "factor_name": factor_name,
        "display_name": factor_name,
        "rows": df.shape[0],
        "stock_count": df.shape[1],
        "non_null": int(df.notna().sum().sum()),
        "time_granularity": "daily",
        "dataset": "full",
        "date_range": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "tags": ["literature_factor", "minute", "full_dataset", "16workers"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": 0,
    }
    existing_meta = meta_path
    if existing_meta.exists():
        try:
            eval_meta = json.loads(existing_meta.read_text())
            meta.update(eval_meta)
        except Exception:
            pass
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return True


def run_factor(factor: dict, factor_dir: Path) -> bool:
    """运行一个因子：写代码 → 本地执行 → 评估 → 绘图"""
    factor_name = factor["name"]
    print(f"\n{'='*60}", flush=True)
    print(f"因子: {factor_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # 检查是否已有完整结果 — 跳过计算和评估
    dst_parquet = factor_dir / f"{factor_name}.parquet"
    if dst_parquet.exists() and (factor_dir / f"{factor_name}.meta.json").exists():
        print(f"  已有完整结果，跳过", flush=True)
        return True
    if dst_parquet.exists() or (factor_dir / "result.parquet").exists():
        print(f"  有结果文件但无meta，补meta...", flush=True)
        return _evaluate_and_finish(factor_dir, factor_name)

    # 1. 提取并重新包装
    user_code, lookback = extract_user_code(factor["code_path"])
    if not user_code:
        print(f"  ❌ 无法提取 calc_factors_one_day", flush=True)
        return False

    wrapped = wrap_with_new_template(user_code, lookback)
    code_dst = factor_dir / f"{factor_name}.code.py"
    code_dst.write_text(wrapped)
    print(f"  代码已重新包装: {code_dst.name}", flush=True)

    # 2. 本地执行
    env = os.environ.copy()
    env["FACTOR_DATA_DIR"] = str(FULL_DATA_DIR)
    env["FACTOR_N_WORKERS"] = str(N_WORKERS)
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
    # 杀该因子的残留worker进程
    cleanup_workers(factor_name)

    if proc.returncode != 0:
        print(f"  ⚠️ 运行异常退出 (code={proc.returncode})", flush=True)
        # 尝试合并checkpoint（即使崩溃也可能有部分数据）
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
        else:
            for line in last_lines[-10:]:
                print(f"    {line}", end="", flush=True)
            print(f"  ❌ 运行失败，无checkpoint可恢复", flush=True)
            return False

    result_parquet = factor_dir / "result.parquet"
    if not result_parquet.exists():
        print(f"  ❌ result.parquet 未生成", flush=True)
        return False

    df = pd.read_parquet(result_parquet)
    print(f"  ✅ 完成: {df.shape[0]}天 x {df.shape[1]}只, {elapsed:.0f}s", flush=True)

    # 3. 重命名
    dst_parquet = factor_dir / f"{factor_name}.parquet"
    if dst_parquet.exists():
        dst_parquet.unlink()
    shutil.move(result_parquet, dst_parquet)

    # 4. 评估
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

    # 5. 十分组图
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
        "factor_description": test_meta.get("factor_description", ""),
        "factor_formulation": test_meta.get("factor_formulation", ""),
        "variables": test_meta.get("variables", {}),
        "source_report_title": test_meta.get("source_report_title", ""),
        "source_report_path": test_meta.get("source_report_path", ""),
        "rows": df.shape[0],
        "stock_count": df.shape[1],
        "non_null": int(df.notna().sum().sum()),
        "time_granularity": "daily",
        "dataset": "full",
        "date_range": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "tags": ["literature_factor", "minute", "full_dataset", "32workers"],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed),
    }
    eval_meta_path = factor_dir / f"{factor_name}.meta.json"
    if eval_meta_path.exists():
        try:
            eval_meta = json.loads(eval_meta_path.read_text())
            meta.update(eval_meta)
        except Exception:
            pass
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 7. 复制原始报告
    src_report = factor["code_path"].parent / f"{factor_name}.report.md"
    if src_report.exists():
        shutil.copy(src_report, factor_dir / f"{factor_name}.report.md")

    return True


def sync_to_remote(factor_dir: Path):
    """同步到远程E盘"""
    try:
        from scripts.sync_utils import ensure_remote_mounted, REMOTE_BASE_FULL as REMOTE_BASE
    except ModuleNotFoundError:
        from sync_utils import ensure_remote_mounted, REMOTE_BASE_FULL as REMOTE_BASE

    if not ensure_remote_mounted():
        print(f"  ⚠️ 远程不可用，跳过同步", flush=True)
        return False

    remote_parent = REMOTE_BASE / factor_dir.parent.name
    remote_dir = remote_parent / factor_dir.name
    remote_dir.mkdir(parents=True, exist_ok=True)

    for f in factor_dir.iterdir():
        if f.is_file() and f.suffix in (".parquet", ".py", ".json", ".png", ".md"):
            shutil.copy2(f, remote_dir / f.name)

    print(f"  ✅ 已同步到远程: {remote_dir}", flush=True)
    return True


def cleanup_workers(factor_name: str = None):
    """清除可能残留的worker进程，防止因子间内存不释放"""
    import subprocess as _sp
    # 杀卡死的进程（SIGKILL确保杀掉D/Z状态进程）
    _sp.run(["pkill", "-9", "-f", "python3 -c  import os"], capture_output=True, timeout=5)
    # 如果有指定因子名，杀该因子的所有残留进程（包括worker orphan）
    if factor_name:
        _sp.run(["pkill", "-9", "-f", f"{factor_name}.code.py"], capture_output=True, timeout=5)
    gc.collect()


def main():
    t_start = time.time()

    # 启动前清理所有可能的残留worker进程
    cleanup_workers()

    # 扫描分钟因子
    factors = detect_minute_factors()
    print(f"找到 {len(factors)} 个分钟因子", flush=True)
    for f in factors:
        print(f"  {f['name']} (lookback={f['lookback']})", flush=True)

    if not factors:
        print("没有分钟因子需要处理", flush=True)
        return 0

    success_count = 0
    fail_count = 0

    for i, factor in enumerate(factors, 1):
        factor_name = factor["name"]
        factor_dir = FULL_OUTPUT_BASE / factor["report_dir"] / factor["factor_dir"]
        factor_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(factors)}] 处理: {factor_name}", flush=True)

        ok = run_factor(factor, factor_dir)
        if ok:
            success_count += 1
            print(f"  同步中...", flush=True)
            sync_to_remote(factor_dir)
        else:
            fail_count += 1

        print(f"  ({time.time()-t_start:.0f}s 累计)", flush=True)
        # 因子间清理：杀残留worker + gc
        cleanup_workers(factor_name)
        gc.collect()

    total = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"全部完成! {total/60:.1f} 分钟", flush=True)
    print(f"  成功: {success_count}", flush=True)
    print(f"  失败: {fail_count}", flush=True)
    print(f"{'='*60}", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
