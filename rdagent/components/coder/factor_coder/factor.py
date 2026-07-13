from __future__ import annotations

import hashlib
import json
import os
import site
import subprocess
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union

import pandas as pd
import docker  # type: ignore[import-untyped]
from filelock import FileLock

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.utils.env import DockerConf, DockerEnv


class FactorTask(CoSTEERTask):
    # factor_type: "daily_single" | "cross_section" | "minute" | "minute_cross_section" | "deep_learning"
    FACTOR_TYPE_SINGLE = "daily_single"
    FACTOR_TYPE_CROSS = "cross_section"
    FACTOR_TYPE_MINUTE = "minute"
    FACTOR_TYPE_MINUTE_CROSS = "minute_cross_section"
    FACTOR_TYPE_DL = "deep_learning"

    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        factor_type: str = "daily_single",
        lookback_days: int = 0,
        special_conditions: str = "",
        source_excerpt: str = "",
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        self.factor_type = factor_type
        self.lookback_days = lookback_days
        self.special_conditions = special_conditions
        self.source_excerpt = source_excerpt
        self.llm_review: dict | None = None  # 测试阶段LLM审查结果，全量阶段复用
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """for compatibility"""
        return self.description

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_type: {getattr(self, 'factor_type', 'daily_single')}
lookback_days: {getattr(self, 'lookback_days', 0)}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}
special_conditions: {getattr(self, 'special_conditions', '')}"""

    def get_task_brief_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}
special_conditions: {getattr(self, 'special_conditions', '')}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorDockerConf(DockerConf):
    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent / "docker"
    image: str = FACTOR_COSTEER_SETTINGS.docker_image
    mount_path: str = "/workspace/factor_workspace"
    default_entry: str = "python _rdagent_factor_launcher.py"
    enable_cache: bool = False
    shm_size: str | None = "16g"
    mem_limit: str | None = "48g"
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 20
    running_timeout_period: int | None = 600  # 10 minutes, was 3600


class FactorDockerEnv(DockerEnv):
    def __init__(self, conf: DockerConf | None = None):
        super().__init__(conf or FactorDockerConf())

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        force_build = os.environ.get("FACTOR_CoSTEER_FORCE_DOCKER_BUILD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if not force_build:
            try:
                docker.from_env().images.get(self.conf.image)
                return
            except docker.errors.ImageNotFound:
                pass
        super().prepare(*args, **kwargs)


def _conda_env_exists(env_name: str) -> bool:
    result = subprocess.run(
        f"conda env list | grep -q '^{env_name} '",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _docker_daemon_available() -> bool:
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# ========== 模板间共享代码段 ==========

_LOAD_MINUTE_STOCK_SRC = r'''
def load_minute_stock(stock, columns=None):
    """加载分钟数据，支持列过滤"""
    path = MINUTE_DATA_DIR / f"{stock}.parquet"
    df = pd.read_parquet(path, columns=columns)
    if "datetime" in df.columns:
        df.index = pd.to_datetime(df.pop("datetime"))
    elif df.index.name == "datetime":
        df.index = pd.to_datetime(df.index)
    return df
'''

# ========== 类定义开始 ==========

class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."
    EXPORTED_PARQUET_DIR = Path.cwd() / "git_ignore_folder" / "factor_outputs"
    EXECUTION_LAUNCHER = "_rdagent_factor_launcher.py"

    # 日线框架代码模板
    DAILY_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path
from joblib import Parallel, delayed
DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))
LOOKBACK_DAYS = {lookback_days}  # 由框架注入，0=不切片

def load_stock(stock, columns=None):
    if columns:
        return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet", columns=columns)
    return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet")

# 行业分类数据（申万一级行业）：INDUSTRY_DICT[股票代码] = 行业名
_INDUSTRY_FILE = STOCK_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {{}}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    \"\"\"通用聚宽数据获取函数。优先读本地缓存，没有再通过聚宽在线下载。
    本地数据中已有的字段（如日频价量、基本面等）直接走本地，不会调用聚宽。
    用法:
      idx = get_jq_data('000300.XSHG', 'price')  # 指数行情
      stocks = get_jq_data('000905.XSHG', 'index_components')  # 中证500成分股列表
    data_type 支持: 'price'(行情), 'index_components'(指数成分股)
    \"\"\"
    import hashlib as _hashlib
    _cache_key = f"jq_{{data_type}}_{{_hashlib.md5(symbol.encode()).hexdigest()[:8]}}"
    _cache_path = STOCK_DATA_DIR / f"{{_cache_key}}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
    # 文件锁防止并发 JQData 连接数超限（账号最多3个连接）
    import filelock as _fl
    _lock_path = STOCK_DATA_DIR / f"{{_cache_key}}.parquet.lock"
    with _fl.FileLock(str(_lock_path), timeout=120):
        if _cache_path.exists():
            return pd.read_parquet(_cache_path)
        _jq_user = os.environ.get("JQ_USER", "")
        _jq_pass = os.environ.get("JQ_PASS", "")
        if not _jq_user or not _jq_pass:
            raise RuntimeError("JQ_USER/JQ_PASS 环境变量未设置，无法通过聚宽获取数据")
        import jqdatasdk as jq
        jq.auth(_jq_user, _jq_pass)
        try:
            if data_type == 'price':
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TErr
                _tp = _TPE(max_workers=1)
                _tf = _tp.submit(jq.get_price, symbol, start_date=start_date, end_date=end_date, frequency='daily', skip_paused=False, fq='pre')
                try:
                    df = _tf.result(timeout=180)
                except _TErr:
                    print(f"JQData get_price timeout (180s), symbol={symbol}", flush=True)
                    df = pd.DataFrame()
                finally:
                    _tp.shutdown(wait=False)
            elif data_type == 'index_components':
                stocks = jq.get_index_stocks(symbol)
                df = pd.DataFrame({{'stock': stocks}})
            else:
                raise ValueError(f"unsupported data_type: {{data_type}}")
            if df is not None and not df.empty:
                try:
                    df.to_parquet(_cache_path)
                except OSError:
                    pass
            return df
        finally:
            jq.logout()

{user_code}

def _compute_stock(stock, _LOAD_COLS=None):
    df = load_stock(stock, _LOAD_COLS)
    if df.empty:
        return []
    results = []
    _td_index = pd.DatetimeIndex(TRADE_DATES)
    if LOOKBACK_DAYS > 0:
        # 批量预计算所有日期的切片位置（一次 searchsorted 调用）
        _positions = np.searchsorted(df.index.values.astype('int64'), _td_index.values.astype('int64'), side='right')
        for i, td in enumerate(_td_index):
            pos = int(_positions[i])
            if pos == 0:
                continue
            start = max(0, pos - LOOKBACK_DAYS - 1)  # +1 buffer for diff/shift
            sub = df.iloc[start:pos]
            try:
                r = calc_factor_single_stock(sub, td, stock)
            except Exception:
                r = None
            if r:
                results.append({{"datetime": str(td.date()), "instrument": stock, **r}})
    else:
        _positions_all = np.searchsorted(df.index.values.astype('int64'), _td_index.values.astype('int64'), side='right')
        for i, td in enumerate(_td_index):
            pos = int(_positions_all[i])
            if pos == 0:
                continue
            sub = df.iloc[:pos]
            try:
                r = calc_factor_single_stock(sub, td, stock)
            except Exception:
                r = None
            if r:
                results.append({{"datetime": str(td.date()), "instrument": stock, **r}})
    return results

if __name__ == '__main__':
    try:
        # ── 自动列推断：分析用户函数，只加载需要的列 ──
        import re as _re, inspect as _inspect, pyarrow.parquet as _pq
        _SAMPLE_FILE = next(STOCK_DATA_DIR.glob("*.parquet"))
        _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {'datetime', 'instrument'}
        _USER_SOURCE = _inspect.getsource(calc_factor_single_stock)
        # 提取代码中所有引号字符串，与可用列取交集（覆盖 df['col']、.get_group()['col']、.columns 等所有模式）
        _ALL_QUOTED = set(_re.findall(r'''['\"](\w+)['\"]''', _USER_SOURCE))
        _LOAD_COLS = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
        if not _LOAD_COLS:
            _LOAD_COLS = None
        print(f"检测到因子使用的列: {_LOAD_COLS}", flush=True)
        # ──

        N_JOBS = int(os.environ.get("FACTOR_N_WORKERS", "4"))  # 日线因子4核足够，避免多因子并行时OOM
        print(f"计算因子 (n_jobs={N_JOBS}), {len(STOCK_LIST)} stocks...", flush=True)
        all_records = []
        _total = len(STOCK_LIST)
        for _idx, stock_results in enumerate(Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_compute_stock)(s, _LOAD_COLS) for s in STOCK_LIST
        )):
            all_records.extend(stock_results)
            if (_idx + 1) % 500 == 0 or (_idx + 1) == _total:
                print(f"  进度: {_idx+1}/{_total}", flush=True)
        long_df = pd.DataFrame(all_records)
        long_df["datetime"] = pd.to_datetime(long_df["datetime"])
        factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
        wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "trade_date"
        wide.columns.name = "stock_code"
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide.attrs["factor_name"] = factor_name
        # 涨停剔除
        _LU_PATH = DATA_DIR / "limit_up_daily.parquet"
        if _LU_PATH.exists():
            _lu_df = pd.read_parquet(_LU_PATH, columns=['datetime', 'instrument'])
            for _lu_dt, _lu_grp in _lu_df.groupby(_lu_df['datetime'].dt.normalize()):
                if _lu_dt in wide.index:
                    _c = [str(x) for x in _lu_grp['instrument'] if str(x) in wide.columns]
                    if _c:
                        wide.loc[_lu_dt, _c] = np.nan
        # /涨停剔除
        # 统一格式：index→string日期, columns→int股票代码
        wide.index = wide.index.strftime('%Y-%m-%d')
        wide.columns = wide.columns.astype(int)
        wide.to_parquet("result.parquet")
        print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}}, 只股票")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)
"""

    # 分钟线框架代码模板（按日期并行，每个日期一个 parquet，MultiIndex(instrument, datetime)）
    MINUTE_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os, gc, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
pd.set_option("mode.copy_on_write", False)

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
MINUTE_BY_DATE_DIR = DATA_DIR / "stock_data" / "minute_by_date"
STOCK_LIST = json.load(open(MINUTE_BY_DATE_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(MINUTE_BY_DATE_DIR / "trade_dates.json"))
LOOKBACK_DAYS = max(1, {lookback_days})  # 分钟线至少1天

# 并行控制：默认8核（fork COW内存安全），环境变量FACTOR_N_WORKERS覆盖
N_WORKERS = int(os.environ.get("FACTOR_N_WORKERS", "8"))

# 列过滤（由LLM推断，不含datetime等索引列），建议只加载因子所需的列以减少内存
{_LOAD_COLS_DEF}

def load_day(td):
    return pd.read_parquet(MINUTE_BY_DATE_DIR / f"{td}.parquet", columns=_LOAD_COLS)

# 行业分类数据（申万一级行业）：INDUSTRY_DICT[股票代码] = 行业名
_DAILY_DATA_DIR = DATA_DIR / "stock_data" / "daily"
_INDUSTRY_FILE = _DAILY_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    import hashlib as _hashlib
    _cache_key = f"jq_{data_type}_{_hashlib.md5(symbol.encode()).hexdigest()[:8]}"
    _cache_path = _DAILY_DATA_DIR / f"{_cache_key}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
    # 文件锁防止并发 JQData 连接数超限（账号最多3个连接）
    import filelock as _fl
    _lock_path = _DAILY_DATA_DIR / f"{_cache_key}.parquet.lock"
    with _fl.FileLock(str(_lock_path), timeout=120):
        if _cache_path.exists():
            return pd.read_parquet(_cache_path)
        _jq_user = os.environ.get("JQ_USER", "")
        _jq_pass = os.environ.get("JQ_PASS", "")
        if not _jq_user or not _jq_pass:
            raise RuntimeError("JQ_USER/JQ_PASS 环境变量未设置，无法通过聚宽获取数据")
        import jqdatasdk as jq
        jq.auth(_jq_user, _jq_pass)
        try:
            if data_type == 'price':
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TErr
                _tp = _TPE(max_workers=1)
                _tf = _tp.submit(jq.get_price, symbol, start_date=start_date, end_date=end_date, frequency='daily', skip_paused=False, fq='pre')
                try:
                    df = _tf.result(timeout=180)
                except _TErr:
                    print(f"JQData get_price timeout (180s), symbol={symbol}", flush=True)
                    df = pd.DataFrame()
                finally:
                    _tp.shutdown(wait=False)
            elif data_type == 'index_components':
                stocks = jq.get_index_stocks(symbol)
                df = pd.DataFrame({'stock': stocks})
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


# ── 主进程加载数据，线程共享 _WDATA（线程池无fork问题）──
_WDATA = None

def _worker_chunk_batch(args):
    \"\"\"线程worker：通过共享 _WDATA 访问数据，用 xs() 快速提取单股票数据\"\"\"
    global _WDATA
    chunk_dates, stocks = args
    results = []
    for stock in stocks:
        try:
            sub = _WDATA.xs(stock, level='instrument')
        except KeyError:
            continue
        sub.index = pd.DatetimeIndex(sub.index.values)
        if sub.empty:
            continue
        r = calc_factors_one_day(sub, stock)
        if r is not None and not r.empty:
            # 统一索引类型为date（兼容用户返回 DatetimeIndex 或 date Index）
            if hasattr(r.index, 'date'):
                r.index = pd.Index(r.index.date)
            fname = r.name if r.name is not None else "factor"
            for d in chunk_dates:
                if d in r.index:
                    val = r.loc[d]
                    if not (isinstance(val, float) and np.isnan(val)):
                        results.append({
                            "datetime": d.strftime("%Y-%m-%d"),
                            "instrument": stock,
                            fname: float(val)
                        })
    return results
def _compute_day_sequential(td):
    \"\"\"顺序处理一天（Docker测试用）\"\"\"
    idx = TRADE_DATES.index(td)
    start_idx = max(0, idx - LOOKBACK_DAYS + 1)
    lookback_dates = TRADE_DATES[start_idx:idx + 1]
    day_all = pd.concat([load_day(d) for d in lookback_dates])
    results = []
    for stock in day_all.index.get_level_values("instrument").unique():
        day_df = day_all.xs(stock, level="instrument")
        if day_df.empty:
            continue
        day_df = day_df.copy()
        day_df.index = pd.DatetimeIndex(day_df.index.values)
        r = calc_factors_one_day(day_df, stock)
        if r is not None and not r.empty:
            if hasattr(r.index, 'date'):
                r.index = pd.Index(r.index.date)
            results.append({"datetime": str(td), "instrument": stock, **r})
    return results


if __name__ == '__main__':
    t0 = time.time()
    total_dates = len(TRADE_DATES)
    _CHK_DIR = Path("checkpoints")
    _CHK_DIR.mkdir(exist_ok=True)
    long_df = None

    # ── 自动列推断：分析用户函数，只加载需要的列 ──
    # 覆盖 {_LOAD_COLS_DEF} 中的 None（加载全部），避免分钟数据 OOM
    import re as _re, inspect as _inspect, pyarrow.parquet as _pq
    try:
        _SAMPLE_FILE = next(MINUTE_BY_DATE_DIR.glob("*.parquet"))
        _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {'datetime', 'instrument'}
        _USER_SOURCE = _inspect.getsource(calc_factors_one_day)
        # 提取代码中所有引号字符串，与可用列取交集
        _ALL_QUOTED = set(_re.findall(r'''['\"](\w+)['\"]''', _USER_SOURCE))
        _DETECTED = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
        if _DETECTED:
            _LOAD_COLS = _DETECTED
        print(f"检测到因子使用的列: {_LOAD_COLS}", flush=True)
    except Exception:
        print(f"列自动检测失败，使用默认: {_LOAD_COLS}", flush=True)

    # ── 内存自适应stock group ──
    # 根据列数和lookback自动拆分stock group，确保 _WDATA 不超1.5GB峰值
    _N_COLS_LOADED = len(_LOAD_COLS) if (_LOAD_COLS is not None and len(_LOAD_COLS) > 0) else 10
    _MAX_WDATA_BYTES = 1.5e9
    _BYTES_PER_STOCK_DAY = 330 * 8 * 1.5 * _N_COLS_LOADED
    _DAYS_PER_LOAD = max(LOOKBACK_DAYS + 24, 1)  # LOOKBACK_DAYS + CHUNK_SIZE - 1 (CHUNK_SIZE≈25)
    _MAX_STOCKS_PER_LOAD = max(1, int(_MAX_WDATA_BYTES / max(_BYTES_PER_STOCK_DAY * _DAYS_PER_LOAD, 1)))
    if _MAX_STOCKS_PER_LOAD < len(STOCK_LIST):
        _N_GROUPS = (len(STOCK_LIST) + _MAX_STOCKS_PER_LOAD - 1) // _MAX_STOCKS_PER_LOAD
        # 细粒度控制：环境变量 FACTOR_N_GROUPS 覆盖自动计算值
        _N_GROUPS = max(1, int(os.environ.get("FACTOR_N_GROUPS", str(_N_GROUPS))))
        _group_size = (len(STOCK_LIST) + _N_GROUPS - 1) // _N_GROUPS
        _stock_groups = [STOCK_LIST[i:i+_group_size] for i in range(0, len(STOCK_LIST), _group_size)]
        _stock_group_sets = [set(g) for g in _stock_groups]
        print(f"  内存保护: {_N_GROUPS} stock groups ({_group_size}只/组, "
              f"~{_DAYS_PER_LOAD}天/load, {_N_COLS_LOADED}列)", flush=True)
    else:
        _stock_groups = [STOCK_LIST]
        _stock_group_sets = [set(STOCK_LIST)]
        _N_GROUPS = 1

    try:
        if N_WORKERS <= 1:
            # ── Docker测试模式：顺序逐天 ──
            print("顺序处理（Docker测试模式）...", flush=True)
            all_records = []
            for i, td in enumerate(TRADE_DATES):
                day_records = _compute_day_sequential(td)
                all_records.extend(day_records)
                if (i + 1) % 200 == 0 or i == total_dates - 1:
                    rss = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
                    print(f"  进度: {i+1}/{total_dates} 天, {len(all_records)} 条, "
                          f"{time.time()-t0:.0f}s, RSS={rss}MB", flush=True)
            if all_records:
                long_df = pd.DataFrame(all_records)
                del all_records
        else:
            # ── 全量模式：Chunk + Stock Group 批处理（环境变量FACTOR_CHUNK_SIZE覆盖）──
            CHUNK_SIZE = int(os.environ.get("FACTOR_CHUNK_SIZE", "25"))
            n_chunks = (total_dates + CHUNK_SIZE - 1) // CHUNK_SIZE
            _expected_chks = n_chunks * _N_GROUPS
            print(f"Chunk模式: {N_WORKERS} workers, {_N_GROUPS} group(s), "
                  f"{total_dates} 天, {CHUNK_SIZE}天/chunk={n_chunks} chunks, "
                  f"{_expected_chks} checkpoints", flush=True)

            # 预转换日期字符串为date对象（避免每个worker重复转换）
            import datetime as _dt
            _TDS_OBJ = [_dt.datetime.strptime(d, "%Y-%m-%d").date() for d in TRADE_DATES]

            # ── checkpoint recovery：检测已有checkpoint，从中断处继续 ──
            # checkpoint 命名: chk_{chunk:04d}_g{group:02d}.parquet
            _existing_chks = sorted(_CHK_DIR.glob("chk_*.parquet"))
            if _existing_chks:
                _max_chk = 0
                for _f in _existing_chks:
                    _parts = _f.stem.split("_")
                    # chk_NNNN_gNN
                    if len(_parts) >= 2:
                        _c = int(_parts[1])
                        if _c > _max_chk:
                            _max_chk = _c
                _resume_from = _max_chk * CHUNK_SIZE
                if _resume_from >= total_dates:
                    print(f"  所有 checkpoint 已完成，跳过计算", flush=True)
                    _resume_from = total_dates
                else:
                    _existing_chk_names = {f.name for f in _existing_chks}
                    print(f"  检测到 {len(_existing_chks)} 个已有checkpoint，"
                          f"从chunk {_max_chk + 1}继续", flush=True)
            else:
                _resume_from = 0
                _existing_chk_names = set()

            for chunk_start in range(_resume_from, total_dates, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, total_dates)
                chunk_dates = _TDS_OBJ[chunk_start:chunk_end]
                data_start = max(0, chunk_start - LOOKBACK_DAYS + 1)
                chunk_num = chunk_start // CHUNK_SIZE + 1

                for _g_idx, (_g_stocks, _g_stock_set) in enumerate(zip(_stock_groups, _stock_group_sets)):
                    _cp_name = f"chk_{chunk_num:04d}_g{_g_idx:02d}.parquet"
                    if _cp_name in _existing_chk_names:
                        continue

                    _g_t0 = time.time()

                    # 加载数据：并行读 parquet，按天过滤到当前stock group
                    _day_parts = []
                    _load_range = list(range(data_start, chunk_end))
                    if len(_load_range) > 4:
                        # 4线程并行 I/O
                        with ThreadPoolExecutor(max_workers=min(4, len(_load_range))) as _io_pool:
                            _fut_map = {_io_pool.submit(load_day, TRADE_DATES[_d]): _d for _d in _load_range}
                            for _fut in as_completed(_fut_map):
                                _d = _fut_map[_fut]
                                _day_df = _fut.result()
                                if _N_GROUPS > 1:
                                    _day_df = _day_df.loc[_day_df.index.isin(_g_stock_set, level='instrument')]
                                _day_parts.append(_day_df)
                    else:
                        for _d in _load_range:
                            _day_df = load_day(TRADE_DATES[_d])
                            if _N_GROUPS > 1:
                                _day_df = _day_df.loc[_day_df.index.isin(_g_stock_set, level='instrument')]
                            _day_parts.append(_day_df)
                    _WDATA = pd.concat(_day_parts).sort_index()
                    del _day_parts

                    # 当前组的线程chunk划分
                    _stock_chunks = [_g_stocks[i::N_WORKERS] for i in range(N_WORKERS)]

                    # 线程池：共享 _WDATA，无fork问题
                    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                        futures = []
                        for w_stocks in _stock_chunks:
                            futures.append(pool.submit(
                                _worker_chunk_batch, (chunk_dates, w_stocks)
                            ))
                        chunk_records = []
                        for fut in as_completed(futures):
                            for rec in fut.result():
                                chunk_records.append(rec)

                    # 释放 _WDATA
                    _WDATA = None
                    gc.collect()
                    try:
                        import ctypes
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

                    # 保存checkpoint（空chunk也保存空文件以保持计数完整）
                    _cp = _CHK_DIR / _cp_name
                    if chunk_records:
                        pd.DataFrame(chunk_records).to_parquet(_cp)
                    else:
                        # 空chunk写一个仅含列的DataFrame，保持checkpoint计数一致
                        pd.DataFrame({"datetime": pd.Series(dtype=str),
                                      "instrument": pd.Series(dtype=str)}).to_parquet(_cp)

                    _g_t = time.time() - _g_t0
                    rss = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
                    print(f"  Chk {chunk_num}/{n_chunks} g{_g_idx+1}/{_N_GROUPS}: "
                          f"{TRADE_DATES[chunk_start]}~{TRADE_DATES[chunk_end-1]} "
                          f"({len(chunk_dates)}天) {len(chunk_records)}条 "
                          f"col={_g_t:.1f}s RSS={rss}MB", flush=True)

                    del chunk_records
                    gc.collect()
                    try:
                        import ctypes
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

        # ── 合并所有 checkpoint（允许空 chunk）──
        _chk_files = sorted(_CHK_DIR.glob("chk_*.parquet"))
        if _chk_files:
            _parts = [pd.read_parquet(f) for f in _chk_files]
            _parts = [p for p in _parts if not p.empty]
            if long_df is not None and not long_df.empty:
                _parts.append(long_df)
            if _parts:
                long_df = pd.concat(_parts, ignore_index=True)
            del _parts
            for _f in _chk_files:
                _f.unlink(missing_ok=True)
        elif long_df is None or long_df.empty:
            print("警告：没有产生任何因子值！", flush=True)

        if long_df is not None and not long_df.empty:
            long_df["datetime"] = pd.to_datetime(long_df["datetime"])
            factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
            wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
            wide = wide.sort_index().sort_index(axis=1)
            wide.index.name = "trade_date"
            wide.columns.name = "stock_code"
            wide = wide.replace([np.inf, -np.inf], np.nan)
            wide.attrs["factor_name"] = factor_name
            # 涨停剔除(minute)
            _LU_PATH = DATA_DIR / "limit_up_daily.parquet"
            if _LU_PATH.exists():
                _lu_df = pd.read_parquet(_LU_PATH, columns=['datetime', 'instrument'])
                for _lu_dt, _lu_grp in _lu_df.groupby(_lu_df['datetime'].dt.normalize()):
                    if _lu_dt in wide.index:
                        _c = [str(x) for x in _lu_grp['instrument'] if str(x) in wide.columns]
                        if _c:
                            wide.loc[_lu_dt, _c] = np.nan
            # /涨停剔除
            # 统一格式：index→string日期, columns→int股票代码
            wide.index = wide.index.strftime('%Y-%m-%d')
            wide.columns = wide.columns.astype(int)
            wide.to_parquet("result.parquet")
            print(f"完成！{wide.shape[0]} 天 x {wide.shape[1]} 只股票, "
                  f"{time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        os._exit(1)
    finally:
        os._exit(0)
"""

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    # 截面因子框架代码模板（loky 并行，每个 worker 独立加载数据）
    CROSS_SECTION_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))
LOOKBACK_DAYS = {lookback_days}  # 由框架注入，0=不切片
N_WORKERS = int(os.environ.get("FACTOR_N_WORKERS", "4"))

def load_stock(stock, columns=None):
    import pyarrow.parquet as pq
    path = STOCK_DATA_DIR / f"{stock}.parquet"
    if columns:
        table = pq.read_table(path, columns=columns, memory_map=True)
    else:
        table = pq.read_table(path, memory_map=True)
    return table.to_pandas()

_INDUSTRY_FILE = STOCK_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    \"\"\"通用聚宽数据获取函数。优先读本地缓存，没有再通过聚宽在线下载。
    本地数据中已有的字段（如日频价量、基本面等）直接走本地，不会调用聚宽。
    用法:
      idx = get_jq_data('000300.XSHG', 'price')  # 指数行情
      stocks = get_jq_data('000905.XSHG', 'index_components')  # 中证500成分股列表
    data_type 支持: 'price'(行情), 'index_components'(指数成分股)
    \"\"\"
    import hashlib as _hashlib
    _cache_key = f"jq_{data_type}_{_hashlib.md5(symbol.encode()).hexdigest()[:8]}"
    _cache_path = STOCK_DATA_DIR / f"{_cache_key}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
    import filelock as _fl
    _lock_path = STOCK_DATA_DIR / f"{_cache_key}.parquet.lock"
    with _fl.FileLock(str(_lock_path), timeout=120):
        if _cache_path.exists():
            return pd.read_parquet(_cache_path)
        _jq_user = os.environ.get("JQ_USER", "")
        _jq_pass = os.environ.get("JQ_PASS", "")
        if not _jq_user or not _jq_pass:
            raise RuntimeError("JQ_USER/JQ_PASS 环境变量未设置，无法通过聚宽获取数据")
        import jqdatasdk as jq
        jq.auth(_jq_user, _jq_pass)
        try:
            if data_type == 'price':
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TErr
                _tp = _TPE(max_workers=1)
                _tf = _tp.submit(jq.get_price, symbol, start_date=start_date, end_date=end_date, frequency='daily', skip_paused=False, fq='pre')
                try:
                    df = _tf.result(timeout=180)
                except _TErr:
                    print(f"JQData get_price timeout (180s), symbol={symbol}", flush=True)
                    df = pd.DataFrame()
                finally:
                    _tp.shutdown(wait=False)
            elif data_type == 'index_components':
                stocks = jq.get_index_stocks(symbol)
                df = pd.DataFrame({'stock': stocks})
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

# ── 进程共享缓存（fork COW，子进程共享父进程预加载数据） ──
_WCACHE = {}
_WPOS = {}
_WVALID = None
_WTDIDX = None
_SD = None
_LOAD_COLS = None

def _init_shared():
    \"\"\"初始化全局共享缓存（主进程执行一次，fork COW 共享给子进程）\"\"\"
    global _WVALID, _WTDIDX, _SD, _LOAD_COLS
    _SD = STOCK_DATA_DIR
    _WVALID = STOCK_LIST
    _WTDIDX = pd.DatetimeIndex(TRADE_DATES)
    print(f"  [主进程] 共享缓存就绪，{len(STOCK_LIST)}只股票按需加载", flush=True)

def _get_stock(s):
    \"\"\"延迟加载 — 全量位置预计算，跨chunk复用（fork前预填充，子进程COW读取）\"\"\"
    global _WCACHE, _WPOS
    if s not in _WCACHE:
        try:
            import pyarrow.parquet as pq
            _t = pq.read_table(_SD / f"{s}.parquet", columns=_LOAD_COLS, memory_map=True)
            df = _t.to_pandas()
            # _LOAD_COLS非None时pyarrow按列读取会丢失datetime索引
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            # 确保索引有序
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            pos = np.searchsorted(df.index.values.astype('int64'), _WTDIDX.values.astype('int64'), side='right')
            # 先写_WPOS再写_WCACHE：防止另一个线程读到_WCACHE有值但_WPOS还没有
            _WPOS[s] = pos
            _WCACHE[s] = df
        except Exception:
            _WCACHE[s] = None
            _WPOS[s] = None
    return _WCACHE[s]

def _worker_days(day_indices):
    \"\"\"进程：处理一组日期的截面因子（全量_WPOS跨chunk复用）\"\"\"
    global _WCACHE, _WVALID, _WTDIDX
    lb = LOOKBACK_DAYS
    results = []
    for i in day_indices:
        with np.errstate(invalid='ignore'):
            td = _WTDIDX[i]
            td_str = str(td.date())
            ad = {}
            for s in _WVALID:
                df = _get_stock(s)
                if df is None:
                    continue
                p = _WPOS[s][i]
                if p == 0:
                    continue
                st = max(0, p - lb) if lb > 0 else 0
                ad[s] = df.iloc[st:p]
            if not ad:
                continue
            try:
                r = calc_factor_cross_section(ad, td)
            except Exception:
                r = {}
            for s, fd in r.items():
                if fd and not any(v is None or (isinstance(v, float) and np.isnan(v)) for v in fd.values()):
                    results.append({"datetime": td_str, "instrument": s, **fd})
    return results
if __name__ == '__main__':
    try:
        # ---- Auto-detect needed columns from user code ----
        import re, inspect, pyarrow.parquet as pq
        _SAMPLE_FILE = next(STOCK_DATA_DIR.glob("*.parquet"))
        _AVAILABLE_COLS = set(pq.read_schema(_SAMPLE_FILE).names) - {'instrument'}
        _USER_SOURCE = inspect.getsource(calc_factor_cross_section)
        # 提取代码中所有引号字符串，与可用列取交集
        _ALL_QUOTED = set(re.findall(r'''['\"](\w+)['\"]''', _USER_SOURCE))
        _LOAD_COLS = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
        # 确保datetime列总是被加载（parquet按列读取时会丢失索引列）
        if _LOAD_COLS is not None and 'datetime' not in _LOAD_COLS:
            _LOAD_COLS = ['datetime'] + _LOAD_COLS
        if not _LOAD_COLS:
            _LOAD_COLS = None
        print(f"检测到因子使用的列: {_LOAD_COLS}", flush=True)
        # ----

        _CHK_DIR = Path("checkpoints")
        _CHK_DIR.mkdir(exist_ok=True)
        _CHUNK = 200
        _t0_main = time.time()

        print(f"截面计算: {len(TRADE_DATES)} 天, chunk={_CHUNK} 天, processes={N_WORKERS}", flush=True)
        print(f"进程共享缓存(fork COW): 5435只股票fork前预加载，子进程共享", flush=True)

        # 初始化全局共享缓存
        _init_shared()

        # 预加载所有股票数据到 _WCACHE，确保 fork 子进程通过 COW 共享
        for _s in STOCK_LIST:
            _get_stock(_s)
        print(f"  [主进程] {len(_WCACHE)}只股票预加载完成", flush=True)

        _ranges = list(range(0, len(TRADE_DATES), _CHUNK))

        for _ci, _cs in enumerate(_ranges):
            _ce = min(_cs + _CHUNK, len(TRADE_DATES))
            _t_chk = time.time()

            # ProcessPoolExecutor(fork)：子进程通过 COW 共享预加载数据
            with ProcessPoolExecutor(max_workers=N_WORKERS) as _pool:
                # 将chunk内日期分成N_WORKERS组
                _day_indices = list(range(_cs, _ce))
                _splits = np.array_split(_day_indices, min(N_WORKERS, len(_day_indices)))
                _futures = {_pool.submit(_worker_days, list(split)): split for split in _splits}

                _all_recs = []
                for _f in as_completed(_futures):
                    _recs = _f.result()
                    if _recs:
                        _all_recs.extend(_recs)

            _sd = TRADE_DATES[_cs]
            _ed = TRADE_DATES[min(_ce - 1, len(TRADE_DATES) - 1)]
            if _all_recs:
                pd.DataFrame(_all_recs).to_parquet(_CHK_DIR / f"chk_{_ci:04d}.parquet")
                print(f"  chunk {_ci + 1}/{len(_ranges)} [{_sd} ~ {_ed}]: "
                      f"valid={len(_all_recs)} recs, {time.time()-_t_chk:.0f}s", flush=True)
            else:
                print(f"  chunk {_ci + 1}/{len(_ranges)} [{_sd} ~ {_ed}]: "
                      f"no valid records, {time.time()-_t_chk:.0f}s", flush=True)

        _chk_files = sorted(_CHK_DIR.glob("chk_*.parquet"))
        if not _chk_files:
            print("无有效数据，退出")
            sys.exit(0)

        long_df = pd.concat([pd.read_parquet(f) for f in _chk_files], ignore_index=True)
        for f in _chk_files:
            f.unlink()
        _CHK_DIR.rmdir()

        long_df["datetime"] = pd.to_datetime(long_df["datetime"])
        factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
        wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "trade_date"
        wide.columns.name = "stock_code"
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide.attrs["factor_name"] = factor_name
        # 涨停剔除(cross_section)
        _LU_PATH = DATA_DIR / "limit_up_daily.parquet"
        if _LU_PATH.exists():
            _lu_df = pd.read_parquet(_LU_PATH, columns=['datetime', 'instrument'])
            for _lu_dt, _lu_grp in _lu_df.groupby(_lu_df['datetime'].dt.normalize()):
                if _lu_dt in wide.index:
                    _c = [str(x) for x in _lu_grp['instrument'] if str(x) in wide.columns]
                    if _c:
                        wide.loc[_lu_dt, _c] = np.nan
        # /涨停剔除
        # 统一格式：index→string日期, columns→int股票代码
        wide.index = wide.index.strftime('%Y-%m-%d')
        wide.columns = wide.columns.astype(int)
        wide.to_parquet("result.parquet")
        nn = int(wide.notna().sum().sum())
        print(f"完成: {wide.shape[0]}天 x {wide.shape[1]}只, 非空={nn}/{wide.size}={nn/wide.size*100:.1f}%, "
              f"{time.time()-_t0_main:.0f}s", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        pass"""

    # 分钟线截面因子框架代码模板（按天并行，minute_by_date 格式，MultiIndex(instrument, datetime)）
    # 用户需实现两个函数：
    #   calc_factor_minute_raw(df, stock) → dict {"因子名": 值}  （单只股票分钟数据 → 原始值）
    #   cross_section_transform(all_values) → dict {stock: 值 或 {"因子名": 值}}  （全市场截面处理）
    MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path
from joblib import Parallel, delayed
import gc as _gc

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
MINUTE_BY_DATE_DIR = DATA_DIR / "stock_data" / "minute_by_date"
STOCK_LIST = json.load(open(MINUTE_BY_DATE_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(MINUTE_BY_DATE_DIR / "trade_dates.json"))
LOOKBACK_DAYS = max(1, {lookback_days})  # 分钟线至少1天

# 并行控制：threading 后端共享进程内存，默认4线程，环境变量FACTOR_N_WORKERS覆盖
N_WORKERS = int(os.environ.get("FACTOR_N_WORKERS", "4"))

def load_day(td):
    return pd.read_parquet(MINUTE_BY_DATE_DIR / f"{{td}}.parquet", columns=_LOAD_COLS)

# 行业分类数据（申万一级行业）：INDUSTRY_DICT[股票代码] = 行业名
_DAILY_DATA_DIR = DATA_DIR / "stock_data" / "daily"
_INDUSTRY_FILE = _DAILY_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {{}}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    \"\"\"通用聚宽数据获取函数。优先读本地缓存，没有再通过聚宽在线下载。
    本地数据中已有的字段（如日频价量、基本面等）直接走本地，不会调用聚宽。
    用法:
      idx = get_jq_data('000300.XSHG', 'price')  # 指数行情
      stocks = get_jq_data('000905.XSHG', 'index_components')  # 中证500成分股列表
    data_type 支持: 'price'(行情), 'index_components'(指数成分股)
    \"\"\"
    import hashlib as _hashlib
    _cache_key = f"jq_{{data_type}}_{{_hashlib.md5(symbol.encode()).hexdigest()[:8]}}"
    _cache_path = _DAILY_DATA_DIR / f"{{_cache_key}}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
    # 文件锁防止并发 JQData 连接数超限（账号最多3个连接）
    import filelock as _fl
    _lock_path = _DAILY_DATA_DIR / f"{{_cache_key}}.parquet.lock"
    with _fl.FileLock(str(_lock_path), timeout=120):
        if _cache_path.exists():
            return pd.read_parquet(_cache_path)
        _jq_user = os.environ.get("JQ_USER", "")
        _jq_pass = os.environ.get("JQ_PASS", "")
        if not _jq_user or not _jq_pass:
            raise RuntimeError("JQ_USER/JQ_PASS 环境变量未设置，无法通过聚宽获取数据")
        import jqdatasdk as jq
        jq.auth(_jq_user, _jq_pass)
        try:
            if data_type == 'price':
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TErr
                _tp = _TPE(max_workers=1)
                _tf = _tp.submit(jq.get_price, symbol, start_date=start_date, end_date=end_date, frequency='daily', skip_paused=False, fq='pre')
                try:
                    df = _tf.result(timeout=180)
                except _TErr:
                    print(f"JQData get_price timeout (180s), symbol={symbol}", flush=True)
                    df = pd.DataFrame()
                finally:
                    _tp.shutdown(wait=False)
            elif data_type == 'index_components':
                stocks = jq.get_index_stocks(symbol)
                df = pd.DataFrame({{'stock': stocks}})
            else:
                raise ValueError(f"unsupported data_type: {{data_type}}")
            if df is not None and not df.empty:
                try:
                    df.to_parquet(_cache_path)
                except OSError:
                    pass
            return df
        finally:
            jq.logout()

{user_code}

# ── 自动列推断：分析用户函数，只加载需要的列 ──
import re as _re, inspect as _inspect, pyarrow.parquet as _pq
_LOAD_COLS = None
try:
    _SAMPLE_FILE = next(MINUTE_BY_DATE_DIR.glob("*.parquet"))
    _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {'datetime', 'instrument'}
    _USER_SOURCE = ""
    try:
        _USER_SOURCE += _inspect.getsource(calc_factor_minute_raw)
    except Exception:
        pass
    try:
        _USER_SOURCE += "\\n" + _inspect.getsource(cross_section_transform)
    except Exception:
        pass
    # 提取代码中所有引号字符串，与可用列取交集
    _ALL_QUOTED = set(_re.findall(r'''['\"](\w+)['\"]''', _USER_SOURCE))
    _LOAD_COLS = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
    if not _LOAD_COLS:
        _LOAD_COLS = None
except Exception:
    pass
print(f"检测到因子使用的列: {_LOAD_COLS}", flush=True)
# ──

# ── 内存自适应 stock group ──
_N_COLS_LOADED = len(_LOAD_COLS) if (_LOAD_COLS is not None and len(_LOAD_COLS) > 0) else 10
_MAX_WDATA_BYTES = 1.5e9
_BYTES_PER_STOCK_DAY = 330 * 8 * 1.5 * _N_COLS_LOADED
_DAYS_PER_LOAD = max(LOOKBACK_DAYS, 1)
_MAX_STOCKS_PER_LOAD = max(1, int(_MAX_WDATA_BYTES / max(_BYTES_PER_STOCK_DAY * _DAYS_PER_LOAD, 1)))
if _MAX_STOCKS_PER_LOAD < len(STOCK_LIST):
    _N_GROUPS = (len(STOCK_LIST) + _MAX_STOCKS_PER_LOAD - 1) // _MAX_STOCKS_PER_LOAD
    _N_GROUPS = max(1, int(os.environ.get("FACTOR_N_GROUPS", str(_N_GROUPS))))
    _group_size = (len(STOCK_LIST) + _N_GROUPS - 1) // _N_GROUPS
    _stock_subset_groups = [STOCK_LIST[i:i+_group_size] for i in range(0, len(STOCK_LIST), _group_size)]
    _stock_subset_sets = [set(g) for g in _stock_subset_groups]
    print(f"  内存保护: {_N_GROUPS} stock groups ({_group_size}只/组, "
          f"{_DAYS_PER_LOAD}天/load, {_N_COLS_LOADED}列)", flush=True)
else:
    _stock_subset_groups = [None]  # None means load all stocks
    _stock_subset_sets = [None]
    _N_GROUPS = 1

def _compute_day_raw(td, stock_subset=None):
    \"\"\"加载一天（+lookback）的分钟数据，计算指定股票的原始值\"\"\"
    if LOOKBACK_DAYS > 1:
        idx = TRADE_DATES.index(td)
        start_idx = max(0, idx - LOOKBACK_DAYS + 1)
        lookback_dates = TRADE_DATES[start_idx:idx + 1]
        if stock_subset is not None:
            _stock_set = stock_subset
            _parts = []
            for _d in lookback_dates:
                _df = load_day(_d)
                _df = _df[_df.index.get_level_values('instrument').isin(_stock_set)]
                _parts.append(_df)
            day_all = pd.concat(_parts).sort_index()
            del _parts
        else:
            day_all = pd.concat([load_day(d) for d in lookback_dates]).sort_index()
    else:
        day_all = load_day(td)
        if stock_subset is not None:
            day_all = day_all[day_all.index.get_level_values('instrument').isin(stock_subset)]
    results = {{}}
    for stock, sub in day_all.groupby(level="instrument"):
        sub = sub.droplevel("instrument")
        if sub.empty:
            continue
        r = calc_factor_minute_raw(sub, stock)
        if r:
            results[str(stock)] = r
    del day_all
    _gc.collect()
    return str(td), results

def _process_chunk(chunk, fname):
    \"\"\"每个 worker 独立处理一段日期的截面变换\"\"\"
    records = []
    for td, stock_raw_dict in chunk:
        all_values = {{s: d.get(fname, np.nan) for s, d in stock_raw_dict.items()}}
        transformed = cross_section_transform(all_values)
        for stock, val in transformed.items():
            if isinstance(val, dict):
                records.append({{"datetime": td, "instrument": stock, **val}})
            else:
                records.append({{"datetime": td, "instrument": stock, fname: val}})
    return records

def _merge_group_results(group_results_list, factor_name):
    \"\"\"合并多个stock group的结果为单个 DataFrame\"\"\"
    _parts = []
    for _gr in group_results_list:
        if _gr is not None and not _gr.empty:
            _parts.append(_gr)
    if _parts:
        return pd.concat(_parts, ignore_index=True)
    return pd.DataFrame()

if __name__ == '__main__':
    try:
        _t0_main = time.time()

        # 按 stock group 处理，每个 group 独立跑完整流程
        _all_group_results = []
        for _g_idx in range(_N_GROUPS):
            _g_stock_subset = _stock_subset_groups[_g_idx]
            _g_t0 = time.time()
            _g_tag = f"[g{_g_idx+1}/{_N_GROUPS}]" if _N_GROUPS > 1 else ""
            print(f"{_g_tag} 开始处理 stock group...", flush=True)

            # 第一步：按天并行计算原始值
            raw_day_results = {{}}
            _progress_interval = max(1, len(TRADE_DATES) // 20)
            for _day_idx, (day_td, day_values) in enumerate(
                Parallel(n_jobs=N_WORKERS, backend="threading")(
                    delayed(_compute_day_raw)(td, _g_stock_subset) for td in TRADE_DATES
                ),
                1
            ):
                if day_values:
                    raw_day_results[day_td] = day_values
                if _day_idx % _progress_interval == 0 or _day_idx == len(TRADE_DATES):
                    print(f"  {_g_tag} 第一步进度: {{_day_idx}}/{{len(TRADE_DATES)}} 天完成", flush=True)

            if not raw_day_results:
                print(f"{_g_tag} 无有效数据，跳过", flush=True)
                _all_group_results.append(pd.DataFrame())
                continue

            # 确定因子名（取第一个非空结果的首个 key）
            _first_day = next(iter(raw_day_results.values()))
            _first_stock = next(iter(_first_day))
            factor_name = list(_first_day[_first_stock].keys())[0]

            # 第二步：按天做截面处理（并行）
            all_dates = list(raw_day_results.items())
            N_CS_JOBS = min(N_WORKERS, len(all_dates))
            def _chunkify(lst, n):
                k, m = divmod(len(lst), n)
                return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]
            td_chunks = _chunkify(all_dates, N_CS_JOBS)
            all_records = []
            print(f"  {_g_tag} 第二步截面变换 ({{N_CS_JOBS}} workers)...", flush=True)
            for _chunk_idx, chunk_records in enumerate(
                Parallel(n_jobs=N_CS_JOBS, backend="threading")(
                    delayed(_process_chunk)(chunk, factor_name) for chunk in td_chunks
                ),
                1
            ):
                all_records.extend(chunk_records)
                print(f"  {_g_tag} 第二步进度: {{_chunk_idx}}/{{N_CS_JOBS}} chunks 完成", flush=True)

            long_df = pd.DataFrame(all_records)
            long_df["datetime"] = pd.to_datetime(long_df["datetime"])
            _g_wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
            _g_wide = _g_wide.sort_index().sort_index(axis=1)
            _g_wide = _g_wide.replace([np.inf, -np.inf], np.nan)
            _g_wide.attrs["factor_name"] = factor_name
            _all_group_results.append(_g_wide)
            _g_elapsed = time.time() - _g_t0
            print(f"  {_g_tag} group完成: {{_g_wide.shape[0]}}天 x {{_g_wide.shape[1]}}只, "
                  f"{{_g_elapsed:.0f}}s", flush=True)

        # 合并所有 group 结果
        _result_parts = [g for g in _all_group_results if g is not None and not g.empty]
        if not _result_parts:
            print("无有效数据，退出")
            sys.exit(0)
        wide = pd.concat(_result_parts, axis=1)
        # 去重列（如果 group 间有重叠）
        wide = wide.loc[:, ~wide.columns.duplicated(keep='first')]
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "trade_date"
        wide.columns.name = "stock_code"
        # 涨停剔除(minute_cs)
        _LU_PATH = DATA_DIR / "limit_up_daily.parquet"
        if _LU_PATH.exists():
            _lu_df = pd.read_parquet(_LU_PATH, columns=['datetime', 'instrument'])
            wide = wide.replace([np.inf, -np.inf], np.nan)
            for _lu_dt, _lu_grp in _lu_df.groupby(_lu_df['datetime'].dt.normalize()):
                if _lu_dt in wide.index:
                    _c = [str(x) for x in _lu_grp['instrument'] if str(x) in wide.columns]
                    if _c:
                        wide.loc[_lu_dt, _c] = np.nan
        else:
            wide = wide.replace([np.inf, -np.inf], np.nan)
        # /涨停剔除
        # 统一格式：index→string日期, columns→int股票代码
        wide.index = wide.index.strftime('%Y-%m-%d')
        wide.columns = wide.columns.astype(int)
        wide.to_parquet("result.parquet")
        print(f"完成！共 {{wide.shape[0]}} 天 x {{wide.shape[1]}} 只股票, "
              f"{{time.time()-_t0_main:.0f}}s", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
"""

    # 深度学习因子框架代码模板
    DEEP_LEARNING_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os
from pathlib import Path

# Fix Intel VTune JIT stubs: undefined symbol iJIT_NotifyEvent
import ctypes as _ctypes
# Locate libittnotify_stub.so in current conda/pip environment
_sys_prefix = getattr(sys, 'prefix', None) or os.path.dirname(sys.executable)
for _search_dir in [os.path.join(_sys_prefix, "lib"), os.path.join(os.path.dirname(sys.executable), "..", "lib")]:
    _stub = os.path.join(_search_dir, "libittnotify_stub.so")
    if os.path.exists(_stub):
        try:
            _ctypes.CDLL(_stub, mode=_ctypes.RTLD_GLOBAL)
            break
        except OSError:
            pass

import torch

DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or ".")
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))
LOOKBACK_DAYS = {lookback_days}  # 由框架注入，0=不切片

def load_stock(stock, columns=None):
    if columns:
        return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet", columns=columns)
    return pd.read_parquet(STOCK_DATA_DIR / f"{{stock}}.parquet")

# 行业分类数据（申万一级行业）：INDUSTRY_DICT[股票代码] = 行业名
_INDUSTRY_FILE = STOCK_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {{}}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    \"\"\"通用聚宽数据获取函数。优先读本地缓存，没有再通过聚宽在线下载。
    本地数据中已有的字段（如日频价量、基本面等）直接走本地，不会调用聚宽。
    用法:
      idx = get_jq_data('000300.XSHG', 'price')  # 指数行情
      stocks = get_jq_data('000905.XSHG', 'index_components')  # 中证500成分股列表
    data_type 支持: 'price'(行情), 'index_components'(指数成分股)
    \"\"\"
    import hashlib as _hashlib
    _cache_key = f"jq_{{data_type}}_{{_hashlib.md5(symbol.encode()).hexdigest()[:8]}}"
    _cache_path = STOCK_DATA_DIR / f"{{_cache_key}}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
    # 文件锁防止并发 JQData 连接数超限（账号最多3个连接）
    import filelock as _fl
    _lock_path = STOCK_DATA_DIR / f"{{_cache_key}}.parquet.lock"
    with _fl.FileLock(str(_lock_path), timeout=120):
        if _cache_path.exists():
            return pd.read_parquet(_cache_path)
        _jq_user = os.environ.get("JQ_USER", "")
        _jq_pass = os.environ.get("JQ_PASS", "")
        if not _jq_user or not _jq_pass:
            raise RuntimeError("JQ_USER/JQ_PASS 环境变量未设置，无法通过聚宽获取数据")
        import jqdatasdk as jq
        jq.auth(_jq_user, _jq_pass)
        try:
            if data_type == 'price':
                from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TErr
                _tp = _TPE(max_workers=1)
                _tf = _tp.submit(jq.get_price, symbol, start_date=start_date, end_date=end_date, frequency='daily', skip_paused=False, fq='pre')
                try:
                    df = _tf.result(timeout=180)
                except _TErr:
                    print(f"JQData get_price timeout (180s), symbol={{symbol}}", flush=True)
                    df = pd.DataFrame()
                finally:
                    _tp.shutdown(wait=False)
            elif data_type == 'index_components':
                stocks = jq.get_index_stocks(symbol)
                df = pd.DataFrame({{'stock': stocks}})
            else:
                raise ValueError(f"unsupported data_type: {{data_type}}")
            if df is not None and not df.empty:
                try:
                    df.to_parquet(_cache_path)
                except OSError:
                    pass
            return df
        finally:
            jq.logout()

{user_code}

if __name__ == '__main__':
    try:
        # ── 自动列推断：分析用户函数，只加载需要的列 ──
        import re as _re, inspect as _inspect, pyarrow.parquet as _pq
        _SAMPLE_FILE = next(STOCK_DATA_DIR.glob("*.parquet"))
        _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {'datetime', 'instrument'}
        try:
            _USER_SOURCE = _inspect.getsource(train_model) + "\\n" + _inspect.getsource(predict_batch)
        except (NameError, OSError):
            _USER_SOURCE = _inspect.getsource(train_model) + "\\n" + _inspect.getsource(predict)
        # 提取代码中所有引号字符串，与可用列取交集
        _ALL_QUOTED = set(_re.findall(r'''['"](\\w+)['"]''', _USER_SOURCE))
        _LOAD_COLS = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
        if not _LOAD_COLS:
            _LOAD_COLS = None
        print(f"检测到因子使用的列: {_LOAD_COLS}", flush=True)
        # ──

        print("计算深度学习因子...")
        # 预加载所有股票数据
        all_data = {{}}
        _total = len(STOCK_LIST)
        for _idx, stock in enumerate(STOCK_LIST):
            all_data[stock] = load_stock(stock, _LOAD_COLS)
            if (_idx + 1) % 1000 == 0 or (_idx + 1) == _total:
                print(f"  加载数据: {_idx+1}/{_total}", flush=True)

        _td_index = pd.DatetimeIndex(TRADE_DATES)
        # 预计算每只股票对所有日期的切片位置
        _stock_positions = {{}}
        for stock, df in all_data.items():
            _stock_positions[stock] = np.searchsorted(df.index.values.astype('int64'), _td_index.values.astype('int64'), side='right')

        all_records = []
        _has_predict_batch = 'predict_batch' in dir()

        # 按年份分组训练 + 推理，每年度只训练一次
        _year_groups = {{}}
        for i, td in enumerate(_td_index):
            _year_groups.setdefault(td.year, []).append(i)

        for _year, _date_idxs in sorted(_year_groups.items()):
            # 只用该年第一天之前的数据训练一次
            _first_td = _td_index[_date_idxs[0]]
            data_for_train = {{}}
            for stock, df in all_data.items():
                pos = _stock_positions[stock][_date_idxs[0]]
                if LOOKBACK_DAYS > 0:
                    if pos == 0:
                        continue
                    start = max(0, pos - LOOKBACK_DAYS - 1)
                    sub = df.iloc[start:pos]
                else:
                    sub = df.iloc[:pos]
                    if sub.empty:
                        continue
                data_for_train[stock] = sub
            if not data_for_train:
                continue
            model = train_model(data_for_train, _first_td)

            print(f"  按日期计算 [{_year}]: {len(_date_idxs)} 天", flush=True)
            for _batch_idx, i in enumerate(_date_idxs):
                td = _td_index[i]
                if (_batch_idx + 1) % 200 == 0 or (_batch_idx + 1) == len(_date_idxs):
                    print(f"    {_batch_idx+1}/{len(_date_idxs)}", flush=True)

                # 准备当日数据切片
                data_for_predict = {{}}
                for stock, df in all_data.items():
                    pos = _stock_positions[stock][i]
                    if LOOKBACK_DAYS > 0:
                        if pos == 0:
                            continue
                        start = max(0, pos - LOOKBACK_DAYS - 1)
                        sub = df.iloc[start:pos]
                    else:
                        sub = df.iloc[:pos]
                        if sub.empty:
                            continue
                    data_for_predict[stock] = sub
                if not data_for_predict:
                    continue

                if _has_predict_batch:
                    # GPU batch inference: returns (factor_name, {stock: value})
                    fname, results = predict_batch(model, data_for_predict, td)
                    for stock, val in results.items():
                        all_records.append({{"datetime": str(td.date()), "instrument": stock, fname: val}})
                else:
                    # Fallback: per-stock predict
                    for stock, df in data_for_predict.items():
                        r = predict(model, df, td, stock)
                        if r:
                            all_records.append({{"datetime": str(td.date()), "instrument": stock, **r}})

        long_df = pd.DataFrame(all_records)
        long_df["datetime"] = pd.to_datetime(long_df["datetime"])
        factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
        wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "trade_date"
        wide.columns.name = "stock_code"
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide.attrs["factor_name"] = factor_name
        # 涨停剔除(DL)
        _LU_PATH = DATA_DIR / "limit_up_daily.parquet"
        if _LU_PATH.exists():
            _lu_df = pd.read_parquet(_LU_PATH, columns=['datetime', 'instrument'])
            for _lu_dt, _lu_grp in _lu_df.groupby(_lu_df['datetime'].dt.normalize()):
                if _lu_dt in wide.index:
                    _c = [str(x) for x in _lu_grp['instrument'] if str(x) in wide.columns]
                    if _c:
                        wide.loc[_lu_dt, _c] = np.nan
        # /涨停剔除
        # 统一格式：index→string日期, columns→int股票代码
        wide.index = wide.index.strftime('%Y-%m-%d')
        wide.columns = wide.columns.astype(int)
        wide.to_parquet("result.parquet")
        print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}} 只股票", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        os._exit(0)"""

    # Target function names that constitute the "user code" portion.
    # Everything else (imports, DATA_DIR, load_stock, _compute_stock, __main__)
    # is framework boilerplate and should be stripped before re-wrapping.
    _TARGET_FUNC_NAMES = {
        "calc_factor_single_stock",
        "calc_factor_cross_section",
        "calc_factor_minute_raw",
        "cross_section_transform",
        "train_model",
        "predict",
        "predict_batch",
        "calc_factors_one_day",
    }

    @staticmethod
    def _extract_user_functions(code: str) -> str:
        """Strip framework boilerplate, keep only target function definitions."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        target_nodes = [
            node for node in ast.iter_child_nodes(tree)
            if isinstance(node, ast.FunctionDef) and node.name in FactorFBWorkspace._TARGET_FUNC_NAMES
        ]
        if not target_nodes:
            return code

        lines = code.splitlines()
        chunks = []
        for node in target_nodes:
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else len(lines)
            chunks.append("\n".join(lines[start:end]))
        return "\n\n".join(chunks).strip()

    _DEFAULT_COLS = ["open", "high", "low", "close", "volume", "return", "vwap", "datetime"]

    @staticmethod
    def _build_factor_code(template: str, code: str, lookback_days: int, load_cols: list = None) -> str:
        """安全填充模板：用 replace 替代 format，避免 {xxx} 被错误解释。
        load_cols: LLM推断的列名列表，None=使用默认常用列。
        """
        if load_cols:
            # 过滤掉索引列（datetime 是 MultiIndex 的一部分，不是数据列）
            filtered = [c for c in load_cols if c not in ("datetime", "instrument")]
            cols_def = f"_LOAD_COLS = {filtered}  # LLM推断"
        else:
            cols_def = "_LOAD_COLS = None  # 加载全部列"
        result = (template
                  .replace('{_LOAD_COLS_DEF}', cols_def)
                  .replace('{_LOAD_MINUTE_STOCK}', _LOAD_MINUTE_STOCK_SRC)
                  .replace('{lookback_days}', str(lookback_days))
                  .replace('{user_code}', code))
        # 只有含双花括号的旧模板需要解义（日线/截面等模板），分钟模板已用单花括号
        if '{{' in result:
            result = result.replace('{{', '{').replace('}}', '}')
        return result

    _INFER_COL_PROMPT = """Analyze the factor code below and list which stock data columns it reads from the DataFrame.

Available columns: {available}

Rules:
- Return ONLY a JSON array of column names without the $ prefix.
- Do NOT include "datetime" (it is always loaded as index).
- Include columns accessed via df[...], df., .agg(...), .assign(...) etc.
- If the code dynamically references columns (e.g. from a config dict), output all possible candidates.
- If unsure, return ["*"] to load all columns.

Factor code:
```python
{code}
```"""

    @staticmethod
    def _infer_columns_llm(code: str, is_minute: bool = False) -> list[str] | None:
        """用 LLM 分析因子代码，返回需要的列名列表（无 $ 前缀）。返回 None 表示全部加载。"""
        import re as _re_json
        if is_minute:
            available = "open, high, low, close, volume, vwap, return, factor"
        else:
            available = ("open, high, low, close, volume, factor, pct_chg, pre_close, turnover_rate, "
                       "roe, roa, pe_ttm, pb, revenue_yoy, profit_yoy, gross_margin, net_margin, "
                       "debt_to_asset, ocf_per_share, market_cap, circulating_market_cap, total_shares, "
                       "float_shares, adjusted_profit, gross_profit")
        user_prompt = FactorFBWorkspace._INFER_COL_PROMPT.replace("{available}", available).replace("{code}", code)
        try:
            response = APIBackend(use_chat_cache=True).build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt="You are a factor code analyzer. Return a JSON array of column names only.",
                json_mode=False,
            )
            # 从响应中提取 JSON 数组
            _m = _re_json.search(r'\[.*?\]', response, _re_json.DOTALL)
            if _m:
                cols = json.loads(_m.group(0))
            else:
                cols = json.loads(response)
            if not isinstance(cols, list) or len(cols) == 0:
                return None
            if cols == ["*"]:
                return None
            return sorted(set(c.strip() for c in cols if isinstance(c, str)))
        except Exception:
            return None

    def inject_files(self, *args, **kwargs):
        """Override to wrap AI-generated code with framework if needed."""
        # Call parent inject_files first
        super().inject_files(*args, **kwargs)
        # Check if factor.py needs framework wrapping
        if "factor.py" in self.file_dict:
            code = self.file_dict["factor.py"]
            # Strip existing boilerplate (from knowledge base replay or previous iteration)
            code = self._extract_user_functions(code)
            lookback = getattr(self.target_task, "lookback_days", 0) or 0
            # 分钟线默认 lookback 至少 1
            is_minute = "calc_factors_one_day" in code
            if is_minute and lookback <= 0:
                lookback = 1
            # LLM 推断需要的列（仅主流程调用时触发，不阻塞重跑缓存）
            load_cols = None
            if self.raise_exception:
                # raise_exception=True 表示是 LLM 生成阶段（非重跑）
                load_cols = self._infer_columns_llm(code, is_minute)
            # 按代码内容检测模板类型
            if "def train_model" in code and ("def predict" in code or "def predict_batch" in code):
                wrapped = self._build_factor_code(self.DEEP_LEARNING_FRAMEWORK_TEMPLATE, code, lookback, load_cols)
            elif "calc_factor_minute_raw" in code and "cross_section_transform" in code:
                wrapped = self._build_factor_code(self.MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE, code, lookback, load_cols)
            elif "calc_factor_cross_section" in code:
                wrapped = self._build_factor_code(self.CROSS_SECTION_FRAMEWORK_TEMPLATE, code, lookback, load_cols)
            elif is_minute:
                wrapped = self._build_factor_code(self.MINUTE_FRAMEWORK_TEMPLATE, code, lookback, load_cols)
            else:
                wrapped = self._build_factor_code(self.DAILY_FRAMEWORK_TEMPLATE, code, lookback, load_cols)
            self.file_dict["factor.py"] = wrapped
            (self.workspace_path / "factor.py").write_text(wrapped)

    def hash_func(self, data_type: str = "Debug") -> str:
        if "factor.py" not in self.file_dict or self.raise_exception:
            return None
        # Include data file mtimes so cache invalidates when data changes
        data_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug if data_type == "Debug" else FACTOR_COSTEER_SETTINGS.data_folder)
        data_sig = ""
        if data_folder.exists():
            for f in sorted(data_folder.iterdir()):
                data_sig += f"{f.name}:{f.stat().st_mtime_ns}"
        return md5_hash(str(data_folder.resolve()) + self.file_dict["factor.py"] + data_sig)

    @staticmethod
    def _sanitize_factor_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_") or "factor"

    def _resolve_export_dir(self, review_metadata: dict[str, Any] | None = None) -> Path:
        review_metadata = review_metadata or {}
        if review_metadata.get("source_type") == "literature_report":
            report_title = str(review_metadata.get("source_report_title") or "unknown_report")
            return self.EXPORTED_PARQUET_DIR / "literature_reports" / self._sanitize_factor_name(report_title)
        return self.EXPORTED_PARQUET_DIR

    def _clear_rejected_marker(self, factor_name: str, review_metadata: dict[str, Any] | None = None) -> None:
        review_metadata = review_metadata or {}
        if review_metadata.get("source_type") != "literature_report":
            return
        report_dir = self._resolve_export_dir(review_metadata)
        reason_path = report_dir / f"SKIPPED__{self._sanitize_factor_name(factor_name)}.md"
        if reason_path.exists():
            reason_path.unlink()

        summary_path = report_dir / "_SKIPPED_FACTORS.md"
        if not summary_path.exists():
            return
        lines = summary_path.read_text(encoding="utf-8").splitlines()
        kept_lines = [line for line in lines if not line.startswith(f"- `{factor_name}`：")]
        has_remaining_skip = any(line.startswith("- `") for line in kept_lines)
        if has_remaining_skip:
            summary_path.write_text("\n".join(kept_lines).rstrip() + "\n", encoding="utf-8")
        else:
            summary_path.unlink()

    @staticmethod
    def _hash_factor_dataframe(df: pd.DataFrame) -> str:
        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(hashed.tobytes()).hexdigest()

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _infer_time_granularity(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "unknown"
        # 处理宽表格式 (Date index) 和旧格式 (MultiIndex with datetime)
        if df.index.name == "Date":
            dt_index = pd.to_datetime(df.index)
        elif "datetime" in df.index.names:
            dt_index = pd.to_datetime(df.index.get_level_values("datetime"))
        else:
            return "unknown"
        diffs = dt_index.to_series().diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)].unique()
        if len(positive_diffs) == 0:
            return "unknown"
        min_step = min(positive_diffs)
        if min_step <= pd.Timedelta(minutes=1):
            return "minute"
        if min_step >= pd.Timedelta(days=1):
            return "daily"
        return str(min_step)

    @staticmethod
    def _infer_factor_tags(task: FactorTask | None, extra_tags: list[str] | None = None) -> list[str]:
        content = " ".join(
            [
                getattr(task, "factor_name", "") or "",
                getattr(task, "factor_description", "") or "",
                getattr(task, "factor_formulation", "") or "",
                str(getattr(task, "variables", {}) or {}),
            ]
        ).lower()
        tags: set[str] = set(extra_tags or [])
        keyword_to_tag = {
            "momentum": "momentum",
            "reversal": "reversal",
            "rev_": "reversal",
            "volatility": "volatility",
            "range": "range",
            "volume": "volume",
            "liquidity": "liquidity",
            "spread": "liquidity",
            "vwap": "vwap",
            "minute": "minute_input",
            "intraday": "minute_input",
            "microstructure": "microstructure",
            "gap": "gap",
            "price-volume": "price_volume",
            "correlation": "correlation",
            "acceleration": "acceleration",
            "trend": "trend",
        }
        for keyword, tag in keyword_to_tag.items():
            if keyword in content:
                tags.add(tag)
        if "minute_pv" in content or '/minute"' in content or "/minute'" in content:
            tags.add("minute_input")
        if "daily_pv" in content or '/daily"' in content or "/daily'" in content:
            tags.add("daily_input")
        return sorted(tags)

    @staticmethod
    def _compact_logic_summary(text: str | None, limit: int = 160) -> str | None:
        if text is None:
            return None
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _write_factor_metadata(
        self,
        factor_name: str,
        latest_path: Path,
        df: pd.DataFrame,
        factor_hash: str,
        review_metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_path = latest_path.with_suffix(".meta.json")
        code_path = latest_path.with_suffix(".code.py")
        task = self.target_task if isinstance(self.target_task, FactorTask) else None
        metadata = {
            "factor_name": factor_name,
            "display_name": factor_name,
            "factor_description": task.factor_description if task is not None else None,
            "factor_formulation": task.factor_formulation if task is not None else None,
            "variables": task.variables if task is not None else None,
            "hash": factor_hash,
            "rows": len(df),
            "non_null": int(df.stack().notna().sum()) if df.index.name == "Date" and df.columns.name == "Code" else int(df.iloc[:, 0].notna().sum()),
            "time_granularity": self._infer_time_granularity(df),
            "logic_summary": (
                task.factor_description if task is not None else "No factor description recorded."
            ),
            "tags": self._infer_factor_tags(task, extra_tags=(review_metadata or {}).get("tags")),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "workspace_path": str(self.workspace_path),
            "latest_path": str(latest_path),
            "metadata_path": str(metadata_path),
            "code_path": str(code_path),
        }
        if review_metadata:
            metadata.update(review_metadata)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_factor_code_snapshot(self, latest_path: Path) -> Path | None:
        code_path = latest_path.with_suffix(".code.py")
        code = self.file_dict.get("factor.py")
        if code is None:
            workspace_code_path = self.workspace_path / "factor.py"
            if workspace_code_path.exists():
                code = workspace_code_path.read_text(encoding="utf-8", errors="replace")
        if code is None:
            return None
        code_path.write_text(code, encoding="utf-8")
        return code_path

    def _export_factor_dataframe(self, df: pd.DataFrame, review_metadata: dict[str, Any] | None = None) -> None:
        if df is None or df.empty:
            return

        self.EXPORTED_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        export_dir = self._resolve_export_dir(review_metadata)
        # 从 attrs 获取因子名（宽表格式），或从列名获取（旧格式）
        if "factor_name" in df.attrs:
            factor_name = self._sanitize_factor_name(df.attrs["factor_name"])
        elif df.index.name == "Date" and df.columns.name == "Code":
            # 宽表但没有 attrs，尝试从任务获取
            factor_name = self._sanitize_factor_name(self.target_task.factor_name if self.target_task else "unknown")
        else:
            factor_name = self._sanitize_factor_name(str(df.columns[0]))
        # 文献因子使用 per-factor 子目录
        if review_metadata and review_metadata.get("source_type") == "literature_report":
            export_dir = export_dir / factor_name
        export_dir.mkdir(parents=True, exist_ok=True)
        latest_path = export_dir / f"{factor_name}.parquet"
        current_hash = self._hash_factor_dataframe(df)

        if latest_path.exists():
            try:
                existing_df = pd.read_parquet(latest_path)
                if self._hash_factor_dataframe(existing_df) == current_hash:
                    self._write_factor_code_snapshot(latest_path)
                    self._write_factor_metadata(factor_name, latest_path, df, current_hash, review_metadata)
                    self._clear_rejected_marker(factor_name, review_metadata)
                    return
            except Exception:
                # If the previous parquet cannot be read, overwrite it with the current successful output.
                pass

        # 统一 size：reindex 到完整的日期×股票，NaN 填充
        if df.index.name == "Date" and df.columns.name == "Code":
            full_dates = pd.read_json(
                Path(FACTOR_COSTEER_SETTINGS.data_folder) / "stock_data" / "daily" / "trade_dates.json",
                typ="series"
            )
            full_dates = pd.to_datetime(full_dates).sort_values()
            full_stocks = json.loads(
                (Path(FACTOR_COSTEER_SETTINGS.data_folder) / "stock_data" / "daily" / "stock_list.json").read_text()
            )
            full_stocks = sorted(full_stocks)
            df = df.reindex(index=full_dates, columns=full_stocks)
            df.index.name = "Date"
            df.columns.name = "Code"
            # 重新计算 hash
            current_hash = self._hash_factor_dataframe(df)

        df.to_parquet(latest_path, engine="pyarrow")
        self._write_factor_code_snapshot(latest_path)
        self._write_factor_metadata(factor_name, latest_path, df, current_hash, review_metadata)
        self._clear_rejected_marker(factor_name, review_metadata)

        if self._env_flag("FACTOR_EXPORT_KEEP_SNAPSHOTS"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = export_dir / f"{timestamp}__{factor_name}.parquet"
            df.to_parquet(snapshot_path, engine="pyarrow")

    def export_reviewed_factor(
        self,
        df: pd.DataFrame,
        *,
        accepted: bool,
        logic_summary: str | None = None,
        tags: list[str] | None = None,
        review_notes: str | None = None,
        **extra_review_metadata: Any,
    ) -> None:
        review_metadata = {
            "accepted": accepted,
            "logic_summary": logic_summary,
            "tags": tags or [],
            "review_notes": review_notes,
            "source_type": "agent_generated",
        }
        review_metadata.update(extra_review_metadata)
        self._export_factor_dataframe(df, review_metadata=review_metadata)

    @classmethod
    def _build_shared_data_launcher(cls, source_data_path: Path, code_path: Path) -> str:
        source_data_path = source_data_path.resolve() if source_data_path.is_absolute() else source_data_path
        code_path = code_path.resolve() if code_path.is_absolute() else code_path
        return textwrap.dedent(
            f"""
            import builtins
            import os
            import runpy
            from pathlib import Path

            import pandas as pd

            DATA_DIR = Path({str(source_data_path)!r})
            os.environ["FACTOR_DATA_DIR"] = str(DATA_DIR)
            os.environ["RDAGENT_FACTOR_DATA_DIR"] = str(DATA_DIR)

            def _resolve_data_path(path_like):
                if path_like is None:
                    return path_like
                try:
                    candidate = Path(path_like)
                except TypeError:
                    return path_like
                if candidate.is_absolute() or candidate.exists():
                    return path_like
                fallback = DATA_DIR / candidate
                if fallback.exists():
                    return str(fallback)
                return path_like

            _orig_read_hdf = pd.read_hdf
            _orig_read_pickle = pd.read_pickle
            _orig_read_csv = pd.read_csv
            _orig_read_parquet = pd.read_parquet
            _orig_open = builtins.open

            pd.read_hdf = lambda path_or_buf, *args, **kwargs: _orig_read_hdf(
                _resolve_data_path(path_or_buf), *args, **kwargs
            )
            pd.read_pickle = lambda filepath_or_buffer, *args, **kwargs: _orig_read_pickle(
                _resolve_data_path(filepath_or_buffer), *args, **kwargs
            )
            pd.read_csv = lambda filepath_or_buffer, *args, **kwargs: _orig_read_csv(
                _resolve_data_path(filepath_or_buffer), *args, **kwargs
            )
            pd.read_parquet = lambda path, *args, **kwargs: _orig_read_parquet(
                _resolve_data_path(path), *args, **kwargs
            )
            builtins.open = lambda file, *args, **kwargs: _orig_open(_resolve_data_path(file), *args, **kwargs)

            runpy.run_path({str(code_path)!r}, run_name="__main__")
            """
        ).strip() + "\n"

    @staticmethod
    def _resolve_execution_backend() -> str:
        backend = str(FACTOR_COSTEER_SETTINGS.execution_backend).strip().lower()
        if backend != "auto":
            return backend
        if _docker_daemon_available():
            return "docker"
        if _conda_env_exists(FACTOR_COSTEER_SETTINGS.execution_conda_env_name):
            return "conda"
        return "local"

    @staticmethod
    def _python_command_for_backend() -> str:
        backend = FactorFBWorkspace._resolve_execution_backend()
        if backend == "conda":
            env_name = FACTOR_COSTEER_SETTINGS.execution_conda_env_name
            return f"conda run -n {env_name} python"
        return FACTOR_COSTEER_SETTINGS.python_bin

    @staticmethod
    def _sanitize_execution_feedback(raw_feedback: str, execution_code_path: Path) -> str:
        feedback = (
            raw_feedback.replace(str(execution_code_path.parent.absolute()), r"/path/to")
            .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
        )
        if len(feedback) > 2000:
            feedback = feedback[:1000] + "....hidden long error message...." + feedback[-1000:]
        return feedback

    def _execute_locally(
        self,
        execution_code_path: Path,
        source_data_path: Path,
    ) -> tuple[bool, str]:
        command = f"{self._python_command_for_backend()} {execution_code_path.name}"
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.workspace_path,
            env={
                **os.environ,
                "FACTOR_DATA_DIR": str(source_data_path.resolve()),
                "RDAGENT_FACTOR_DATA_DIR": str(source_data_path.resolve()),
            },
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
            timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
        )
        return completed.returncode == 0, self._sanitize_execution_feedback(completed.stdout or "", execution_code_path)

    def _execute_in_docker(
        self,
        execution_code_path: Path,
        source_data_path: Path,
    ) -> tuple[bool, str]:
        docker_env = FactorDockerEnv()
        docker_env.prepare()

        resolved_data = source_data_path.resolve()
        extra_volumes = {
            str(resolved_data): {
                "bind": "/workspace/factor_data",
                "mode": "rw",
            }
        }
        # Resolve symlinks that point outside the mounted directory.
        # Docker does not follow symlinks escaping the mount root, so we
        # mount the real targets as additional volumes at the same path.
        if resolved_data.is_dir():
            for entry in resolved_data.iterdir():
                if entry.is_symlink():
                    real_target = entry.resolve()
                    # Only mount if the real target is outside the data dir
                    if not str(real_target).startswith(str(resolved_data)):
                        mount_point = f"/workspace/factor_data/{entry.name}"
                        extra_volumes[str(real_target)] = {
                            "bind": mount_point,
                            "mode": "ro",
                        }

        result = docker_env.run(
            local_path=str(self.workspace_path),
            entry=f"python {execution_code_path.name}",
            env={
                "FACTOR_DATA_DIR": "/workspace/factor_data",
                "RDAGENT_FACTOR_DATA_DIR": "/workspace/factor_data",
                "HDF5_USE_FILE_LOCKING": "FALSE",
                "JQ_USER": os.environ.get("JQ_USER", ""),
                "JQ_PASS": os.environ.get("JQ_PASS", ""),
            },
            running_extra_volume=extra_volumes,
        )
        return result.exit_code == 0, self._sanitize_execution_feedback(result.full_stdout or "", execution_code_path)

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. expose the shared source data directory to the execution process
        if call_factor_py is True:
            4. execute the code
        else:
            4. generate a script from template to import the factor.py dump get the factor value to result.parquet
        5. read the factor value from the output file in the workspace path folder
        returns the execution feedback as a string and the factor value as a pandas dataframe


        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a tuple with the following: (execution_feedback, executed_factor_value_dataframe, Optional[Exception])

        """
        self.before_execute()
        if self.file_dict is None or "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return self.FB_CODE_NOT_SET, None
        with FileLock(self.workspace_path / "execution.lock"):
            backend = self._resolve_execution_backend()
            if self.target_task.version == 1:
                source_data_path = (
                    Path(
                        FACTOR_COSTEER_SETTINGS.data_folder_debug,
                    )
                    if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                    else Path(
                        FACTOR_COSTEER_SETTINGS.data_folder,
                    )
                )
            elif self.target_task.version == 2:
                raise CustomRuntimeError("Only paper_factor factor tasks (version=1) are supported in this package.")

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = self.workspace_path / f"factor.py"

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None

            if self.target_task.version == 1:
                launcher_data_path = source_data_path.resolve()
                launcher_code_path = code_path
                if backend == "docker":
                    launcher_data_path = Path("/workspace/factor_data")
                    launcher_code_path = Path("factor.py")
                execution_code_path = self.workspace_path / self.EXECUTION_LAUNCHER
                execution_code_path.write_text(
                    self._build_shared_data_launcher(
                        source_data_path=launcher_data_path,
                        code_path=launcher_code_path,
                    ),
                    encoding="utf-8",
                )
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            try:
                if backend == "docker":
                    execution_success, execution_feedback = self._execute_in_docker(
                        execution_code_path=execution_code_path,
                        source_data_path=source_data_path,
                    )
                elif backend in {"local", "conda"}:
                    execution_success, execution_feedback = self._execute_locally(
                        execution_code_path=execution_code_path,
                        source_data_path=source_data_path,
                    )
                else:
                    raise RuntimeError(f"Unsupported factor execution backend: {backend}")

                if not execution_success:
                    if self.raise_exception:
                        raise CustomRuntimeError(execution_feedback)
                    execution_error = CustomRuntimeError(execution_feedback)
            except subprocess.TimeoutExpired:
                execution_feedback += (
                    f"Execution timeout error and the timeout is set to "
                    f"{FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                execution_error = CustomRuntimeError(execution_feedback)
            except Exception as e:
                if isinstance(e, CustomRuntimeError):
                    raise
                execution_feedback = str(e)
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback) from e
                execution_error = CustomRuntimeError(execution_feedback)

            workspace_output_file_path = self.workspace_path / "result.parquet"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_parquet(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading parquet file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
