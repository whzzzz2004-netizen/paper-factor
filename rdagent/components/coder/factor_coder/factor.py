from __future__ import annotations

import hashlib
import json
import os
import pickle
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
_D = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")
if not _D or not (_D/"stock_data"/"daily").exists():
    _D = Path(__file__).parent/"factor_implementation_source_data"
    if not (_D/"stock_data"/"daily").exists():
        _D = Path(__file__).parent.parent/"factor_implementation_source_data"
        if not (_D/"stock_data"/"daily").exists():
            _D = Path(".")
DATA_DIR = _D
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))
LOOKBACK_DAYS = {lookback_days}  # 由框架注入，0=不切片
# ── 增量更新：设 FACTOR_INCREMENTAL_START_DATE 环境变量则只算该日期之后的数据 ──
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _pos = max(0, pd.DatetimeIndex(TRADE_DATES).searchsorted(pd.Timestamp(_INC_START)) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_pos:]
# ── ──
_CODE_DIR = Path(__file__).parent

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

# ── 默认 calc_factor_series（用户自定义版本会覆盖此默认） ──
# 用户应定义 calc_factor_series(df, stock) 返回 pd.Series(index=日期, name="因子名")。
# 如果用户只定义了 calc_factor_single_stock，则使用默认包装（逐日调用）。
try:
    calc_factor_series  # 用户已定义 → 跳过
except NameError:
    def calc_factor_series(df, stock):
        '''default: call calc_factor_single_stock per day (users can override with vectorized version)'''
        _td_index = pd.DatetimeIndex(TRADE_DATES)
        _positions = np.searchsorted(df.index.values.astype('int64'), _td_index.values.astype('int64'), side='right')
        _result = pd.Series(index=pd.DatetimeIndex(TRADE_DATES), dtype=float)
        _has_any = False
        for i, td in enumerate(_td_index):
            pos = int(_positions[i])
            if pos == 0:
                continue
            start = max(0, pos - LOOKBACK_DAYS - 1)
            sub = df.iloc[start:pos]
            try:
                r = calc_factor_single_stock(sub, td, stock)
            except Exception:
                r = None
            if r:
                for _k, _v in r.items():
                    if not (np.isnan(_v) or np.isinf(_v)):
                        _result.loc[td] = _v
                        _has_any = True
                    break
        _result.name = "factor"
        return _result if _has_any else pd.Series(dtype=float, name="factor")
    # ── /默认 calc_factor_series ──

def _compute_stock(stock, _LOAD_COLS=None):
    try:
        df = load_stock(stock, _LOAD_COLS)
        if df.empty:
            return []
        results = []
        _td_index = pd.DatetimeIndex(TRADE_DATES)
        _series = calc_factor_series(df, stock)
        if _series is not None and isinstance(_series, pd.Series) and len(_series) > 0:
            _fname = _series.name if _series.name else 'factor'
            for i, td in enumerate(_td_index):
                if i < LOOKBACK_DAYS:
                    continue
                if td in _series.index:
                    _val = _series.loc[td]
                    if isinstance(_val, pd.Series):  # 重复日期兜底：取最后一行
                        _val = _val.iloc[-1]
                    if not (np.isnan(_val) or np.isinf(_val)):
                        results.append({{"datetime": str(td.date()), "instrument": stock, _fname: float(_val)}})
            return results
        return results
    except Exception:
        return []  # 单只股票异常不阻塞整批，避免 joblib 线程卡死

if __name__ == '__main__':
    try:
        # ── 自动列推断：分析用户函数，只加载需要的列 ──
        import re as _re, inspect as _inspect, pyarrow.parquet as _pq
        _SAMPLE_FILE = next(STOCK_DATA_DIR.glob("*.parquet"))
        _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {'datetime', 'instrument'}
        _USER_SOURCE = ""
        try:
            _USER_SOURCE += _inspect.getsource(calc_factor_single_stock)
        except Exception:
            pass
        try:
            _USER_SOURCE += "\\n" + _inspect.getsource(calc_factor_series)
        except Exception:
            pass
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
        for _idx, stock_results in enumerate(Parallel(n_jobs=N_JOBS, backend="threading")(
            delayed(_compute_stock)(s, _LOAD_COLS) for s in STOCK_LIST
        )):
            all_records.extend(stock_results)
            if (_idx + 1) % 500 == 0 or (_idx + 1) == _total:
                print(f"  进度: {_idx+1}/{_total}", flush=True)
        long_df = pd.DataFrame(all_records)
        if long_df.empty:
            _debug = (f"all_records 为空! TRADE_DATES={len(TRADE_DATES)}条, "
                      f"STOCK_LIST={len(STOCK_LIST)}只, "
                      f"LOOKBACK_DAYS={LOOKBACK_DAYS}, "
                      f"DATA_DIR={DATA_DIR}")
            print(f"  ❌ {_debug}", flush=True)
            raise RuntimeError(_debug)
        long_df["datetime"] = pd.to_datetime(long_df["datetime"])
        factor_name = [c for c in long_df.columns if c not in ("datetime", "instrument")][0]
        wide = long_df.pivot(index="datetime", columns="instrument", values=factor_name)
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "trade_date"
        wide.columns.name = "stock_code"
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide = wide.reindex(index=pd.DatetimeIndex(TRADE_DATES, name=wide.index.name),
                            columns=pd.Index(STOCK_LIST, name=wide.columns.name))
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
        wide.to_parquet(_CODE_DIR / f"{Path(__file__).stem.removesuffix('.code')}.parquet")
        print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}}, 只股票")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        pass
"""

    # 分钟线框架代码模板（预加载 + ProcessPoolExecutor chunk 并行）
    MINUTE_FRAMEWORK_TEMPLATE = """import pandas as pd
import numpy as np
import sys, json, os, gc, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as _mp

warnings.filterwarnings("ignore")
pd.set_option("mode.copy_on_write", False)

_D = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")
if not _D or not (_D/"stock_data"/"minute_by_date").exists():
    _D = Path(__file__).parent/"factor_implementation_source_data"
    if not (_D/"stock_data"/"minute_by_date").exists():
        _D = Path(__file__).parent.parent/"factor_implementation_source_data"
        if not (_D/"stock_data"/"minute_by_date").exists():
            _D = Path(".")
DATA_DIR = _D
MINUTE_BY_DATE_DIR = DATA_DIR / "stock_data" / "minute_by_date"
# chunk 目录按 CHUNK_SIZE 分目录缓存，避免不同分片尺寸的因子互相污染（管线=15 vs 部署代码=25）
_CHUNK_SIZE = int(os.environ.get("FACTOR_CHUNK_SIZE", "25"))
_CHUNK_DIR = MINUTE_BY_DATE_DIR / f"_minute_chunks_c{_CHUNK_SIZE}"
STOCK_LIST = json.load(open(MINUTE_BY_DATE_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(MINUTE_BY_DATE_DIR / "trade_dates.json"))
LOOKBACK_DAYS = min(max(1, {lookback_days}), 120)  # 分钟线至少1天，不超过120天（约6个月）
# ── 增量更新：设 FACTOR_INCREMENTAL_START_DATE 环境变量则只算该日期之后的数据 ──
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _pos = max(0, pd.DatetimeIndex(TRADE_DATES).searchsorted(pd.Timestamp(_INC_START)) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_pos:]
# ── ──
_CODE_DIR = Path(__file__).parent

N_WORKERS = int(os.environ.get("FACTOR_N_WORKERS", str(min(4, os.cpu_count() or 4))))

# 列过滤（由LLM自动推断）
{_LOAD_COLS_DEF}

def load_day(td):
    return pd.read_parquet(MINUTE_BY_DATE_DIR / f"{{td}}.parquet", columns=_LOAD_COLS)

# 行业分类
_DAILY_DATA_DIR = DATA_DIR / "stock_data" / "daily"
_INDUSTRY_FILE = _DAILY_DATA_DIR / "industry.json"
INDUSTRY_DICT = json.load(open(_INDUSTRY_FILE, encoding="utf-8")) if _INDUSTRY_FILE.exists() else {{}}

def get_jq_data(symbol, data_type='price', start_date='2018-01-01', end_date='2026-05-15'):
    import hashlib as _hashlib
    _cache_key = f"jq_{{data_type}}_{_hashlib.md5(symbol.encode()).hexdigest()[:8]}"
    _cache_path = _DAILY_DATA_DIR / f"{{_cache_key}}.parquet"
    if _cache_path.exists():
        return pd.read_parquet(_cache_path)
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
                from concurrent.futures import ThreadPoolExecutor as _TPE2, TimeoutError as _TErr
                _tp = _TPE2(max_workers=1)
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


# ── 自动列推断 ──
import re as _re, inspect as _inspect, pyarrow.parquet as _pq
try:
    _SAMPLE_FILE = next(MINUTE_BY_DATE_DIR.glob("*.parquet"))
    _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {{'datetime', 'instrument'}}
    _USER_SOURCE = _inspect.getsource(calc_factors_one_day)
    try:
        _USER_SOURCE += _inspect.getsource(calc_factor_series)
    except Exception:
        pass
    _ALL_QUOTED = set(_re.findall(r'''['\"](\w+)['\"]''', _USER_SOURCE))
    _DETECTED = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
    if _DETECTED:
        _LOAD_COLS = _DETECTED
except Exception:
    pass
print(f"检测到因子使用的列: {{_LOAD_COLS}}", flush=True)

# ── 列配置 + 文件列表 ──
_READ_COLS = sorted(set((_LOAD_COLS or []) + ['close']))
_ALL_COLS = sorted(set(_pq.read_schema(next(MINUTE_BY_DATE_DIR.glob("*.parquet"))).names) - {'datetime', 'instrument'})
_FILES = sorted([MINUTE_BY_DATE_DIR / f"{d}.parquet" for d in TRADE_DATES if (MINUTE_BY_DATE_DIR / f"{d}.parquet").exists()])

# ── 按 Chunk 并行计算（各 worker 从预分片文件加载，无 I/O 冗余）──
def _compute_chunk(chunk_stocks, chunk_idx, chunk_file, read_cols):
    if not chunk_file.exists():
        return None, chunk_idx
    _WDATA = pd.read_parquet(chunk_file, columns=read_cols)

    def _proc_one(stock):
        # 处理单只股票，返回记录列表或None
        try:
            sub = _WDATA.xs(stock, level='instrument')
        except KeyError:
            return None
        if sub.empty:
            return None
        sub.index = pd.DatetimeIndex(sub.index.values)

        # ── 向量化模式：calc_factor_series 一次返回整个序列 ──
        try:
            if 'calc_factor_series' in globals():
                _series_ret = calc_factor_series(sub, stock)
                if _series_ret is not None and isinstance(_series_ret, pd.Series) and len(_series_ret) > 0:
                    stock_records = []
                    if hasattr(_series_ret.index, 'date'):
                        _series_ret.index = pd.Index(_series_ret.index.date)
                    _fname_series = _series_ret.name if _series_ret.name is not None else "factor"
                    for _idx_date in _series_ret.index:
                        _val = _series_ret.loc[_idx_date]
                        if not (isinstance(_val, float) and np.isnan(_val)):
                            try:
                                stock_records.append({{
                                    "datetime": str(_idx_date),
                                    "instrument": stock,
                                    _fname_series: float(_val)
                                }})
                            except (ValueError, TypeError):
                                pass
                    if stock_records:
                        return stock_records
        except Exception:
            pass

        # ── 非向量化路径：calc_factors_one_day 逐日滑动窗口 ──
        # 预切片缓存：按日拆分，避免重复 isin
        _norm_idx = sub.index.normalize()
        _uniq_dates = sorted(_norm_idx.unique())
        # TRADE_DATES 可能为 '20180102'(无横线) 或 '2025-02-19'(带横线)，统一归一化为 date 对象再匹配
        _td_set = set(pd.Timestamp(d).date() for d in TRADE_DATES)
        _date_slices = {}
        for _dt in _uniq_dates:
            _mask = _norm_idx == _dt
            _slice = sub.loc[_mask]
            if not _slice.empty:
                _date_slices[_dt] = _slice

        stock_records = []
        for _i, _dt in enumerate(_uniq_dates):
            _dt_str = _dt.strftime('%Y-%m-%d')
            if _dt.date() not in _td_set:
                continue
            _start_idx = max(0, _i - LOOKBACK_DAYS + 1)
            _window_dates = _uniq_dates[_start_idx:_i + 1]
            _chunks = [_date_slices[d] for d in _window_dates if d in _date_slices]
            if not _chunks:
                continue
            _sub = pd.concat(_chunks) if len(_chunks) > 1 else _chunks[0]
            try:
                _dr = calc_factors_one_day(_sub, stock)
            except Exception:
                continue
            if _dr is None:
                continue
            if isinstance(_dr, pd.Series):
                if _dr.name is not None and _dr.name not in ("", "factor"):
                    # 取最后一个值（最新日期），兼容 LOOKBACK>1 滑动窗口返回多日结果
                    _fv = _dr.iloc[-1] if len(_dr) > 0 else None
                    if _fv is not None and pd.notna(_fv) and np.isfinite(float(_fv)):
                        stock_records.append({{
                            "datetime": _dt_str,
                            "instrument": stock,
                            str(_dr.name): float(_fv)
                        }})
                else:
                    # 无 name → 用 index 值做列名（兼容多因子返回）
                    for _fn in _dr.index:
                        _v = _dr.loc[_fn]
                        if pd.notna(_v):
                            try:
                                _fv = float(_v)
                                if np.isfinite(_fv):
                                    stock_records.append({{
                                        "datetime": _dt_str,
                                        "instrument": stock,
                                        str(_fn): _fv
                                    }})
                            except (ValueError, TypeError):
                                pass
            else:
                # 标量返回
                if pd.notna(_dr):
                    try:
                        _fv = float(_dr)
                        if np.isfinite(_fv):
                            stock_records.append({{
                                "datetime": _dt_str,
                                "instrument": stock,
                                "factor": _fv
                            }})
                    except (ValueError, TypeError):
                        pass
        return stock_records if stock_records else None

    # ── 逐股票顺序处理（chunk 级并行由外层 ProcessPoolExecutor 提供）──
    records = []
    for stock in chunk_stocks:
        _r = _proc_one(stock)
        if _r:
            records.extend(_r)
    # 写中间结果到文件，避免 spawn pipe deadlock
    if records:
        pd.DataFrame(records).to_parquet(_CHUNK_DIR / f"_result_{chunk_idx}.pq")
    return chunk_idx


if __name__ == '__main__':
    t0 = time.time()
    long_df = None

    _ALL_STOCKS = STOCK_LIST  # 从 stock_list.json 取全量5435只（字符串类型，与parquet instrument匹配）
    _N = len(_ALL_STOCKS)
    _n_files = len(_FILES)
    _n_chunks = (_N + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    print(f"{{_N}} 只股票, {{_n_files}} 文件, {{_n_chunks}} chunks x{{_CHUNK_SIZE}}, {{N_WORKERS}} 进程", flush=True)

    # ── 预分片：一趟顺序扫描，按 chunk 拆分到独立 parquet（流式写入，低内存）──
    # 共享chunk目录，跨因子复用（第一因子生成后后续跳过）
    import pyarrow as _pa
    _t_split = time.time()
    _CHUNKS_LIST = [_ALL_STOCKS[i:i+_CHUNK_SIZE] for i in range(0, _N, _CHUNK_SIZE)]
    _CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    _CHUNK_FILES = [_CHUNK_DIR / f"_chunk_{{ci}}.pq" for ci in range(len(_CHUNKS_LIST))]

    _MANIFEST = _CHUNK_DIR / "_manifest.json"
    _STOCKS_KEY = sorted(STOCK_LIST)
    _MANIFEST_DATA = json.load(open(_MANIFEST)) if _MANIFEST.exists() else None
    _chunks_ok = (
        all(cf.exists() for cf in _CHUNK_FILES)
        and _MANIFEST_DATA is not None
        and _MANIFEST_DATA.get("stocks") == _STOCKS_KEY
        and _MANIFEST_DATA.get("chunk_size") == _CHUNK_SIZE
    )

    if _chunks_ok:
        print(f"共享chunk已存在且股票列表/尺寸匹配: {{_CHUNK_DIR}}, 跳过预分片 ({{time.time()-_t_split:.0f}}s)", flush=True)
    else:
        if all(cf.exists() for cf in _CHUNK_FILES):
            print(f"⚠️ 股票列表/CHUNK_SIZE 变化或 manifest 缺失，重新预分片 ({{_CHUNK_DIR}})", flush=True)
        _writers = [None] * len(_CHUNKS_LIST)
        _stock2ci = {{}}
        for _ci, _cstocks in enumerate(_CHUNKS_LIST):
            for _s in _cstocks:
                _stock2ci[_s] = _ci

        try:
            for _f_idx, _f in enumerate(_FILES):
                _df = pd.read_parquet(_f, columns=_ALL_COLS)
                _ix = _df.index.get_level_values('instrument')
                # 一趟映射：每行 -> chunk index（比 119 次 isin 快 100x）
                _ci_s = pd.Series(_ix).map(_stock2ci)
                _valid_mask = _ci_s.notna()
                if not _valid_mask.any():
                    continue
                _df = _df.iloc[_valid_mask.values].copy()
                _df['_chunk_ci'] = _ci_s[_valid_mask].astype(int).values
                for _ci, _gdf in _df.groupby('_chunk_ci'):
                    _gdf = _gdf.drop(columns=['_chunk_ci'])
                    _table = _pa.Table.from_pandas(_gdf, preserve_index=True)
                    if _writers[_ci] is None:
                        _writers[_ci] = _pq.ParquetWriter(_CHUNK_FILES[_ci], _table.schema)
                    _writers[_ci].write_table(_table)
                if _f_idx % 200 == 0:
                    print(f"  分片进度: {{_f_idx}}/{{_n_files}} 文件", flush=True)
        finally:
            for _w in _writers:
                if _w is not None:
                    _w.close()
        _n_created = sum(1 for f in _CHUNK_FILES if f.exists())
        print(f"分片完成: {{time.time()-_t_split:.0f}}s, {{_n_created}}/{{len(_CHUNKS_LIST)}} chunk 文件, "
              f"总 {{int(sum(f.stat().st_size for f in _CHUNK_FILES if f.exists())/1024**3)}} GB", flush=True)
        with open(_MANIFEST, "w") as _mf:
            json.dump({"stocks": _STOCKS_KEY, "chunk_size": _CHUNK_SIZE, "n_chunks": len(_CHUNKS_LIST)}, _mf)

    # ── 清理大对象，腾出 fork 内存 ──
    try:
        del _writers
    except NameError:
        pass
    try:
        del _stock2ci
    except NameError:
        pass
    try:
        del _df
    except NameError:
        pass
    gc.collect()

    # ── 计算：ProcessPoolExecutor 并行处理 chunks，结果写文件避免 pipe deadlock ──
    _t1 = time.time()
    _active_chunks = [(ci, cf) for ci, cf in enumerate(_CHUNK_FILES) if cf.exists()]
    _n_futs = len(_active_chunks)
    _done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=_mp.get_context("spawn")) as _pool:
        _futs = {{_pool.submit(_compute_chunk, _CHUNKS_LIST[ci], ci, cf, _READ_COLS): ci for ci, cf in _active_chunks}}
        for _fut in as_completed(_futs):
            try:
                _fut.result(timeout=600)  # 单chunk超时10分钟
            except Exception as _exc:
                print(f"  ⚠️ chunk {{_futs[_fut]}} 异常: {{_exc}}", flush=True)
            _done += 1
            if _done % max(1, _n_futs // 10) == 0 or _done == _n_futs:
                _elapsed = time.time() - _t1
                _pct = _done / _n_futs * 100
                _eta = _elapsed / _done * _n_futs if _done > 0 else 0
                print(f"  进度: {{_done}}/{{_n_futs}} chunks ({{_pct:.0f}}%), "
                      f"{{_elapsed:.0f}}s, ETA {{_eta-_elapsed:.0f}}s", flush=True)
    # 从结果文件读取（逐个 pivot → join，避免大 list 爆内存）
    wide = None
    _rec_cnt = 0
    for _ci in range(len(_CHUNKS_LIST)):
        _rf = _CHUNK_DIR / f"_result_{_ci}.pq"
        if _rf.exists():
            try:
                _rd = pd.read_parquet(_rf)
                _rec_cnt += len(_rd)
                _rd["datetime"] = pd.to_datetime(_rd["datetime"])
                _factor_name = [c for c in _rd.columns if c not in ("datetime", "instrument")][0]
                _w = _rd.pivot(index="datetime", columns="instrument", values=_factor_name)
                if wide is None:
                    wide = _w
                else:
                    wide = wide.join(_w, how="outer")
                _rf.unlink()
                del _rd, _w
            except Exception as _exc:
                print(f"  ⚠️ 结果文件读取失败 _result_{_ci}.pq: {{_exc}}", flush=True)
    print(f"  计算完成: {{time.time()-_t1:.0f}}s, {{_rec_cnt}} 条记录", flush=True)

    if wide is None or wide.empty:
        print("警告：没有产生任何因子值！", flush=True)
        wide = pd.DataFrame(index=pd.Index(TRADE_DATES, name="trade_date"),
                           columns=pd.Index(STOCK_LIST, name="stock_code"), dtype=float)
    else:
        wide = wide.sort_index().sort_index(axis=1)
        wide.index.name = "trade_date"
        wide.columns.name = "stock_code"
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide = wide.reindex(index=pd.DatetimeIndex(TRADE_DATES, name=wide.index.name),
                            columns=pd.Index(STOCK_LIST, name=wide.columns.name))
        wide.attrs["factor_name"] = _factor_name
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
        wide.index = wide.index.strftime('%Y-%m-%d')
        # 重索引到全量日期×全量股票，缺值用NaN（在strftime之后，因为TRADE_DATES是字符串）
        wide = wide.reindex(index=pd.DatetimeIndex(TRADE_DATES).strftime('%Y-%m-%d'),
                            columns=pd.Index(STOCK_LIST, name=wide.columns.name))
        wide.columns = wide.columns.astype(int)
        wide.to_parquet(_CODE_DIR / f"{Path(__file__).stem.removesuffix('.code')}.parquet")
        print(f"完成！{{wide.shape[0]}} 天 x {{wide.shape[1]}} 只股票, "
              f"{{time.time()-t0:.0f}}s", flush=True)"""

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
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as _mp

_D = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")
if not _D or not (_D/"stock_data"/"daily").exists():
    _D = Path(__file__).parent/"factor_implementation_source_data"
    if not (_D/"stock_data"/"daily").exists():
        _D = Path(__file__).parent.parent/"factor_implementation_source_data"
        if not (_D/"stock_data"/"daily").exists():
            _D = Path(".")
DATA_DIR = _D
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))
LOOKBACK_DAYS = {lookback_days}  # 由框架注入，0=不切片
# ── 增量更新：设 FACTOR_INCREMENTAL_START_DATE 环境变量则只算该日期之后的数据 ──
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _pos = max(0, pd.DatetimeIndex(TRADE_DATES).searchsorted(pd.Timestamp(_INC_START)) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_pos:]
# ── ──
_CODE_DIR = Path(__file__).parent
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

# ── 进程缓存（spawn 模式下各进程独立加载） ──
_WCACHE = {}
_WPOS = {}
_WVALID = None
_WTDIDX = None
_SD = None
_LOAD_COLS = None

def _init_shared():
    \"\"\"初始化全局变量（主进程）\"\"\"
    global _WVALID, _WTDIDX, _SD, _LOAD_COLS
    _SD = STOCK_DATA_DIR
    _WVALID = STOCK_LIST
    _WTDIDX = pd.DatetimeIndex(TRADE_DATES)
    print(f"  [主进程] 共享缓存就绪，{len(STOCK_LIST)}只股票按需加载", flush=True)

def _init_worker():
    \"\"\"子进程初始化（spawn 模式下子进程需要重新初始化共享变量）\"\"\"
    global _WVALID, _WTDIDX, _SD, _WCACHE, _WPOS
    _SD = STOCK_DATA_DIR
    _WVALID = STOCK_LIST
    _WTDIDX = pd.DatetimeIndex(TRADE_DATES)
    _WCACHE = {}
    _WPOS = {}

def _get_stock(s):
    \"\"\"延迟加载 — 全量位置预计算，跨chunk复用\"\"\"
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

        _CHK_DIR = _CODE_DIR / "checkpoints"
        _CHK_DIR.mkdir(exist_ok=True)
        _CHUNK = 200
        _t0_main = time.time()

        print(f"截面计算: {len(TRADE_DATES)} 天, chunk={_CHUNK} 天, processes={N_WORKERS}", flush=True)
        print(f"并行模式: ProcessPoolExecutor (spawn)", flush=True)

        # 初始化全局共享缓存
        _init_shared()

        _ranges = list(range(0, len(TRADE_DATES), _CHUNK))

        # ProcessPoolExecutor：spawn 上下文避免 pyarrow fork 不兼容
        # 创建在 chunk 循环外，worker 跨 chunk 复用，缓存累积
        _ctx = _mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, mp_context=_ctx) as _pool:
            for _ci, _cs in enumerate(_ranges):
                _ce = min(_cs + _CHUNK, len(TRADE_DATES))
                _t_chk = time.time()

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
        wide = wide.reindex(index=pd.DatetimeIndex(TRADE_DATES, name=wide.index.name),
                            columns=pd.Index(STOCK_LIST, name=wide.columns.name))
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
        wide.to_parquet(_CODE_DIR / f"{Path(__file__).stem.removesuffix('.code')}.parquet")
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
import sys, json, os, time
from pathlib import Path
import gc as _gc
from collections import OrderedDict
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed

_D = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")
if not _D or not (_D/"stock_data"/"minute_by_date").exists():
    _D = Path(__file__).parent/"factor_implementation_source_data"
    if not (_D/"stock_data"/"minute_by_date").exists():
        _D = Path(__file__).parent.parent/"factor_implementation_source_data"
        if not (_D/"stock_data"/"minute_by_date").exists():
            _D = Path(".")
DATA_DIR = _D
MINUTE_BY_DATE_DIR = DATA_DIR / "stock_data" / "minute_by_date"
STOCK_LIST = json.load(open(MINUTE_BY_DATE_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(MINUTE_BY_DATE_DIR / "trade_dates.json"))
LOOKBACK_DAYS = min(max(1, {lookback_days}), 120)  # 分钟线至少1天，不超过120天（约6个月）
# ── 增量更新：设 FACTOR_INCREMENTAL_START_DATE 环境变量则只算该日期之后的数据 ──
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _pos = max(0, pd.DatetimeIndex(TRADE_DATES).searchsorted(pd.Timestamp(_INC_START)) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_pos:]
# ── ──
_CODE_DIR = Path(__file__).parent

N_WORKERS = int(os.environ.get("FACTOR_N_WORKERS", "2"))  # 多进程并行，默认2防OOM

{user_code}

# ── 自动列推断：分析用户函数，只加载需要的列 ──
import re as _re, inspect as _inspect, pyarrow.parquet as _pq
_LOAD_COLS = None
try:
    _SAMPLE_FILE = next(MINUTE_BY_DATE_DIR.glob("*.parquet"))
    _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {{'datetime', 'instrument'}}
    _USER_SOURCE = ""
    try:
        _USER_SOURCE += _inspect.getsource(calc_factor_minute_raw)
    except Exception:
        pass
    try:
        _USER_SOURCE += "\\n" + _inspect.getsource(cross_section_transform)
    except Exception:
        pass
    _ALL_QUOTED = set(_re.findall(r'''['\"](\\w+)['\"]''', _USER_SOURCE))
    _LOAD_COLS = sorted(_ALL_QUOTED & _AVAILABLE_COLS) if _ALL_QUOTED else None
    if not _LOAD_COLS:
        _LOAD_COLS = None
except Exception:
    pass
print(f"检测到因子使用的列: {{_LOAD_COLS}}", flush=True)
# ──

# ── LRU Parquet Cache（每个 worker 独立，限制内存）──
_PARQUET_CACHE = OrderedDict()
_CACHE_MAX_SIZE = int(os.environ.get("FACTOR_CACHE_SIZE", "20"))

def _load_minute_data(td_str):
    if td_str in _PARQUET_CACHE:
        _PARQUET_CACHE.move_to_end(td_str)
        return _PARQUET_CACHE[td_str]
    df = pd.read_parquet(MINUTE_BY_DATE_DIR / f"{{td_str}}.parquet", columns=_LOAD_COLS)
    if len(_PARQUET_CACHE) >= _CACHE_MAX_SIZE:
        _PARQUET_CACHE.popitem(last=False)
    _PARQUET_CACHE[td_str] = df
    return df

def _compute_day(td):
    \"\"\"单日计算：并行I/O加载滑动窗口 + 逐股票计算 + 截面变换\"\"\"
    idx = TRADE_DATES.index(td)
    start_idx = max(0, idx - LOOKBACK_DAYS + 1)
    window_dates = TRADE_DATES[start_idx:idx + 1]

    _window_dates = [d for d in window_dates if (MINUTE_BY_DATE_DIR / f"{{d}}.parquet").exists()]
    if not _window_dates:
        return []
    # 并行 I/O 加载
    if len(_window_dates) > 4:
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _AC
        _dfs = []
        with _TPE(max_workers=min(8, len(_window_dates))) as _io_pool:
            _io_futs = {{_io_pool.submit(_load_minute_data, _d): _d for _d in _window_dates}}
            for _io_f in _AC(_io_futs):
                _dfs.append(_io_f.result())
    else:
        _dfs = [_load_minute_data(d) for d in _window_dates]
    all_data = pd.concat(_dfs)
    del _dfs

    # 逐股票计算（groupby 比 xs 快 15x，对全量 5000+ 股票至关重要）
    raw = {{}}
    for stk, _grp in all_data.groupby(level='instrument'):
        try:
            grp = _grp.droplevel('instrument')
            val = calc_factor_minute_raw(grp, stk)
            if val:
                raw[stk] = list(val.values())[0]
        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
            pass

    if not raw:
        return []
    del all_data

    # 截面变换
    transformed = cross_section_transform(raw)
    del raw

    # 记录输出：查找第一个 dict 值来确定因子名
    _fname = None
    for __v in transformed.values():
        if isinstance(__v, dict):
            _fname = list(__v.keys())[0]
            break
    if _fname is None:
        _fname = "factor"

    records = []
    for stock, val in transformed.items():
        if isinstance(val, dict):
            records.append({{"datetime": td, "instrument": stock, **val}})
        else:
            records.append({{"datetime": td, "instrument": stock, _fname: val}})
    return records

# ── Checkpoint 配置 ──
_CHUNK_SIZE = int(os.environ.get("FACTOR_CHUNK_SIZE", "200"))


if __name__ == '__main__':
    try:
        _t0_main = time.time()

        _all_dates = [d for d in TRADE_DATES if (MINUTE_BY_DATE_DIR / f"{{d}}.parquet").exists()]
        _base_name = Path(__file__).stem.removesuffix('.code')

        # ── Checkpoint 扫描：跳过已完成日期 ──
        _ckpt_dir = _CODE_DIR / f".checkpoints_{{_base_name}}"
        _ckpt_dir.mkdir(exist_ok=True)
        _completed = set()
        for _ckpt in sorted(_ckpt_dir.glob("*.parquet")):
            _ckpt_df = pd.read_parquet(_ckpt)
            _completed.update(pd.to_datetime(_ckpt_df['datetime']).dt.strftime('%Y-%m-%d'))

        _pending = [d for d in _all_dates if d not in _completed]
        _n_pending = len(_pending)
        _n_chunks = (_n_pending + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        _ckpt_idx = 0

        if not _pending:
            print(f"所有 {{len(_all_dates)}} 个日期均已完成，跳过计算", flush=True)
        else:
            print(f"共 {{len(_all_dates)}} 个交易日，已完成 {{len(_completed)}}，"
                  f"待处理 {{len(_pending)}} ({{N_WORKERS}} workers)", flush=True)

        _ckpt_idx = 0
        for _ci in range(_n_chunks):
            _cs = _ci * _CHUNK_SIZE
            _ce = min(_cs + _CHUNK_SIZE, _n_pending)
            _chunk_dates = _pending[_cs:_ce]

            _t0_chunk = time.time()
            _chunk_records = []
            with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(_chunk_dates)),
                                     mp_context=_mp.get_context("spawn")) as _pool:
                _futs = {{_pool.submit(_compute_day, td): td for td in _chunk_dates}}
                for _fut in as_completed(_futs):
                    _res = _fut.result()
                    if _res:
                        _chunk_records.extend(_res)

            # with 块退出已自动 shutdown，无需额外清理

            if _chunk_records:
                _ckpt_df = pd.DataFrame(_chunk_records)
                _ckpt_df.to_parquet(_ckpt_dir / f"_{{_base_name}}_{_ckpt_idx:04d}.parquet")
                _ckpt_idx += 1
                del _ckpt_df

            _elapsed = time.time() - _t0_chunk
            _total_elapsed = time.time() - _t0_main
            _done = _ce
            _pct = _done / _n_pending * 100 if _n_pending else 100
            _rate = _done / _total_elapsed if _total_elapsed > 0 else 0
            _eta = (_n_pending - _done) / _rate if _rate > 0 else 0
            print(f"Chunk {{_ci+1}}/{{_n_chunks}}: {{_done}}/{{_n_pending}}天 ({{_pct:.0f}}%), "
                  f"chunk {{_elapsed:.0f}}s, 累计 {{_total_elapsed:.0f}}s, ETA {{_eta:.0f}}s",
                  flush=True)

            del _chunk_records
            _gc.collect()

        # ── Merge Checkpoints ──
        print(f"合并 {{_ckpt_idx}} 个 checkpoint...", flush=True)
        _all_parts = []
        for _ckpt in sorted(_ckpt_dir.glob("*.parquet")):
            _all_parts.append(pd.read_parquet(_ckpt))
        if _all_parts:
            long_df = pd.concat(_all_parts, ignore_index=True)
            del _all_parts
        else:
            long_df = pd.DataFrame()

        # Cleanup checkpoints
        import shutil as _shutil
        _shutil.rmtree(_ckpt_dir)

        if long_df.empty:
            print("无有效数据，退出")
            sys.exit(0)

        long_df["datetime"] = pd.to_datetime(long_df["datetime"])
        factor_cols = [c for c in long_df.columns if c not in ("datetime", "instrument")]
        for _fc in factor_cols:
            _g_wide = long_df.pivot(index="datetime", columns="instrument", values=_fc)
            _g_wide = _g_wide.sort_index().sort_index(axis=1)
            _g_wide = _g_wide.replace([np.inf, -np.inf], np.nan)
            _g_wide = _g_wide.reindex(index=pd.DatetimeIndex(TRADE_DATES, name=_g_wide.index.name),
                                      columns=pd.Index(STOCK_LIST, name=_g_wide.columns.name))
            _g_wide.attrs["factor_name"] = _fc
            # 涨停剔除(minute_cs)
            _LU_PATH = DATA_DIR / "limit_up_daily.parquet"
            if _LU_PATH.exists():
                _lu_df = pd.read_parquet(_LU_PATH, columns=['datetime', 'instrument'])
                for _lu_dt, _lu_grp in _lu_df.groupby(_lu_df['datetime'].dt.normalize()):
                    if _lu_dt in _g_wide.index:
                        _c = [str(x) for x in _lu_grp['instrument'] if str(x) in _g_wide.columns]
                        if _c:
                            _g_wide.loc[_lu_dt, _c] = np.nan
            # /涨停剔除
            _g_wide.index = _g_wide.index.strftime('%Y-%m-%d')
            _g_wide.columns = _g_wide.columns.astype(int)
            _out_path = _CODE_DIR / (f"{{_base_name}}.parquet" if len(factor_cols) == 1 else f"{{_base_name}}_{{_fc}}.parquet")
            _g_wide.to_parquet(_out_path)
            print(f"保存因子 {{_fc}}: {{_g_wide.shape[0]}} 天 x {{_g_wide.shape[1]}} 只股票, "
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

_D = Path(os.environ.get("FACTOR_DATA_DIR") or os.environ.get("RDAGENT_FACTOR_DATA_DIR") or "")
if not _D or not (_D/"stock_data"/"daily").exists():
    _D = Path(__file__).parent/"factor_implementation_source_data"
    if not (_D/"stock_data"/"daily").exists():
        _D = Path(__file__).parent.parent/"factor_implementation_source_data"
        if not (_D/"stock_data"/"daily").exists():
            _D = Path(".")
DATA_DIR = _D
STOCK_DATA_DIR = DATA_DIR / "stock_data" / "daily"
STOCK_LIST = json.load(open(STOCK_DATA_DIR / "stock_list.json"))
TRADE_DATES = json.load(open(STOCK_DATA_DIR / "trade_dates.json"))
LOOKBACK_DAYS = {lookback_days}  # 由框架注入，0=不切片
# ── 增量更新：设 FACTOR_INCREMENTAL_START_DATE 环境变量则只算该日期之后的数据 ──
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _pos = max(0, pd.DatetimeIndex(TRADE_DATES).searchsorted(pd.Timestamp(_INC_START)) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_pos:]
# ── ──
_CODE_DIR = Path(__file__).parent

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

# 列过滤（由LLM自动推断）
{_LOAD_COLS_DEF}

{user_code}

if __name__ == '__main__':
    try:
        # ── 自动列推断：分析用户函数，只加载需要的列 ──
        import re as _re, inspect as _inspect, pyarrow.parquet as _pq
        _SAMPLE_FILE = next(STOCK_DATA_DIR.glob("*.parquet"))
        _AVAILABLE_COLS = set(_pq.read_schema(_SAMPLE_FILE).names) - {'datetime', 'instrument'}
        _USER_SOURCE = open(__file__, encoding="utf-8").read()
        # 提取代码中所有引号字符串，与可用列取交集（覆盖辅助函数、模块级代码等所有位置）
        _ALL_QUOTED = set(_re.findall(r'''['"]([A-Za-z_][A-Za-z0-9_]*)['"]''', _USER_SOURCE))
        _DETECTED = sorted(_ALL_QUOTED & _AVAILABLE_COLS)
        _LOAD_COLS = sorted(set((_LOAD_COLS or []) + _DETECTED))  # 注入值 ∪ 扫描值
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
        # 训练切片用 side='left'（严格早于当年第一天），避免把当年首个交易日纳入训练（off-by-one）
        _stock_positions_train = {{}}
        for stock, df in all_data.items():
            _stock_positions[stock] = np.searchsorted(df.index.values.astype('int64'), _td_index.values.astype('int64'), side='right')
            _stock_positions_train[stock] = np.searchsorted(df.index.values.astype('int64'), _td_index.values.astype('int64'), side='left')

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
                pos = _stock_positions_train[stock][_date_idxs[0]]
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
        wide = wide.reindex(index=pd.DatetimeIndex(TRADE_DATES, name=wide.index.name),
                            columns=pd.Index(STOCK_LIST, name=wide.columns.name))
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
        wide.to_parquet(_CODE_DIR / f"{Path(__file__).stem.removesuffix('.code')}.parquet")
        print(f"完成，共 {{wide.shape[0]}} 天 x {{wide.shape[1]}} 只股票", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        pass"""

    # Target function names that constitute the "user code" portion.
    # Everything else (imports, DATA_DIR, load_stock, _compute_stock, __main__)
    # is framework boilerplate and should be stripped before re-wrapping.
    _TARGET_FUNC_NAMES = {
        "calc_factor_single_stock",
        "calc_factor_series",
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

    # ---- 模板缓存（L1 内存 + L2 磁盘） ----
    _TEMPLATE_CACHE_DIR = Path(__file__).resolve().parent / "_template_cache"
    _TEMPLATE_CACHE: dict = {}  # L1: { cache_key → base_template }

    @staticmethod
    def _template_cache_key(template: str, lookback_days: int, cols_def: str) -> str:
        """基于模板内容+参数算 hash，保证不同进程 key 一致且可缓存到磁盘。"""
        raw = template + "\x00" + str(lookback_days) + "\x00" + cols_def
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_factor_code(template: str, code: str, lookback_days: int, load_cols: list = None) -> str:
        """安全填充模板：用 replace 替代 format，避免 {xxx} 被错误解释。

        缓存机制（L1 内存 + L2 磁盘）：
        相同 template + lookback + load_cols 的 base 只编译一次，
        后续只替换 {user_code}。子进程之间共享磁盘缓存。
        """
        if load_cols:
            # 过滤掉索引列（datetime 是 MultiIndex 的一部分，不是数据列）
            filtered = [c for c in load_cols if c not in ("datetime", "instrument")]
            cols_def = f"_LOAD_COLS = {filtered}  # LLM推断"
        else:
            cols_def = "_LOAD_COLS = None  # 加载全部列"

        cache_key = FactorFBWorkspace._template_cache_key(template, lookback_days, cols_def)

        # L1: 内存缓存
        base = FactorFBWorkspace._TEMPLATE_CACHE.get(cache_key)
        if base is not None:
            return base.replace('{user_code}', code)

        # L2: 磁盘缓存
        FactorFBWorkspace._TEMPLATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = FactorFBWorkspace._TEMPLATE_CACHE_DIR / cache_key
        if cache_file.exists():
            try:
                base = pickle.loads(cache_file.read_bytes())
                FactorFBWorkspace._TEMPLATE_CACHE[cache_key] = base
                return base.replace('{user_code}', code)
            except Exception:
                pass  # 损坏的缓存，重新编译

        # 编译 base（含列定义、LOOKBACK、双花括号解义）
        base = (template
                .replace('{_LOAD_COLS_DEF}', cols_def)
                .replace('{_LOAD_MINUTE_STOCK}', _LOAD_MINUTE_STOCK_SRC)
                .replace('{lookback_days}', str(lookback_days)))
        if '{{' in base:
            base = base.replace('{{', '{').replace('}}', '}')

        # 写入 L1 + L2
        FactorFBWorkspace._TEMPLATE_CACHE[cache_key] = base
        try:
            cache_file.write_bytes(pickle.dumps(base))
        except Exception:
            pass  # 磁盘缓存非关键，静默失败

        return base.replace('{user_code}', code)

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
