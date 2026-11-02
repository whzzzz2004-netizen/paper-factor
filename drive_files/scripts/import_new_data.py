#!/usr/bin/env python3
"""
导入「原始数据/」→「数据仓库/」的增量导入脚本（可读精简版）。

每个环节输出一行摘要：处理了什么时间范围、新增了哪个因子；隐藏中间过程。

用法（Windows PowerShell 直接运行）:
  python import_new_data.py --check                 # 扫描预览
  python import_new_data.py --dry-run             # 预览将导入内容
  python import_new_data.py                       # 自动增量导入
  python import_new_data.py --update-prompts-only # 用 schema 刷新因子提示词
  python import_new_data.py --summary             # 打印总结

数据域:
  1. 日线行情  dailyData.parquet  → 全量/测试行情 per-stock（按 daily_market 增量；
     新股票导入全部历史，新列回填全部历史）
  2. 分钟线    YYYYMMDD.parquet    → 全量/测试 minute_by_date（已存在则只合并新列；
     新日期写完整 DataFrame）
  3. 截面因子  非行情/日线/*.parquet → 全量非行情 per-stock（自动对齐交易日；新因子全量/旧增量）
  4. 分钟截面 非行情/分钟线/*.parquet → 合并进分钟 by_date（逐日防 OOM；遇到缺失基础日期暂停）
  5. Barra 文件（文件名含特征）         → 复制到 barra_model/
  6. 新列自动注册 schema.json + factor_field_schema.json + 测试数据补列

实现约定:
  - 所有 parquet/json 写入都是「临时文件 + os.replace」的原子写；
  - 日频索引统一归一到零点并去重，分钟索引统一到 datetime64[ns]；
  - 日线和分钟交易日历分别重建，绝不把分钟日期并进日线日历。
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ═══════════════════════════════════════════════════════════════════════
# 1. 路径与常量
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
def _default_data_root() -> Path:
    env = os.environ.get("PAPER_FACTOR_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        candidates = (Path(r"G:\paper-factor-data"), Path(r"D:\paper-factor-data"))
    else:
        candidates = (Path("/mnt/g/paper-factor-data"), Path("/mnt/d/paper-factor-data"))
    for candidate in candidates:
        if (candidate / "数据仓库").is_dir() or (candidate / "schema.json").is_file():
            return candidate
    return candidates[-1]


DATA_ROOT = _default_data_root()
WH = "数据仓库"

# 项目仓库（prompts.yaml 所属）：环境变量 > repo_path.txt > 数据根
_REPO_ENV = os.environ.get("PAPER_FACTOR_REPO", "").strip()
if _REPO_ENV:
    PROJECT_REPO = Path(_REPO_ENV)
else:
    _rt = DATA_ROOT / "repo_path.txt"
    _v = _rt.read_text(encoding="utf-8").strip() if _rt.exists() else ""
    PROJECT_REPO = Path(_v) if _v else PROJECT_ROOT

# 数据仓库目录
FULL_MARKET_DAILY = DATA_ROOT / WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily"
TEST_MARKET_DAILY = DATA_ROOT / WH / "行情数据" / "日线" / "测试" / "stock_data" / "daily"
FULL_FUND_DAILY = DATA_ROOT / WH / "非行情数据" / "全量" / "stock_data" / "daily"
TEST_FUND_DAILY = DATA_ROOT / WH / "非行情数据" / "测试" / "stock_data" / "daily"
FULL_MINUTE = DATA_ROOT / WH / "行情数据" / "分钟线" / "全量" / "stock_data"
TEST_MINUTE = DATA_ROOT / WH / "行情数据" / "分钟线" / "测试" / "stock_data"
FULL_MINUTE_BY_DATE = FULL_MINUTE / "minute_by_date"
TEST_MINUTE_BY_DATE = TEST_MINUTE / "minute_by_date"
BARRA_DIR = DATA_ROOT / WH / "barra_model"

# 元数据文件
FULL_TRADE_DATES_FILE = FULL_MARKET_DAILY / "trade_dates.json"
FULL_STOCK_LIST_FILE = FULL_MARKET_DAILY / "stock_list.json"
TEST_TRADE_DATES_FILE = TEST_MARKET_DAILY / "trade_dates.json"
TEST_STOCK_LIST_FILE = TEST_MARKET_DAILY / "stock_list.json"
FULL_MINUTE_TRADE_DATES = FULL_MINUTE / "trade_dates.json"
FULL_MINUTE_STOCK_LIST = FULL_MINUTE / "stock_list.json"
MINUTE_SUBDIR_TRADE_DATES = FULL_MINUTE_BY_DATE / "trade_dates.json"
MINUTE_SUBDIR_STOCK_LIST = FULL_MINUTE_BY_DATE / "stock_list.json"

SCHEMA_FILE = PROJECT_ROOT / "schema.json"
FACTOR_FIELD_SCHEMA_PATHS = [
    DATA_ROOT / WH / "行情数据" / "日线" / "全量" / "factor_field_schema.json",
    DATA_ROOT / WH / "行情数据" / "日线" / "测试" / "factor_field_schema.json",
]
NEW_DATA_DIR = DATA_ROOT / "原始数据"
DESC_FILE = NEW_DATA_DIR / "新因子描述.csv"
IMPORT_STATE_FILE = DATA_ROOT / WH / "import_state.json"
DAILY_MTIME_FILE = DATA_ROOT / WH / ".last_daily_mtime"

# 列名候选清单（用于识别 symbol / 日期列）
SYMBOL_COL_CANDIDATES = ["symbol", "stock_code", "code", "instrument", "wind_code", "ts_code",
                         "sec_code", "股票代码", "代码", "证券代码"]
DATE_COL_CANDIDATES = ["date", "trade_date", "tradeDate", "datetime", "time", "交易日期", "日期"]
MINUTE_EXPECTED_COLS = {"open", "high", "low", "close", "volume", "return", "factor", "vwap"}
# 新建分钟 by-date 文件时至少要有这些列，否则说明源文件只带了新列、缺少基础行情。
MINUTE_CORE_COLS = {"open", "high", "low", "close", "volume"}
INTERNAL_COLS = {"__index_level_0__", "Unnamed: 0", "level_0", "index"}
BARRA_KEYWORDS = ["因子收益率表", "因子暴露表", "特质收益率表", "特质风险表", "风险因子协方差矩阵表"]
# 截面因子一次合并的因子数上限。避免把全量宽表一次性 stack 成超大长表。
FACTOR_BATCH_SIZE = 4


# ═══════════════════════════════════════════════════════════════════════
# 2. 通用工具
# ═══════════════════════════════════════════════════════════════════════
def log(msg: str):
    print(msg, flush=True)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _temp_path(path: Path) -> Path:
    """同目录临时文件；带 pid/线程号，避免并行写冲突。"""
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def _replace_file(tmp: Path, dst: Path):
    os.replace(tmp, dst)


def write_text_atomic(path: Path, text: str):
    """原子写文本：先写临时文件，再 replace，避免半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        _replace_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Path, obj):
    """原子写 JSON：先写临时文件，再 replace，避免半截文件。"""
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_parquet_atomic(df: pd.DataFrame, path: Path, **kwargs):
    """原子写 parquet：写临时文件后替换，崩溃不会破坏原文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    try:
        df.to_parquet(tmp, **kwargs)
        _replace_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_table_atomic(table: pa.Table, path: Path, **kwargs):
    """原子写 Arrow Table（保留 MultiIndex 等元数据）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    try:
        pq.write_table(table, tmp, **kwargs)
        _replace_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def parquet_columns(path: Path) -> list[str]:
    """读 parquet schema 的列名；文件不存在/损坏时返回空列表。"""
    try:
        return list(pq.read_schema(path).names)
    except Exception:
        return []


def max_date_in_file(path: Path, columns: list) -> Optional[pd.Timestamp]:
    """只读日期列，取源文件最大日期（用于补旧 state，不加载整份数据）。"""
    date_col = next((c for c in DATE_COL_CANDIDATES if c in columns), None)
    try:
        if date_col:
            series = pd.read_parquet(path, columns=[date_col])[date_col]
            value = parse_dates(series).max()
        else:
            index = pd.read_parquet(path, columns=[]).index
            value = pd.DatetimeIndex(index).max()
    except Exception:
        return None
    return pd.Timestamp(value) if pd.notna(value) else None


def normalize_daily_frame(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """日频数据统一：去 NaT、归一到零点、按日期排序、去重复日期。

    日频 per-stock 的索引必须只有日期含义。若源文件带 15:00 时间戳，
    直接合并会和仓库里的 00:00 变成同一天两行，这里统一归零。
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index)
    out = out[~pd.isna(out.index)]
    out.index = out.index.normalize()
    out = out.sort_index()
    if out.index.has_duplicates:
        n = int(out.index.duplicated(keep="last").sum())
        out = out[~out.index.duplicated(keep="last")]
        log(f"  ⚠️ {context}: 去掉 {n} 行重复日期（保留最后一条）")
    return out


def normalize_minute_index(df: pd.DataFrame) -> pd.DataFrame:
    """统一分钟索引：instrument 为 6 位字符串，datetime 为 datetime64[ns]。

    parquet 读回来的索引时间单位可能是 us/ms；和内存里的 ns 直接 reindex
    会产生整列 NaN，所以合并前必须先统一。
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    inst = pd.Index([normalize_code(x) for x in out.index.get_level_values("instrument")])
    dt = pd.DatetimeIndex(out.index.get_level_values("datetime")).as_unit("ns")
    out.index = pd.MultiIndex.from_arrays([inst, dt], names=["instrument", "datetime"])
    return out


def dedupe_minute_frame(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """分钟数据按 (instrument, datetime) 去重并排序。"""
    if df is None or len(df) == 0:
        return df
    out = normalize_minute_index(df)
    out = out.sort_index()
    if out.index.has_duplicates:
        n = int(out.index.duplicated(keep="last").sum())
        out = out[~out.index.duplicated(keep="last")]
        log(f"  ⚠️ {context}: 去掉 {n} 行重复分钟索引（保留最后一条）")
    return out


def load_state() -> dict:
    d = load_json(IMPORT_STATE_FILE)
    return d if isinstance(d, dict) else {}


def save_state(state: dict):
    write_json(IMPORT_STATE_FILE, state)


def load_schema() -> dict:
    d = load_json(SCHEMA_FILE)
    return d if isinstance(d, dict) else {}


def save_schema(s: dict):
    write_json(SCHEMA_FILE, s)


def load_json_list(path: Path) -> list:
    d = load_json(path)
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        return list(d.keys())
    return []


def write_json_list(path: Path, items: list):
    """原子写日期/股票列表，保持原有紧凑 JSON 格式。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    try:
        tmp.write_text(json.dumps(sorted(set(items))), encoding="utf-8")
        _replace_file(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_descriptions() -> dict:
    """解析 新因子描述.csv → {因子名: 描述}。兼容 2/3 列与多编码。"""
    if not DESC_FILE.exists():
        return {}
    raw = DESC_FILE.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return {}
    descs = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if len(row) >= 3:
            desc = row[2].strip()
            for k in (Path(name).stem, row[1].strip()):
                if k:
                    descs[k] = desc
        else:
            desc = row[1].strip() if len(row) > 1 else ""
            descs[name] = desc
    return descs


def parse_dates(s: pd.Series) -> pd.Series:
    """任意日期 Series → datetime Series（兼容字符串/整数/浮点，YYYYMMDD 与 YYYY-MM-DD）。"""
    if s.dtype.kind in "iuf":
        s = s.apply(lambda v: str(int(v)) if pd.notna(v) and float(v) == int(float(v)) else np.nan)
    try:
        s = pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        s = pd.to_datetime(s, errors="coerce")
    if s.isna().all():
        s = pd.to_datetime(s.astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    return s


_RE_D1 = re.compile(r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})")
_RE_D8 = re.compile(r"(\d{8})")


def date_from_filename(name: str) -> Optional[pd.Timestamp]:
    m = _RE_D1.search(name)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except ValueError:
            pass
    m = _RE_D8.search(name)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)[:4]), month=int(m.group(1)[4:6]), day=int(m.group(1)[6:8]))
        except ValueError:
            pass
    return None


def normalize_code(c) -> str:
    """股票代码 → 6 位字符串（去交易所前后缀，补零）。"""
    if isinstance(c, float) and np.isnan(c):
        return ""
    s = str(c).strip().lower()
    for pfx in ("sh", "sz", "bj"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    for sfx in (".xshg", ".xshe", ".xin9", ".sh", ".sz", ".bj"):
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break
    s = s.split(".")[0]
    if not s:
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return str(int(s)).zfill(6)
    except ValueError:
        return s


def dstr(t) -> str:
    return str(pd.Timestamp(t))[:10]


def range_str(ts) -> str:
    """日期索引/列表 → 'YYYY-MM-DD ~ YYYY-MM-DD'（单日则 一天）。"""
    if ts is None:
        return "-"
    if hasattr(ts, "min"):
        _min, _max = ts.min(), ts.max()
    else:
        lst = list(ts)
        if not lst:
            return "-"
        _min, _max = min(lst), max(lst)
    if _min == _max:
        return dstr(_min)
    return f"{dstr(_min)} ~ {dstr(_max)}"


def subdir_hint(path: Path) -> Optional[str]:
    """按子目录给类型提示。"""
    try:
        rel = path.relative_to(NEW_DATA_DIR).parts
    except ValueError:
        return None
    if len(rel) >= 2:
        if rel[0] == "日线":
            return "market"
        if rel[0] == "非行情":
            if len(rel) >= 3 and rel[1] == "分钟线":
                return "fundamental_minute"
            if len(rel) >= 3 and rel[1] == "日线":
                return "fundamental_daily"
            return "fundamental"
        if rel[0] == "分钟线":
            return "minute"
        if rel[0] == "barra":
            return "barra"
    return None


# ═══════════════════════════════════════════════════════════════════════
# 3. 文件检测
# ═══════════════════════════════════════════════════════════════════════
def detect_file_type(path: Path, hint: Optional[str], columns: list) -> str:
    """判定文件类型: barra_model / daily_data / minute_by_date /
    cross_sectional_factor / minute_cross_sectional_factor / standard。

    规则：
    - Barra 模型：在 barra/ 子目录，或文件名含 BARRA_KEYWORDS
    - 日线行情：文件名 == dailyData.parquet
    - 分钟 per-date：在 分钟线/ 且含 instrument/symbol 列
    - 截面因子（日频/分钟）：非行情/ 下无 symbol 列，唯一日期列名为
      datetime 或 trade_date，且剩余列全是股票代码（纯数字）→ 宽表因子
    """
    if hint == "barra" or any(k in path.name for k in BARRA_KEYWORDS):
        return "barra_model"
    if path.name.lower() == "dailydata.parquet":
        return "daily_data"
    if hint == "minute" and ("instrument" in columns or "symbol" in columns):
        return "minute_by_date"
    if hint in ("fundamental", "fundamental_daily", "fundamental_minute"):
        if not any(c in SYMBOL_COL_CANDIDATES for c in columns):
            date_like = [c for c in columns if c in DATE_COL_CANDIDATES]
            only_date_ok = (len(date_like) == 1 and date_like[0] in ("datetime", "trade_date")) or len(date_like) == 0
            if only_date_ok:
                non_std = [c for c in columns if c not in SYMBOL_COL_CANDIDATES
                           and c not in DATE_COL_CANDIDATES and c not in INTERNAL_COLS]
                num = [c for c in non_std if re.match(r"^\d+$", str(c))]
                if num and len(num) == len(non_std):
                    return "minute_cross_sectional_factor" if hint == "fundamental_minute" else "cross_sectional_factor"
    return "standard"


def scan_new_data_dir() -> list:
    """扫描 原始数据/ 的 parquet/csv → [(path, hint)]。"""
    if not NEW_DATA_DIR.exists():
        return []
    out = []
    for p in sorted(NEW_DATA_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".parquet", ".csv") and p.resolve() != DESC_FILE.resolve():
            out.append((p, subdir_hint(p)))
    return out


def read_header(path: Path):
    if path.suffix.lower() == ".csv":
        df0 = pd.read_csv(path, nrows=0)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            n = max(sum(1 for _ in f) - 1, 0)
        return list(df0.columns), n
    pf = pq.ParquetFile(path)
    return pf.schema_arrow.names, pf.metadata.num_rows


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


class FileInfo:
    def __init__(self, path: Path, hint: Optional[str]):
        self.path = path
        self.hint = hint
        self.file_type = "standard"
        self.error = None
        self.columns = []
        self.row_count = 0
        self.factor_name = None


@dataclass
class DomainResult:
    """一个数据域的导入结果，供 do_import 汇总注册 schema / 补测试列。"""

    message: str = ""
    daily_columns: set[str] = field(default_factory=set)
    minute_columns: set[str] = field(default_factory=set)
    updates: dict = field(default_factory=dict)
    processed: bool = False


def inspect_file(path: Path, hint: Optional[str]) -> FileInfo:
    """只读 header 做文件分拣，不加载数据内容。"""
    info = FileInfo(path, hint)
    if hint == "barra" or any(k in path.name for k in BARRA_KEYWORDS):
        info.file_type = "barra_model"
        return info
    try:
        columns, nrows = read_header(path)
        info.columns = columns
        info.row_count = nrows
        info.file_type = detect_file_type(path, hint, columns)
        if info.file_type in ("cross_sectional_factor", "minute_cross_sectional_factor"):
            info.factor_name = path.stem
    except Exception as e:
        info.error = str(e)
    return info


# ═══════════════════════════════════════════════════════════════════════
# 4. 写入助手
# ═══════════════════════════════════════════════════════════════════════
def merge_stock(dst: Path, df: pd.DataFrame):
    """合并单只股票到 per-stock（按日期对齐，新值优先，保留其他列）。

    关键：只更新 df 里出现的列。旧实现用 concat + duplicated(keep="last")，
    新批次只带自己的列，会把其他批次的列在重叠日期整行覆盖成 NaN（已实测）。

    索引统一归一到零点并去重；写入走临时文件 + replace，避免写坏原文件。
    """
    df = normalize_daily_frame(df, f"merge {dst.stem}")
    if dst.exists():
        old = normalize_daily_frame(pd.read_parquet(dst), f"existing {dst.stem}")
        combined = old.reindex(old.index.union(df.index))
        for col in df.columns:
            if col in combined.columns:
                # 新值非空则覆盖旧值；新值为空时保留旧值
                combined[col] = df[col].combine_first(combined[col])
            else:
                combined[col] = df[col]
        combined = combined.sort_index()
        write_table_atomic(pa.Table.from_pandas(combined), dst)
    else:
        write_table_atomic(pa.Table.from_pandas(df), dst)


def parallel_merge(items: list) -> int:
    """并行执行多只股票的 merge_stock，返回成功写入的股票数。

    items: [(目标 per-stock 路径, 该股票的 DataFrame), ...]
    线程数 = min(16, CPU核数, 任务数)，避免磁盘竞争。
    """
    if not items:
        return 0
    n_threads = min(16, os.cpu_count() or 4, len(items))
    done = 0
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        fs = [pool.submit(merge_stock, dst, df) for dst, df in items]
        for f in as_completed(fs):
            f.result()
            done += 1
    return done


def copy_new_col_to_test(full_dir: Path, test_dir: Path, col: str,
                         test_stocks: list) -> tuple[int, int]:
    """把新增列从「全量 per-stock」同步到「测试 per-stock」。

    只补列，不动日期范围/股票池（测试数据固定 300×300）。
    全量/测试索引都先归一到零点，避免 00:00 与 15:00 对不上导致整列 NaN。

    返回 (补列股票数, 其中该列至少有一个非空值的股票数)。
    第二个数用于发现“列补进去了但日期完全对不上”的静默全 NaN 情况。
    """
    n = 0
    n_with_data = 0
    for stock in test_stocks:
        full_f, test_f = full_dir / f"{stock}.parquet", test_dir / f"{stock}.parquet"
        if not full_f.exists() or not test_f.exists():
            continue
        try:
            full_col = pd.read_parquet(full_f, columns=[col])
        except Exception:
            continue
        if col not in full_col.columns:
            continue
        series = full_col[col]
        series.index = pd.DatetimeIndex(series.index).normalize()
        series = series[~series.index.duplicated(keep="last")]

        test = pd.read_parquet(test_f)
        test.index = pd.DatetimeIndex(test.index).normalize()
        test = test[~test.index.duplicated(keep="last")]
        test[col] = series.reindex(test.index)
        write_table_atomic(pa.Table.from_pandas(test), test_f)
        n += 1
        if test[col].notna().any():
            n_with_data += 1
    return n, n_with_data


# ═══════════════════════════════════════════════════════════════════════
# 5. 各数据域导入
# ═══════════════════════════════════════════════════════════════════════
def import_barra(info: FileInfo) -> str:
    """复制 Barra 风险模型文件到 barra_model/。返回一行摘要。"""
    BARRA_DIR.mkdir(parents=True, exist_ok=True)
    dst = BARRA_DIR / info.path.name
    shutil.copy2(info.path, dst)
    return f"Barra：{info.path.name} → barra_model/（{info.path.stat().st_size/1024/1024:.1f} MB）"


def import_daily_data(info: FileInfo, state: dict) -> DomainResult:
    """导入 dailyData.parquet 的新日期段 → 全量行情 per-stock。

    增量边界：
    - 已有 per-stock 文件的股票：只处理 state['daily_market'] 之后的日期；
    - **新股票**：没有 per-stock 文件，导入源文件里的全部历史（不受全局 cutoff 截断）。

    返回本批新列，供 do_import 注册 schema 并补到测试行情。
    """
    pdf = load_file(info.path)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in pdf.columns), None)
    if date_col is not None:
        pdf.index = pd.DatetimeIndex(parse_dates(pdf[date_col]), name="datetime")
        pdf = pdf.drop(columns=[date_col])
    else:
        if not isinstance(pdf.index, pd.DatetimeIndex):
            return DomainResult(message="日线行情 dailyData：缺少日期列，且索引不是日期")
        pdf.index = pdf.index.rename("datetime")

    pdf = pdf[~pd.isna(pdf.index)]
    pdf.index = pdf.index.normalize()
    if len(pdf) == 0:
        return DomainResult(message="日线行情 dailyData：无有效日期")

    symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in pdf.columns), None)
    if symbol_col:
        codes = pdf[symbol_col].apply(normalize_code)
    else:
        codes = pd.Series([normalize_code(info.path.stem)] * len(pdf), index=pdf.index)

    # 空/非法代码不写文件（否则会生成 ".parquet"）。
    valid = codes.ne("")
    if not valid.all():
        log(f"  ⚠️ 日线行情：跳过 {(~valid).sum()} 行空/非法股票代码")
        pdf = pdf[valid]
        codes = codes[valid]
    if len(pdf) == 0:
        return DomainResult(message="日线行情 dailyData：没有有效股票代码")

    market_files = sorted(FULL_MARKET_DAILY.glob("*.parquet"))
    existing_cols = set(parquet_columns(market_files[0])) if market_files else set()
    ignored_cols = set(INTERNAL_COLS) | ({symbol_col} if symbol_col else set())
    new_cols = set(pdf.columns) - existing_cols - ignored_cols

    drop_cols = [c for c in pdf.columns if c in ignored_cols]
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols)

    cutoff = state.get("daily_market")
    cutoff_ts = pd.Timestamp(cutoff) if cutoff else None

    items = []
    new_stocks = set()
    new_col_list = sorted(new_cols)
    # 用位置赋值，避免 dailyData 里同一天多只股票（重复索引）时 groupby 对齐报错。
    pdf = pdf.copy()
    pdf["__code"] = codes.to_numpy()
    for stock, grp in pdf.groupby("__code", sort=True):
        grp = grp.drop(columns=["__code"])
        grp = normalize_daily_frame(grp, f"日线 {stock}")
        if grp.empty:
            continue
        dst = FULL_MARKET_DAILY / f"{stock}.parquet"
        is_new_stock = not dst.exists()

        if is_new_stock:
            sub = grp
        elif new_col_list:
            # 新列需要回填全部历史；已有列只取 cutoff 之后的新日期。
            new_slice = grp[[c for c in new_col_list if c in grp.columns]]
            old_slice = grp.drop(columns=[c for c in new_col_list if c in grp.columns])
            if cutoff_ts is not None:
                old_slice = old_slice[old_slice.index > cutoff_ts]
            sub = pd.concat([old_slice, new_slice], axis=1)
        elif cutoff_ts is not None:
            sub = grp[grp.index > cutoff_ts]
        else:
            sub = grp

        if sub.empty:
            continue
        if is_new_stock:
            new_stocks.add(stock)
        items.append((dst, sub))

    if not items:
        return DomainResult(message=f"日线行情 dailyData：无新数据（现有到 {cutoff or '-'}）")

    n_write = parallel_merge(items)
    written_min = min(sub.index.min() for _, sub in items)
    written_max = max(sub.index.max() for _, sub in items)
    written_rows = sum(len(sub) for _, sub in items)

    # 只前进不后退；新股票的老历史不应把状态拉回去。
    max_date = pdf.index.max()
    if cutoff_ts is None or max_date > cutoff_ts:
        state["daily_market"] = dstr(max_date)

    if new_stocks:
        log(f"  ℹ️ 日线行情：发现 {len(new_stocks)} 只新股票（已导入其全部历史）")

    return DomainResult(
        message=(f"日线行情 dailyData：写入 {dstr(written_min)} ~ {dstr(written_max)}"
                 f"（{written_rows} 行, {len(items)} 只股票写入 {n_write}）"),
        daily_columns=new_cols,
        processed=n_write > 0,
    )


def prepare_minute_frame(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """把分钟源文件整理成 MultiIndex [instrument, datetime]。

    支持两种输入：
    - 已经是 MultiIndex(instrument, datetime)
    - 扁平表：symbol/instrument + datetime/trade_date/time

    统一做：代码补零、时间转 datetime64[ns]、按 (股票, 时间) 去重。
    """
    df = df.copy()

    if "symbol" in df.columns or "instrument" in df.columns:
        code_col = "instrument" if "instrument" in df.columns else "symbol"
        inst = df[code_col].apply(normalize_code)
        df = df.drop(columns=[code_col])

        date_col = next((c for c in ("datetime", "trade_date", "date", "time") if c in df.columns), None)
        if date_col is None:
            raise ValueError(f"分钟文件缺少 datetime/trade_date/time 列: {path.name}")

        if date_col == "time":
            # time 可能只有时分秒；用文件名日期补齐。若它本身是完整 datetime，直接解析。
            delta = pd.to_timedelta(df["time"].astype(str), errors="coerce")
            if delta.notna().any():
                fdate = date_from_filename(path.name)
                if fdate is None:
                    raise ValueError(f"time 列只有时分秒，但文件名没有日期: {path.name}")
                df["datetime"] = fdate.normalize() + delta
            else:
                df["datetime"] = parse_dates(df["time"])
        else:
            df["datetime"] = parse_dates(df[date_col])

        dt = df.pop("datetime")
        # 丢弃所有日期列（真实原始文件常同时带 trade_date 和 date），
        # 以及可能同时存在的 symbol/instrument，避免它们被当成新因子列。
        date_like = [c for c in df.columns if c in DATE_COL_CANDIDATES]
        if date_like:
            df = df.drop(columns=date_like)
        for c in ("symbol", "instrument"):
            if c in df.columns:
                df = df.drop(columns=[c])
        df.index = pd.MultiIndex.from_arrays([inst, dt],
                                             names=["instrument", "datetime"])
    elif isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if "instrument" not in names or "datetime" not in names:
            raise ValueError(f"分钟 MultiIndex 必须有 instrument/datetime: {path.name}")
        inst = pd.Index([normalize_code(x) for x in df.index.get_level_values("instrument")])
        dt = pd.DatetimeIndex(parse_dates(pd.Series(df.index.get_level_values("datetime"))))
        df.index = pd.MultiIndex.from_arrays([inst, dt], names=["instrument", "datetime"])
    else:
        raise ValueError(f"分钟文件既没有 symbol/instrument 列，也不是 MultiIndex: {path.name}")

    df = df[~pd.isna(df.index.get_level_values("datetime"))]
    df = df[df.index.get_level_values("instrument") != ""]
    df = dedupe_minute_frame(df, f"分钟 {path.name}")
    return df


def import_minute_file(info: FileInfo, minute_dates: set, minute_stocks: set, state: dict) -> DomainResult:
    """导入一个分钟文件（可含多个日期）。

    规则：
    - 按文件内部的真实日期分组，不依赖文件名日期；
    - 已存在的 by-date 文件：只合并“新列”，不覆盖基础行情；
    - 不存在的 by-date 文件：源文件必须有 open/high/low/close/volume，
      写完整 DataFrame（基础列 + 新列），绝不只写新列；
    - 测试数据只合并新列，不新增日期。
    """
    fdate = date_from_filename(info.path.name)
    try:
        df = prepare_minute_frame(load_file(info.path), info.path)
    except Exception as e:
        return DomainResult(message=f"分钟 {info.path.name}: 跳过（{e}）")
    if df.empty:
        return DomainResult(message=f"分钟 {info.path.name}: 无有效数据")

    new_cols = [c for c in df.columns if c not in MINUTE_EXPECTED_COLS]
    actual_dates = sorted({pd.Timestamp(x).normalize()
                           for x in df.index.get_level_values("datetime")})
    if fdate is not None and actual_dates and fdate.normalize() not in actual_dates:
        log(f"  ⚠️ 分钟 {info.path.name}: 文件名日期 {dstr(fdate)} 与内部日期不一致，按内部日期导入")

    processed_dates = []
    processed_stocks = set()
    mm = state.get("minute_market")
    minute_cutoff = pd.Timestamp(mm).normalize() if mm else None

    for date in actual_dates:
        d = date.strftime("%Y%m%d")
        mask = df.index.get_level_values("datetime").normalize() == date
        day = df[mask]
        if day.empty:
            continue
        dst = FULL_MINUTE_BY_DATE / f"{d}.parquet"

        if dst.exists():
            minute_dates.add(d)
            if not new_cols:
                continue
            existing = dedupe_minute_frame(pd.read_parquet(dst), f"分钟 {d}")
            for col in new_cols:
                existing[col] = (existing[col].combine_first(day[col])
                                 if col in existing.columns else day[col])
            write_parquet_atomic(existing, dst, compression="lz4")
        else:
            if minute_cutoff is not None and date <= minute_cutoff:
                log(f"  ⚠️ 分钟 {d}: 缺少基础 by-date 文件且在 minute_market 之前，跳过")
                continue
            if not MINUTE_CORE_COLS.issubset(day.columns):
                missing = sorted(MINUTE_CORE_COLS - set(day.columns))
                log(f"  ⚠️ 分钟 {d}: 缺少基础列 {missing}，无法新建 by-date 文件，跳过")
                continue
            write_parquet_atomic(day, dst, compression="lz4")

        processed_dates.append(d)
        minute_dates.add(d)
        processed_stocks.update(day.index.get_level_values("instrument").unique())

        # 测试数据：只补新列，不新增日期/股票。
        test_path = TEST_MINUTE_BY_DATE / f"{d}.parquet"
        if new_cols and test_path.exists():
            te = dedupe_minute_frame(pd.read_parquet(test_path), f"测试分钟 {d}")
            for col in new_cols:
                te[col] = (te[col].combine_first(day[col])
                           if col in te.columns else day[col])
            write_parquet_atomic(te, test_path)

    minute_stocks.update(processed_stocks)
    if not processed_dates:
        return DomainResult(message=f"分钟 {info.path.name}: 无新日期或新列")
    msg = f"分钟 {info.path.name}: 写入 {len(processed_dates)} 天"
    if new_cols:
        msg += f"，新列 {new_cols}"
    return DomainResult(message=msg, minute_columns=set(new_cols), processed=True)


def update_minute_meta(minute_dates: set, minute_stocks: set):
    """把分钟 trade_dates.json / stock_list.json 更新为并集（有变化才写）。
    分钟数据在 minute_by_date 子目录还有一份同步副本（模板从子目录读）。"""
    old_dates = set(load_json_list(FULL_MINUTE_TRADE_DATES))
    if sorted(minute_dates) != sorted(old_dates):
        write_json_list(FULL_MINUTE_TRADE_DATES, sorted(minute_dates))
        write_json_list(MINUTE_SUBDIR_TRADE_DATES, sorted(minute_dates))
        log(f"  ℹ️ 分钟 trade_dates.json: {len(old_dates)} → {len(minute_dates)} 天")
    old_stocks = set(load_json_list(FULL_MINUTE_STOCK_LIST))
    if sorted(minute_stocks) != sorted(old_stocks):
        write_json_list(FULL_MINUTE_STOCK_LIST, sorted(minute_stocks))
        write_json_list(MINUTE_SUBDIR_STOCK_LIST, sorted(minute_stocks))
        log(f"  ℹ️ 分钟 stock_list.json: {len(old_stocks)} → {len(minute_stocks)} 只")


def rebuild_day_trade_dates() -> list:
    """重建「权威交易日历」= 日线行情 per-stock parquet 实际日期的并集。

    不依赖 trade_dates.json / 不并入因子源日期（因子可能含周末/非交易日）。
    返回 'YYYYMMDD' 排序列表；只重建全量，测试数据固定 300 天不动。
    """
    _files = sorted(FULL_MARKET_DAILY.glob("*.parquet")) if FULL_MARKET_DAILY.exists() else []
    if not _files:
        log("  ⚠️ 日线行情 per-stock parquet 不存在，无法重建日线日历")
        return []
    _dates = set()
    for _f in _files:
        try:
            _df = pd.read_parquet(_f, columns=[])
            if _df.index is not None:
                _dates.update(pd.DatetimeIndex(_df.index).strftime("%Y%m%d").unique().tolist())
        except Exception:
            continue
    _dates = sorted(_dates)
    if not _dates:
        log("  ⚠️ 日线 per-stock 文件都没有有效日期，无法重建日线日历")
        return []
    write_json_list(FULL_TRADE_DATES_FILE, _dates)
    log(f"  ℹ️ 日线权威日历: {dstr(_dates[0])} ~ {dstr(_dates[-1])} 共 {len(_dates)} 天")
    return _dates


def rebuild_minute_trade_dates() -> list:
    """权威分钟交易日 = minute_by_date/*.parquet 文件名（YYYYMMDD）并集。

    同写两份同步副本（stock_data/ 与 stock_data/minute_by_date/）。
    返回 'YYYYMMDD' 排序列表。
    """
    import re as _re
    _dates = set()
    _files = sorted(FULL_MINUTE_BY_DATE.glob("*.parquet")) if FULL_MINUTE_BY_DATE.exists() else []
    for _f in _files:
        _m = _re.fullmatch(r"(\d{8})\.parquet", _f.name)
        if _m:
            _dates.add(_m.group(1))
    _dates = sorted(_dates)
    if not _dates:
        log("  ⚠️ 分钟线 per-date parquet 为空，无法重建分钟日历")
        return []
    write_json_list(FULL_MINUTE_TRADE_DATES, _dates)
    write_json_list(MINUTE_SUBDIR_TRADE_DATES, _dates)
    log(f"  ℹ️ 分钟 trade_dates: {dstr(_dates[0])} ~ {dstr(_dates[-1])} 共 {len(_dates)} 天")
    return _dates


def rebuild_stock_list() -> list:
    """从全量行情 per-stock 文件重建股票池（新股票不会漏）。"""
    codes = sorted(p.stem for p in FULL_MARKET_DAILY.glob("*.parquet") if p.stem.isdigit())
    write_json_list(FULL_STOCK_LIST_FILE, codes)
    log(f"  ℹ️ 全量股票池: {len(codes)} 只")
    return codes


def rebuild_authoritative_trade_dates() -> list:
    """**分别**重建日线和分钟交易日历，不做并集写回。

    日线日历只由日线 per-stock 实际日期决定；分钟日历只由 minute_by_date
    文件名决定。两者日期不一致时（例如当天分钟已到、日线还没到），绝不能
    把分钟日期写进日线 trade_dates.json，否则 /all 会跑到没有日线的日期上。
    """
    day = rebuild_day_trade_dates()
    minute = rebuild_minute_trade_dates()
    union = sorted(set(day) | set(minute))
    if not union:
        log("  ❌ 日线与分钟行情都没有实际数据，无法重建日历")
        return []
    rebuild_stock_list()
    log(f"  ✅ 日历分别重建：日线 {len(day)} 天，分钟 {len(minute)} 天"
        f"（不做并集写回）")
    return union


def import_cross_sectional_factors(infos: list, descs: dict, state: dict) -> DomainResult:
    """导入日频截面因子（非行情/日线/*.parquet 宽表）→ 非行情全量 per-stock。

    每个因子独立按自己的日期更新（不被 trade_dates 框死）：
    - 全新因子 → 全量导入该因子源的所有日期
    - 已有因子 → 只处理 state['daily_factors'][因子名] 之后的数据
    - 因子按 FACTOR_BATCH_SIZE 分批 stack/合并，避免一次性构造超大长表

    索引统一归一零点并去重；返回本批涉及的日线列，供 schema 注册。
    """
    if not infos:
        return DomainResult()
    log(f"【截面因子】共 {len(infos)} 个")
    daily_factors = state.get("daily_factors", {})

    existing_cols = set(load_schema().get("daily", {}).get("columns", {}))
    files = sorted(FULL_FUND_DAILY.glob("*.parquet"))
    if files:
        existing_cols.update(parquet_columns(files[0]))

    imported, updates = [], {}
    for start in range(0, len(infos), FACTOR_BATCH_SIZE):
        batch = infos[start:start + FACTOR_BATCH_SIZE]
        all_long = []
        for info in batch:
            fn = info.factor_name or info.path.stem
            desc = descs.get(fn, "（新因子描述.csv 未提供说明）")
            try:
                df = load_file(info.path)
            except Exception as e:
                log(f"  ⚠️ 截面因子 {info.path.name}: 跳过（{e}）")
                continue
            if df is None or len(df) == 0:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = normalize_daily_frame(df, f"截面因子 {fn}")
            if len(df) == 0:
                continue

            is_new = fn not in existing_cols
            last = daily_factors.get(fn)
            if is_new:
                verb = "新增因子 全量导入"
            elif last:
                df = df[df.index > pd.Timestamp(last)]
                if len(df) == 0:
                    log(f"  [{fn}] 已最新（{last}），无更新")
                    continue
                verb = "增量更新"
            else:
                verb = "全量导入（无状态记录）"

            updates[fn] = dstr(df.index.max())
            long_df = df.stack().to_frame(name=fn)
            long_df.index.names = ["datetime", "stock"]
            all_long.append(long_df)
            imported.append(fn)
            log(f"  [{fn}] {verb}：{range_str(df.index)} — {desc}")
            del df

        if not all_long:
            continue
        combined = pd.concat(all_long, axis=1)
        del all_long

        items = []
        seen = set()
        for stock in combined.index.get_level_values("stock").unique():
            stock_str = normalize_code(stock)
            if not stock_str or stock_str in seen:
                continue
            seen.add(stock_str)
            sub = combined.xs(stock, level="stock")
            sub.index.name = "datetime"
            items.append((FULL_FUND_DAILY / f"{stock_str}.parquet", sub))
        n = parallel_merge(items)
        log(f"  → 本批合并写入 {n} 只股票")
        del combined

    return DomainResult(
        message=f"截面因子：处理 {len(imported)} 个" if imported else "截面因子：无更新",
        daily_columns=set(imported),
        updates=updates,
        processed=bool(imported),
    )


def import_minute_factor(infos: list, descs: dict, state: dict) -> DomainResult:
    """导入分钟截面因子（非行情/分钟线/*.parquet 宽表）。

    宽表可能很大（3GB 级），因此：
    - 一次只加载一个因子文件；
    - 按日期逐日 melt 合并，避免一次性 stack；
    - **遇到缺失的基础分钟文件就停在那里**，状态不越过该日期，
      避免中间日期被永久跳过。
    """
    if not infos:
        return DomainResult()
    minute_factors = state.get("minute_factors", {})
    new_cols, updates = [], {}
    processed_any = False

    for info in infos:
        fn = info.factor_name or info.path.stem
        desc = descs.get(fn, "")
        try:
            df = load_file(info.path)
        except Exception as e:
            log(f"  ⚠️ 分钟因子 {info.path.name}: 跳过（{e}）")
            continue
        if df is None or len(df) == 0:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if df.index.has_duplicates:
            n = int(df.index.duplicated(keep="last").sum())
            df = df[~df.index.duplicated(keep="last")]
            log(f"  ⚠️ 分钟因子 {fn}: 去掉 {n} 行重复时间（保留最后一条）")

        last = minute_factors.get(fn)
        if last:
            df = df[df.index > pd.Timestamp(last)]
        if len(df) == 0:
            log(f"  [{fn}] 无新数据（已处理到 {last}）")
            continue
        log(f"  [{fn}] 分钟截面因子增量 {range_str(df.index)} — {desc or fn}")

        date_groups = {}
        for dt in df.index:
            d = pd.Timestamp(dt).strftime("%Y%m%d")
            date_groups.setdefault(d, []).append(dt)

        processed_ts = []
        for d in sorted(date_groups):
            full_path = FULL_MINUTE_BY_DATE / f"{d}.parquet"
            if not full_path.exists():
                log(f"  ⚠️ 分钟因子 {fn}: 缺少基础分钟文件 {d}，在此暂停（状态停在上一个成功日期）")
                break

            day_df = df.loc[date_groups[d]]
            day_df = day_df.loc[:, ~day_df.columns.duplicated()]
            cols = []
            seen_codes = set()
            for c in day_df.columns:
                s = normalize_code(c)
                if s and s not in seen_codes:
                    seen_codes.add(s)
                    cols.append((c, s))
            if not cols:
                continue
            day_df = day_df[[c for c, _ in cols]]
            stocks = [s for _, s in cols]

            times = day_df.index.values
            nt, ns = len(times), len(stocks)
            # day_df.values 是按 (时间, 股票) 行优先展开：
            # instrument = 每个时间都重复一遍股票列表；datetime = 每个时间重复 ns 次。
            idx = pd.MultiIndex.from_arrays(
                [np.tile(stocks, nt), np.repeat(times, ns)],
                names=["instrument", "datetime"],
            )
            series = pd.Series(day_df.values.ravel(), index=idx, name=fn, dtype=float).dropna()
            if series.empty:
                continue
            series.index = pd.MultiIndex.from_arrays(
                [series.index.get_level_values("instrument"),
                 pd.DatetimeIndex(series.index.get_level_values("datetime")).as_unit("ns")],
                names=["instrument", "datetime"],
            )
            if series.index.has_duplicates:
                series = series[~series.index.duplicated(keep="last")]

            if fn in parquet_columns(full_path):
                existing = dedupe_minute_frame(pd.read_parquet(full_path), f"分钟因子 {d}")
                existing[fn] = existing[fn].combine_first(series)
                write_parquet_atomic(existing, full_path, compression="lz4")
            else:
                # 必须读全部列：用 columns=[] 只取索引会丢掉该日期已有的 OHLCV。
                existing = dedupe_minute_frame(pd.read_parquet(full_path), f"分钟因子 {d}")
                existing[fn] = series.reindex(existing.index)
                write_table_atomic(pa.Table.from_pandas(existing, preserve_index=True),
                                   full_path, compression="lz4")

            test_path = TEST_MINUTE_BY_DATE / f"{d}.parquet"
            if test_path.exists():
                te = dedupe_minute_frame(pd.read_parquet(test_path), f"测试分钟因子 {d}")
                te[fn] = (te[fn].combine_first(series)
                          if fn in te.columns else series.reindex(te.index))
                write_parquet_atomic(te, test_path)

            processed_ts.append(max(date_groups[d]))

        if processed_ts:
            processed_any = True
            new_cols.append(fn)
            # 存完整时间戳（含时分），避免下一次把最后一天重算，也避免误跳过。
            updates[fn] = str(pd.Timestamp(max(processed_ts)))

    return DomainResult(
        message=f"分钟截面因子：处理 {len(new_cols)} 个" if new_cols else "分钟截面因子：无更新",
        minute_columns=set(new_cols),
        updates=updates,
        processed=processed_any,
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. 注册 / 提示词 / 检查 / 总结
# ═══════════════════════════════════════════════════════════════════════
def known_desc(c: str) -> str:
    """查列的中文含义：优先 schema.json，其次 factor_field_schema 的 short_name，最后列名本身。"""
    s = load_schema()
    if c in s.get("daily", {}).get("columns", {}):
        return s["daily"]["columns"][c]["description"]
    ff = load_json(FACTOR_FIELD_SCHEMA_PATHS[0]) or {}
    if c in ff and ff[c].get("short_name"):
        return ff[c]["short_name"]
    return c


def register_daily_columns(schema: dict, columns: set, descs: dict) -> set:
    """注册日线列到 schema.daily + 两个 factor_field_schema。

    只返回**真正新增**的列；已有列不重复报告、不覆盖已有描述。
    """
    cols = schema.setdefault("daily", {}).setdefault("columns", {})
    ff_schemas = [(p, load_json(p) or {}) for p in FACTOR_FIELD_SCHEMA_PATHS]
    truly_new = set()

    for c in sorted(columns):
        desc = descs.get(c, "")
        if c not in cols:
            cols[c] = {"description": desc or known_desc(c), "source": "原始数据导入"}
            truly_new.add(c)
        for _, ff in ff_schemas:
            if c not in ff:
                ff[c] = {"factor_name": c, "short_name": desc or c, "formula": "",
                         "source": "原始数据导入", "note": desc or "字段含义待补充（见 新因子描述.csv）"}
                truly_new.add(c)

    save_schema(schema)
    for p, ff in ff_schemas:
        write_json(p, ff)
    return truly_new


def register_minute_columns(schema: dict, columns: set, descs: dict) -> set:
    """注册分钟列到 schema.minute。

    分钟列不写进日线的 factor_field_schema，避免污染日线提示词。
    只返回真正新增的列。
    """
    cols = schema.setdefault("minute", {}).setdefault("columns", {})
    truly_new = set()
    for c in sorted(columns):
        if c not in cols:
            cols[c] = {"description": descs.get(c, "") or c, "source": "原始数据导入"}
            truly_new.add(c)
    save_schema(schema)
    return truly_new


def update_prompts(schema: dict = None) -> str:
    """把 schema 里的字段含义刷新进 rdagent/components/coder/factor_coder/prompts.yaml。

    替换 <!-- DAILY_COLUMNS --> / <!-- MINUTE_COLUMNS --> 两个标记块的内容。
    仓库未同步到本机（找不到 prompts.yaml）时跳过。
    """
    schema = schema or load_schema()
    ff = load_json(FACTOR_FIELD_SCHEMA_PATHS[0]) or {}

    def fmt_line(col: str, src: dict) -> str:
        f = ff.get(col)
        if f and f.get("short_name") and f["short_name"] != col:
            out = f"  - {col}: {f['short_name']}"
            if f.get("note"):
                out += f"（{f['note']}）"
            return out
        desc = src.get("description", col)
        return f"  - {col}: {desc}" if desc and desc != col else f"  - {col}"

    daily = "\n".join(fmt_line(k, v) for k, v in sorted(schema.get("daily", {}).get("columns", {}).items()))
    minute = "\n".join(fmt_line(k, v) for k, v in sorted(schema.get("minute", {}).get("columns", {}).items()))
    rel = "rdagent/components/coder/factor_coder/prompts.yaml"
    full = PROJECT_REPO / rel
    if not full.exists():
        return "提示词：跳过（仓库未同步，找不到 prompts.yaml）"
    content = full.read_text(encoding="utf-8")
    changed = False
    for start, end, text in (("<!-- DAILY_COLUMNS -->", "<!-- /DAILY_COLUMNS -->", daily),
                             ("<!-- MINUTE_COLUMNS -->", "<!-- /MINUTE_COLUMNS -->", minute)):
        content, cnt = re.subn(re.escape(start) + r".*?" + re.escape(end), f"{start}\n{text}\n{end}", content, flags=re.DOTALL)
        if cnt:
            changed = True
    if changed:
        write_text_atomic(full, content)
        return "提示词：已刷新 DAILY_COLUMNS / MINUTE_COLUMNS"
    return "提示词：无需变更（已同步）"


def cmd_summary() -> str:
    descs = load_descriptions()
    schema = load_schema()
    out = []
    out.append(f"字段含义（{len(descs)}，来自 新因子描述.csv 原话）:")
    for k, v in sorted(descs.items()):
        out.append(f"  - {k}: {v}")
    out.append(f"数据仓库列：日线 {len(schema.get('daily', {}).get('columns', {}))} 列 | 分钟 {len(schema.get('minute', {}).get('columns', {}))} 列")
    try:
        sl = load_json_list(TEST_STOCK_LIST_FILE)
        td = load_json_list(TEST_TRADE_DATES_FILE)
        out.append(f"测试数据：{len(sl)} 只 × {len(td)} 天（应为 300×300，固定不动）")
    except Exception:
        out.append("测试数据：读取失败")
    cnt = Counter()
    for p in NEW_DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".parquet", ".csv") and p.resolve() != DESC_FILE.resolve():
            cnt[p.parent.name if p.parent.name != NEW_DATA_DIR.name else "(根目录)"] += 1
    out.append("原始数据/ 文件分布：" + ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items())) if cnt else "原始数据/ 当前无待导入文件")
    return "\n".join(out)


def check_only() -> str:
    """扫描并按类型汇总输出（隐藏逐文件细节，避免刷屏）。"""
    files = scan_new_data_dir()
    if not files:
        return "原始数据/ 无 parquet/csv"
    from collections import defaultdict
    groups = defaultdict(list)
    for p, hint in files:
        try:
            cols, nrows = read_header(p)
            t = detect_file_type(p, hint, cols)
            extra = ""
            if t == "daily_data":
                extra = "（日线行情全天）"
            elif t in ("cross_sectional_factor", "minute_cross_sectional_factor"):
                extra = ""
            elif t == "minute_by_date":
                f = date_from_filename(p.name)
                extra = f"（{f.strftime('%Y%m%d') if f else '?'}）"
            elif t == "barra_model":
                extra = ""
            groups[t].append((p.name, nrows, extra))
        except Exception as e:
            groups["ERROR"].append((p.name, 0, f"❌ {e}"))

    type_label = {
        "daily_data": "日线行情",
        "minute_by_date": "分钟线",
        "cross_sectional_factor": "截面因子",
        "minute_cross_sectional_factor": "分钟截面因子",
        "standard": "标准表",
        "barra_model": "Barra 模型",
        "ERROR": "无法识别",
    }
    out = [f"扫描到 {len(files)} 个文件:"]
    for t, items in groups.items():
        if t == "minute_by_date":
            # 分钟：只列范围 + 数量
            dates = sorted(x[2].strip("（）") or "" for x in items)
            d_clean = [d for d in dates if d]
            rng = f"{d_clean[0]} ~ {d_clean[-1]}" if d_clean else "-"
            out.append(f"  分钟线：{len(items)} 个文件（{rng}）")
        elif t == "cross_sectional_factor":
            names = sorted(x[0] for x in items)
            out.append(f"  截面因子：{len(items)} 个 — {', '.join(names)}")
            continue
        elif t == "ERROR":
            for name, _, e in items:
                out.append(f"  ❌ {name}: {e}")
        else:
            out.append(f"  {type_label.get(t, t)}：{len(items)} 个")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════
# 7. 主导入流程
# ═══════════════════════════════════════════════════════════════════════
def _print_plan(inspections: list) -> list:
    """生成 dry-run 预览行。

    这里只使用文件级信息（不加载内容），避免 --dry-run 把全量分钟/宽表读进内存。
    """
    lines = ["=== 将要导入（dry-run） ==="]
    for i in inspections:
        if i.error:
            lines.append(f"  ❌ {i.path.name}: {i.error}")
            continue
        t = i.file_type
        if t == "daily_data":
            lines.append(f"  {i.path.name} → 日线行情（{i.row_count} 行）")
        elif t == "minute_by_date":
            f = date_from_filename(i.path.name)
            lines.append(f"  {i.path.name} → 分钟（{f.strftime('%Y%m%d') if f else '?'}，"
                         f"{i.row_count} 行）")
        elif t in ("cross_sectional_factor", "minute_cross_sectional_factor"):
            lines.append(f"  {i.path.name} → 截面因子（{i.row_count} 行）")
        elif t == "barra_model":
            lines.append(f"  {i.path.name} → Barra（复制）")
        else:
            lines.append(f"  {i.path.name} → 标准表（不支持自动导入）")
    return lines


def do_import(dry_run: bool = False) -> int:
    """主导入流程入口。

    扫描 → 分拣 → 日线 / 分钟 / 截面因子 / 分钟截面因子 / Barra
    → 按域注册 schema → 重建日历/股票池 → 补测试列 → 保存状态。

    所有数据文件按需加载，不在进入导入前把所有文件读进内存。
    dry_run=True 只打印文件级计划，不加载内容。
    """
    files = scan_new_data_dir()
    if not files:
        log("原始数据/ 下没有可导入的文件（parquet/csv）")
        return 0
    if not FULL_MARKET_DAILY.exists():
        log(f"❌ 全量行情目录不存在: {FULL_MARKET_DAILY}")
        return 1

    descs = load_descriptions()
    state = load_state()
    test_stocks = set(load_json_list(TEST_STOCK_LIST_FILE))
    minute_dates = set(load_json_list(FULL_MINUTE_TRADE_DATES))
    minute_stocks = set(load_json_list(FULL_MINUTE_STOCK_LIST))

    # 只读 header 做分拣；真正导入时各域自己加载内容。
    inspections = [inspect_file(p, h) for p, h in files]
    if dry_run:
        log("\n".join(_print_plan(inspections)))
        return 0

    daily_data = None
    minute_infos, factor_infos, minute_factor_infos = [], [], []

    # ── 分拣 ──
    for info in inspections:
        if info.error:
            log(f"  ⚠️ {info.path.name}: {info.error}（跳过）")
            continue
        if info.file_type == "barra_model":
            log(import_barra(info))
        elif info.file_type == "minute_by_date":
            minute_infos.append(info)
        elif info.file_type == "cross_sectional_factor":
            factor_infos.append(info)
        elif info.file_type == "minute_cross_sectional_factor":
            minute_factor_infos.append(info)
        elif info.file_type == "daily_data":
            daily_data = info
        else:
            log(f"  ⚠️ {info.path.name}: 标准表不支持自动导入（请放到 日线/分钟线/非行情）")

    daily_columns: set[str] = set()
    daily_targets: dict[str, str] = {}
    minute_columns: set[str] = set()
    minute_processed = False

    # ── 日线行情 ──
    if daily_data:
        mtime = daily_data.path.stat().st_mtime
        already_done = (DAILY_MTIME_FILE.exists()
                        and float(DAILY_MTIME_FILE.read_text()) == mtime)
        if already_done:
            # 兼容旧 import_state（可能没有 daily_market）：从 dailyData 源文件
            # 的最大日期补起，而不是用全量行情日历（后者可能已经被分钟/其他来源推得更远）。
            if not state.get("daily_market"):
                last = max_date_in_file(daily_data.path, daily_data.columns)
                if last is not None:
                    state["daily_market"] = dstr(last)
            log("日线行情 dailyData：跳过（文件未变更）")
        else:
            result = import_daily_data(daily_data, state)
            log(result.message)
            if result.processed:
                daily_columns |= result.daily_columns
                for c in result.daily_columns:
                    daily_targets[c] = "market"   # 行情列 → 补测试行情
            write_text_atomic(DAILY_MTIME_FILE, str(mtime))

    # ── 分钟线 ──
    if minute_infos:
        results = [import_minute_file(info, minute_dates, minute_stocks, state)
                   for info in minute_infos]
        for r in results:
            log(r.message)
        for r in results:
            if r.processed:
                minute_columns |= r.minute_columns
                minute_processed = True
        log(f"分钟线：处理 {sum(1 for r in results if r.processed)}/{len(results)} 个文件")

    # ── 日频截面因子 ──
    if factor_infos:
        result = import_cross_sectional_factors(factor_infos, descs, state)
        log(result.message)
        if result.processed:
            daily_columns |= result.daily_columns
            for c in result.daily_columns:
                daily_targets[c] = "fundamental"   # 非行情列 → 补测试非行情
            state.setdefault("daily_factors", {}).update(result.updates)
        # 因子源日期绝不并入交易日历：日历只由行情数据（日线 per-stock / 分钟 per-date）决定。

    # ── 分钟截面因子 ──
    if minute_factor_infos:
        result = import_minute_factor(minute_factor_infos, descs, state)
        log(result.message)
        if result.processed:
            minute_columns |= result.minute_columns
            state.setdefault("minute_factors", {}).update(result.updates)
            minute_processed = True

    # ── 按域注册 schema ──
    # 重新加载，避免 import_minute_factor 等函数写盘后仍用旧对象覆盖。
    schema = load_schema()
    daily_new = register_daily_columns(schema, daily_columns, descs) if daily_columns else set()
    minute_new = register_minute_columns(schema, minute_columns, descs) if minute_columns else set()

    # ── 元数据 ──
    rebuild_day_trade_dates()
    rebuild_stock_list()
    if minute_processed:
        update_minute_meta(minute_dates, minute_stocks)

    # ── 测试补列（日线/非行情）──
    # 处理过的列都补一次，确保之前的测试列缺失也能自愈。
    if test_stocks:
        for col in sorted(daily_columns):
            target = daily_targets.get(col, "fundamental")
            full_dir = FULL_FUND_DAILY if target == "fundamental" else FULL_MARKET_DAILY
            test_dir = TEST_FUND_DAILY if target == "fundamental" else TEST_MARKET_DAILY
            n, n_with_data = copy_new_col_to_test(full_dir, test_dir, col, sorted(test_stocks))
            if n and not n_with_data:
                log(f"  ⚠️ {col}: 补入 {n} 只测试股票，但测试日期内全为空（检查因子日期覆盖）")
            else:
                log(f"  ℹ️ {col}: 补入 {n} 只测试股票（{n_with_data} 只有非空值）")

    save_state(state)

    log("")
    log("✅ 导入完成")
    if daily_new or minute_new:
        parts = []
        if daily_new:
            parts.append("日线 " + ", ".join(sorted(daily_new)))
        if minute_new:
            parts.append("分钟 " + ", ".join(sorted(minute_new)))
        log("   新增列: " + "；".join(parts))
    return 0


# ═══════════════════════════════════════════════════════════════════════
# 8. 入口
# ═══════════════════════════════════════════════════════════════════════
def main():
    """CLI 入口：按参数分发到对应模式（check / dry-run / update-prompts / summary / rebuild-trade-dates / 默认导入）。"""
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--update-prompts", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--rebuild-trade-dates", action="store_true",
                   help="分别重建日线/分钟交易日历（不把分钟日期并进日线）")
    args = p.parse_args()

    if args.update_prompts:
        log(update_prompts())
        return 0
    if args.check:
        log(check_only())
        return 0
    if args.dry_run:
        return do_import(dry_run=True)
    if args.summary:
        log(cmd_summary())
        return 0
    if args.rebuild_trade_dates:
        union = rebuild_authoritative_trade_dates()
        if not union:
            return 1
        log("  ℹ️ 已分别重建日线/分钟交易日历（分钟日期不会写入日线日历）。")
        return 0
    return do_import(dry_run=False)


if __name__ == "__main__":
    sys.exit(main())