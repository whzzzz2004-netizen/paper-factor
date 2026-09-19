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
  1. 日线行情  dailyData.parquet  → 全量/测试行情 per-stock（按 daily_market 增量）
  2. 分钟线    YYYYMMDD.parquet    → 全量/测试 minute_by_date（已存在跳过；新列合并）
  3. 截面因子  非行情/日线/*.parquet → 全量非行情 per-stock（自动对齐交易日；新因子全量/旧增量）
  4. 分钟截面 非行情/分钟线/*.parquet → 合并进分钟 by_date（逐日防 OOM；minute_factors 增量）
  5. Barra 文件（文件名含特征）         → 复制到 barra_model/
  6. 新列自动注册 schema.json + factor_field_schema.json + 测试数据补列
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DATA_ROOT = Path(os.environ.get("PAPER_FACTOR_DATA_ROOT",
                                r"D:\paper-factor-data" if os.name == "nt" else "/mnt/d/paper-factor-data"))
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
INTERNAL_COLS = {"__index_level_0__", "Unnamed: 0", "level_0", "index"}
BARRA_KEYWORDS = ["因子收益率表", "因子暴露表", "特质收益率表", "特质风险表", "风险因子协方差矩阵表"]


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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(items))), encoding="utf-8")


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


def code_from_filename(name: str) -> Optional[str]:
    stem = Path(name).stem
    if date_from_filename(stem) is not None:
        return None
    s = stem.strip().lower()
    for pfx in ("sh", "sz", "bj"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    for sfx in (".xshg", ".xshe", ".xin9"):
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break
    if not s or not s.isdigit():
        return None
    return str(int(s)).zfill(6)


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


def prepare_frame(df: pd.DataFrame, path: Path):
    """(数据df[DatetimeIndex], 股票Series, symbol列名, 日期来源说明)。"""
    df = df.drop(columns=[c for c in df.columns if c in INTERNAL_COLS])
    columns = list(df.columns)
    symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in columns), None)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in columns), None)

    if date_col is not None:
        idx = parse_dates(df[date_col])
        df = df.drop(columns=[date_col])
        df.index = pd.DatetimeIndex(idx, name="datetime")
        date_source = f"列:{date_col}"
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "datetime"
        date_source = "索引"
    else:
        fdate = date_from_filename(path.name)
        if fdate is None:
            raise ValueError(f"无法识别日期: {path.name}")
        df.index = pd.DatetimeIndex([pd.Timestamp(fdate)] * len(df), name="datetime")
        date_source = f"文件名:{path.name}"

    if symbol_col is not None:
        codes = df[symbol_col].apply(normalize_code)
        df = df.drop(columns=[symbol_col])
    else:
        code = code_from_filename(path.name)
        if code is None:
            raise ValueError(f"无法识别股票代码: {path.name}")
        codes = pd.Series([code] * len(df), index=df.index)
        symbol_col = "<文件名>"

    df = df.dropna(axis=1, how="all")
    return df, codes, symbol_col, date_source


class FileInfo:
    def __init__(self, path: Path, hint: Optional[str]):
        self.path = path
        self.hint = hint
        self.file_type = "standard"
        self.error = None
        self.columns = []
        self.row_count = 0
        self.symbol_col = None
        self.date_source = ""
        self.market_cols = []
        self.fund_cols = []
        self.new_cols = []
        self.new_target = None
        self.stocks = []
        self.date_min = None
        self.date_max = None
        self.df = None
        self.codes = None
        self.factor_name = None


def inspect_file(path: Path, hint: Optional[str], load: bool = False) -> FileInfo:
    info = FileInfo(path, hint)
    if hint == "barra" or any(k in path.name for k in BARRA_KEYWORDS):
        info.file_type = "barra_model"
        info.date_source = "Barra 模型"
        return info
    try:
        columns, nrows = read_header(path)
        info.columns = columns
        info.row_count = nrows
        info.file_type = detect_file_type(path, hint, columns)

        if info.file_type == "minute_by_date":
            info.date_source = "分钟 per-date"
            if load:
                info.df = load_file(path)
                fdate = date_from_filename(path.name)
                if fdate is not None:
                    info.date_min = info.date_max = fdate
                if info.df.index.names and "instrument" in info.df.index.names:
                    inst = info.df.index.get_level_values("instrument").unique()
                    info.stocks = sorted({s.zfill(6) for s in inst if pd.notna(s)})
                elif "symbol" in info.df.columns:
                    info.stocks = sorted({normalize_code(s) for s in info.df["symbol"].unique() if pd.notna(s)})
            return info

        if info.file_type in ("cross_sectional_factor", "minute_cross_sectional_factor"):
            info.factor_name = path.stem
            info.date_source = "截面因子索引"
            if load:
                info.df = load_file(path)
                if isinstance(info.df.index, pd.DatetimeIndex):
                    info.date_min = info.date_max = info.df.index.min()
                    info.date_max = info.df.index.max()
                info.stocks = sorted({str(int(c)).zfill(6) for c in info.df.columns if re.match(r"^\d+$", str(c))})
            return info

        if info.file_type == "daily_data":
            info.date_source = "dailyData.parquet"

        # 标准表 / dailyData：记录 symbol 列与日期来源（已去掉 MARKET/FUND 清单分类）
        info.symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in columns), None)
        if next((c for c in DATE_COL_CANDIDATES if c in columns), None):
            info.date_source = "列"
        elif date_from_filename(path.name) is not None:
            info.date_source = "文件名日期"
        else:
            info.date_source = "自动检测"

        if load and info.file_type != "daily_data":
            df = load_file(path)
            data_df, codes, sc, src = prepare_frame(df, path)
            info.df = data_df
            info.codes = codes
            info.symbol_col = sc
            info.date_source = src
            info.stocks = sorted({s for s in codes.unique() if s})
            if not data_df.empty:
                info.date_min = data_df.index.min()
                info.date_max = data_df.index.max()
    except Exception as e:
        info.error = str(e)
    return info


# ═══════════════════════════════════════════════════════════════════════
# 4. 写入助手
# ═══════════════════════════════════════════════════════════════════════
def merge_stock(dst: Path, df: pd.DataFrame):
    """合并单只股票到 per-stock（按日期去重，新数据优先）。

    策略：新列（原数据中不存在）只保留源真实的日期值，绝不 ffill 填充——
    源数据没有该日期 → 保持 NaN；若后续更新补齐了这些日期，增量合并时
    会用新值覆盖 NaN（reindex/combine 天然支持）。
    """
    df = df.sort_index()
    if dst.exists():
        old = pd.read_parquet(dst)
        combined = pd.concat([old, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        pq.write_table(pa.Table.from_pandas(combined), dst)
    else:
        pq.write_table(pa.Table.from_pandas(df), dst)


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


def copy_new_col_to_test(full_dir: Path, test_dir: Path, col: str, test_stocks: list) -> int:
    """把新增列从「全量 per-stock」同步到「测试 per-stock」。

    只补列，不动日期范围/股票池（测试数据固定 300×300）。
    做法：全量该列 reindex 到测试文件的日期索引后写回。
    返回成功补列的股票数。
    """
    n = 0
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
        test = pd.read_parquet(test_f)
        test[col] = full_col[col].reindex(test.index)
        pq.write_table(pa.Table.from_pandas(test), test_f)
        n += 1
    return n


# ═══════════════════════════════════════════════════════════════════════
# 5. 各数据域导入
# ═══════════════════════════════════════════════════════════════════════
def import_barra(info: FileInfo) -> str:
    """复制 Barra 风险模型文件到 barra_model/。返回一行摘要。"""
    BARRA_DIR.mkdir(parents=True, exist_ok=True)
    dst = BARRA_DIR / info.path.name
    shutil.copy2(info.path, dst)
    return f"Barra：{info.path.name} → barra_model/（{info.path.stat().st_size/1024/1024:.1f} MB）"


def import_daily_data(info: FileInfo, state: dict) -> str:
    """导入 dailyData.parquet 的新日期段 → 全量行情 per-stock。

    增量边界：只处理 state['daily_market'] 之后的日期；
    导入完成后把 daily_market 推进到本次最大日期。
    返回一行摘要。
    """
    table = pq.read_table(info.path)
    pdf = table.to_pandas()
    date_col = next((c for c in DATE_COL_CANDIDATES if c in pdf.columns), None)
    if date_col is not None:
        pdf.index = pd.DatetimeIndex(parse_dates(pdf[date_col]), name="datetime")
        pdf = pdf.drop(columns=[date_col])
    else:
        pdf.index.name = "datetime"

    cutoff = state.get("daily_market")
    if cutoff:
        pdf = pdf[pdf.index > pd.Timestamp(cutoff)]
    if len(pdf) == 0:
        return f"日线行情 dailyData：无新数据（现有到 {cutoff or '-'}）"

    symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in pdf.columns), None)
    codes = pdf[symbol_col].apply(normalize_code) if symbol_col else pd.Series([normalize_code(info.path.stem)] * len(pdf))
    drop_cols = [c for c in INTERNAL_COLS if c in pdf.columns]
    if symbol_col:
        drop_cols.append(symbol_col)
    if drop_cols:
        pdf = pdf.drop(columns=drop_cols)
    pdf = pdf.dropna(axis=1, how="all")

    state["daily_market"] = dstr(pdf.index.max())
    items = []
    for stock, grp in pdf.groupby(codes):
        sub = grp.dropna(axis=1, how="all")
        if not sub.empty:
            items.append((FULL_MARKET_DAILY / f"{stock}.parquet", sub))
    n_write = parallel_merge(items)

    return f"日线行情 dailyData：新增 {range_str(pdf.index)}（{len(pdf)} 行, {codes.nunique()} 只股票写入 {n_write}）"


def import_minute_file(info: FileInfo, minute_dates: set, minute_stocks: set, state: dict) -> tuple:
    """导入单个分钟 per-date 文件。

    返回 (是否真正处理, 股票数, 新增分钟列)。
    - 已存在于仓库/已过 minute_market → 返回 (False, 0, []) 跳过
    - 含新列（非标准分钟列）→ 合并进已有 by_date 文件（全量+测试）
    - 标准列 → 直接写入 by_date 新文件
    """
    fdate = date_from_filename(info.path.name)
    if fdate is None:
        return False, 0, []
    d = fdate.strftime("%Y%m%d")
    minute_dates.add(d)
    if (FULL_MINUTE_BY_DATE / f"{d}.parquet").exists():
        return False, 0, []
    mm = state.get("minute_market")
    if mm and fdate <= pd.Timestamp(mm):
        return False, 0, []

    df = info.df.copy()
    new_cols_detected = []
    if "symbol" in df.columns:
        # 扁平格式 → MultiIndex
        df["instrument"] = df["symbol"].apply(normalize_code)
        if "datetime" not in df.columns and "trade_date" in df.columns:
            if df["trade_date"].dtype.kind != "M":
                df["trade_date"] = parse_dates(df["trade_date"])
            df["datetime"] = df.pop("trade_date")
        df = df.drop(columns=[c for c in ["symbol", "date"] if c in df.columns])
        data_cols = [c for c in df.columns if c != "instrument"]
        new_cols_detected = [c for c in data_cols if c not in MINUTE_EXPECTED_COLS and c not in ("datetime", "trade_date")]
        df = df.set_index(["instrument", "datetime"])
        df.index = df.index.set_levels(df.index.levels[1].as_unit("ns"), level="datetime")
        df = df.sort_index()
    elif isinstance(df.index, pd.MultiIndex):
        if df.index.get_level_values("datetime").dtype != np.dtype("datetime64[ns]"):
            df.index = df.index.set_levels(df.index.levels[1].as_unit("ns"), level="datetime")
        new_cols_detected = [c for c in df.columns if c not in MINUTE_EXPECTED_COLS]
    else:
        return False, 0, []

    if new_cols_detected:
        # 新列 → 合并进已有 by_date
        date_groups = {}
        for _, (_, dt) in zip(df.index, df.index):
            dd = pd.Timestamp(dt).strftime("%Y%m%d")
            date_groups.setdefault(dd, []).append(dt)
        # 简化：按 datetime 值分组
        date_groups = {}
        for dt in df.index.get_level_values("datetime"):
            dd = pd.Timestamp(dt).strftime("%Y%m%d")
            date_groups.setdefault(dd, []).append(dt)
        n_stock = 0
        for dd, ts in date_groups.items():
            sub = df.loc[df.index.get_level_values("datetime").isin(ts)]
            full_path = FULL_MINUTE_BY_DATE / f"{dd}.parquet"
            if full_path.exists():
                existing = pd.read_parquet(full_path)
                for col in new_cols_detected:
                    existing[col] = existing[col].combine_first(sub[col]) if col in existing.columns else sub[col]
                existing.to_parquet(full_path)
            else:
                sub[new_cols_detected].to_parquet(full_path)
            test_path = TEST_MINUTE_BY_DATE / f"{dd}.parquet"
            if test_path.exists():
                te = pd.read_parquet(test_path)
                for col in new_cols_detected:
                    te[col] = te[col].combine_first(sub[col]) if col in te.columns else sub[col]
                te.to_parquet(test_path)
            minute_dates.add(dd)
            n_stock += sub.index.get_level_values("instrument").nunique()
        return True, n_stock, new_cols_detected
    else:
        dst = FULL_MINUTE_BY_DATE / f"{d}.parquet"
        pq.write_table(pa.Table.from_pandas(df), dst)
        for inst in df.index.get_level_values("instrument").unique():
            minute_stocks.add(inst.zfill(6))
        return True, 0, []


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


def rebuild_authoritative_trade_dates() -> list:
    """重建全量权威交易日历：日线实际 ∪ 分钟实际（取更宽范围）。

    统一 'YYYYMMDD' 格式，写回：
      - 日线全量 trade_dates.json
      - 分钟 stock_data/trade_dates.json + minute_by_date/trade_dates.json（两份）
    权威日历 = 日线 ∪ 分钟；若日线空，以分钟为准（反向亦然）。
    测试数据不动（固定 300 天）。
    """
    day = rebuild_day_trade_dates()
    minute = rebuild_minute_trade_dates()
    union = sorted(set(day) | set(minute))
    if not union:
        log("  ❌ 日线与分钟行情都没有实际数据，无法重建日历")
        return []
    # 若任一侧以并集为准写回，保证三个文件一致（各自原值可能落后）
    if sorted(union) != day:
        write_json_list(FULL_TRADE_DATES_FILE, union)
        day = union
    if sorted(union) != sorted(minute):
        write_json_list(FULL_MINUTE_TRADE_DATES, union)
        write_json_list(MINUTE_SUBDIR_TRADE_DATES, union)
        minute = union
    log(f"  ✅ 权威交易日历 = {dstr(union[0])} ~ {dstr(union[-1])} 共 {len(union)} 天"
        f"（日线 {len(day)} ∪ 分钟 {len(minute)}）")
    return union


def import_cross_sectional_factors(infos: list, descs: dict, state: dict) -> tuple:
    """批量导入日频截面因子（非行情/日线/*.parquet 宽表）→ 非行情全量 per-stock。

    每个因子独立按自己的日期更新（不被 trade_dates 框死）：
    - 全新因子 → 全量导入该因子源的所有日期
    - 已有因子 → 只处理 state['daily_factors'][因子名] 之后的数据
    - 所有因子按 (日期, 股票) 对齐后一次性 merge 到 per-stock
    返回 (新导入因子列表, {因子名: 最新日期}, 因子源日期集合)。
    """
    if not infos:
        return [], {}, set()
    log(f"【截面因子】共 {len(infos)} 个")
    daily_factors = state.get("daily_factors", {})
    existing_cols = set()
    try:
        files = list(FULL_FUND_DAILY.glob("*.parquet"))
        if files:
            existing_cols = set(pd.read_parquet(files[0]).columns)
    except Exception:
        pass

    all_long, imported, updates = [], [], {}
    factor_dates = set()
    for info in infos:
        fn = info.factor_name or info.path.stem
        desc = descs.get(fn, "（新因子描述.csv 未提供说明）")
        df = info.df
        if df is None or len(df) == 0:
            continue
        if df.index.dtype == object:
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        is_new = fn not in existing_cols
        last = daily_factors.get(fn)
        if is_new:
            verb = "新增因子 全量导入"
        elif last:
            df_inc = df[df.index > pd.Timestamp(last)]
            if len(df_inc) == 0:
                log(f"  [{fn}] 已最新（{last}），无更新")
                continue
            df = df_inc
            verb = "增量更新"
        else:
            verb = "全量导入（无状态记录）"
        updates[fn] = dstr(df.index.max())

        for dt in df.index:
            factor_dates.add(dt.strftime("%Y%m%d"))

        long_df = df.stack().to_frame(name=fn)
        long_df.index.names = ["datetime", "stock"]
        all_long.append(long_df)
        imported.append(fn)
        log(f"  [{fn}] {verb}：{range_str(df.index)} — {desc}")

    if not all_long:
        return [], updates, factor_dates

    combined = pd.concat(all_long, axis=1)
    del all_long
    if combined.index.get_level_values("datetime").dtype == object:
        lvl = combined.index.names.index("datetime")
        combined.index = combined.index.set_levels(pd.to_datetime(combined.index.levels[lvl]), level="datetime")

    items = []
    seen = set()
    for stock in combined.index.get_level_values("stock").unique():
        stock_str = str(int(stock)).zfill(6)
        if stock_str in seen:
            continue
        seen.add(stock_str)
        sub = combined.xs(stock, level="stock")
        sub.index.name = "datetime"
        items.append((FULL_FUND_DAILY / f"{stock_str}.parquet", sub))

    n = parallel_merge(items)
    log(f"  → 合并写入 {n} 只股票")
    return imported, updates, factor_dates


def import_minute_factor(infos: list, descs: dict, state: dict) -> tuple:
    """批量导入分钟截面因子（非行情/分钟线/*.parquet 宽表）。

    宽表可能很大（60K×5000 ≈ 3GB），因此按「日期 → 单日 melt」逐日并行
    合并进 minute_by_date 文件，避免一次性 stack 造成 OOM。

    增量边界：state['minute_factors'][因子名] 之后的数据。
    返回 (新因子列, {因子名: 最新日期})。
    """
    if not infos:
        return [], {}
    minute_factors = state.get("minute_factors", {})
    new_cols, updates = [], {}
    for info in infos:
        fn = info.factor_name or info.path.stem
        df = info.df
        desc = descs.get(fn, "")
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
        day_list = sorted(d for d in date_groups if (FULL_MINUTE_BY_DATE / f"{d}.parquet").exists())

        def _proc(d):
            ts = date_groups[d]
            day_df = df.loc[ts]
            times = day_df.index.values
            stocks = [normalize_code(c) for c in day_df.columns]
            nt, ns = len(times), len(stocks)
            idx = pd.MultiIndex.from_arrays([np.repeat(times, ns), np.tile(stocks, nt)], names=["instrument", "datetime"])
            series = pd.Series(day_df.values.ravel(), index=idx, name=fn, dtype=float).dropna()
            if series.empty:
                return
            full_path = FULL_MINUTE_BY_DATE / f"{d}.parquet"
            if fn in pq.read_schema(full_path).names:
                existing = pd.read_parquet(full_path)
                existing[fn] = existing[fn].combine_first(series)
                existing.to_parquet(full_path, compression="lz4")
            else:
                existing = pd.read_parquet(full_path, columns=[])
                existing[fn] = series.reindex(existing.index)
                pq.write_table(pa.Table.from_pandas(existing, preserve_index=True), full_path, compression="lz4")
            test_path = TEST_MINUTE_BY_DATE / f"{d}.parquet"
            if test_path.exists():
                te = pd.read_parquet(test_path)
                te[fn] = te[fn].combine_first(series) if fn in te.columns else series.reindex(te.index)
                te.to_parquet(test_path)

        n_workers = min(2, len(day_list) or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_proc, day_list))
        new_cols.append(fn)
        updates[fn] = dstr(df.index.max())

    if new_cols:
        schema = load_schema()
        cols = schema.setdefault("minute", {}).setdefault("columns", {})
        for c in new_cols:
            if c not in cols:
                cols[c] = {"description": descs.get(c, c), "source": "原始数据导入"}
        save_schema(schema)
    return new_cols, updates


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


def register_new_cols(schema: dict, encountered: set, write: bool, descs: dict) -> list:
    """把本次遇到的新列注册进 schema.json + 两个 factor_field_schema.json。

    - description / short_name 优先取 新因子描述.csv 原话（descs），缺失则回退 known_desc
    - 返回真正的新列列表（用于测试补列 + NEW_COLUMNS 提示）
    write=True 时才写盘。
    """
    schema_cols = schema.setdefault("daily", {}).setdefault("columns", {})
    truly_new = []
    for c in sorted(encountered):
        desc = descs.get(c, "")
        if c not in schema_cols:
            schema_cols[c] = {"description": desc or known_desc(c), "source": "原始数据导入"}
        for ffp in FACTOR_FIELD_SCHEMA_PATHS:
            ff = load_json(ffp) or {}
            if c not in ff:
                ff[c] = {"factor_name": c, "short_name": desc or c, "formula": "",
                         "source": "原始数据导入", "note": desc or "字段含义待补充（见 新因子描述.csv）"}
                if write:
                    write_json(ffp, ff)
        truly_new.append(c)
    if write:
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
        full.write_text(content, encoding="utf-8")
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
    """生成 dry-run 预览行：每个文件将被导入到哪里 + 覆盖范围。"""
    lines = ["=== 将要导入（dry-run） ==="]
    for i in inspections:
        if i.error:
            lines.append(f"  ❌ {i.path.name}: {i.error}")
            continue
        t = i.file_type
        if t == "daily_data":
            lines.append(f"  {i.path.name} → 日线行情（新日期段）")
        elif t == "minute_by_date":
            f = date_from_filename(i.path.name)
            lines.append(f"  {i.path.name} → 分钟（{f.strftime('%Y%m%d') if f else '?'}）")
        elif t in ("cross_sectional_factor", "minute_cross_sectional_factor"):
            r = range_str([i.date_min, i.date_max] if i.date_min is not None else [])
            lines.append(f"  {i.path.name} → 截面因子（{len(i.stocks)} 只，{r} 天）")
        elif t == "barra_model":
            lines.append(f"  {i.path.name} → Barra（复制）")
        else:
            lines.append(f"  {i.path.name} → 标准表（{len(i.stocks)} 只，{i.date_min} 起）")
    return lines


def do_import(dry_run: bool = False) -> int:
    """主导入流程入口。

    步骤：扫描 → 分拣 → 依次导入日线行情 / 分钟线 / 截面因子 / 分钟截面因子 /
    Barra → 注册新列 → 更新元数据 → 测试补列 → 保存状态 → 汇总。
    每个环节只输出一行摘要。dry_run=True 只打印将要导入的内容。
    返回 0（成功）或 1（全局目录缺失）。
    """
    files = scan_new_data_dir()
    if not files:
        log("原始数据/ 下没有可导入的文件（parquet/csv）")
        return 0
    if not FULL_MARKET_DAILY.exists():
        log(f"❌ 全量行情目录不存在: {FULL_MARKET_DAILY}")
        return 1

    descs = load_descriptions()
    schema = load_schema()
    state = load_state()
    full_dates = set(load_json_list(FULL_TRADE_DATES_FILE))
    full_stocks = set(load_json_list(FULL_STOCK_LIST_FILE))
    test_stocks = set(load_json_list(TEST_STOCK_LIST_FILE))
    minute_dates = set(load_json_list(FULL_MINUTE_TRADE_DATES))
    minute_stocks = set(load_json_list(FULL_MINUTE_STOCK_LIST))

    inspections = [inspect_file(p, h, load=True) for p, h in files]
    if dry_run:
        log("\n".join(_print_plan(inspections)))
        return 0

    all_dates = set(full_dates)
    all_stocks = set(full_stocks)
    cols_encountered = set()
    new_target = {}
    has_minute = False

    factor_infos, minute_factor_infos, minute_infos = [], [], []
    daily_path = None

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
        continue

    # ── 日线行情 ──
    if daily_data:
        mtime = daily_data.path.stat().st_mtime
        if DAILY_MTIME_FILE.exists() and float(DAILY_MTIME_FILE.read_text()) == mtime:
            log("日线行情 dailyData：跳过（文件未变更）")
        else:
            log(import_daily_data(daily_data, state))
            DAILY_MTIME_FILE.write_text(str(mtime))

    # ── 分钟 ──
    if minute_infos:
        t0 = time.time()
        new_days, new_min_cols = 0, []
        for info in minute_infos:
            handled, n_stock, new_cols = import_minute_file(info, minute_dates, minute_stocks, state)
            if handled:
                new_days += 1
                new_min_cols.extend(new_cols)
        new_min_cols = list(dict.fromkeys(new_min_cols))
        log(f"分钟线：新增 {new_days} 个日期，跳过 {len(minute_infos)-new_days} 个已存在日期"
            + (f"，含新列 {new_min_cols}" if new_min_cols else ""))
        has_minute = new_days > 0

    # ── 截面因子 ──
    if factor_infos:
        c_new, c_updates, c_dates = import_cross_sectional_factors(factor_infos, descs, state)
        cols_encountered.update(c_new)
        for c in c_new:
            new_target[c] = "fundamental"
        state.setdefault("daily_factors", {}).update(c_updates)
        # 因子源日期绝不并入交易日历：日历只由行情数据（日线 per-stock / 分钟 per-date）决定，
        # 非行情因子（分析师、EPS 等）可能含周末/非交易日，若并入会把周末幽灵行扩散到所有因子输出。

    # ── 分钟截面因子 ──
    if minute_factor_infos:
        m_new, m_up = import_minute_factor(minute_factor_infos, descs, state)
        cols_encountered.update(m_new)
        for c in m_new:
            new_target[c] = "minute"
        state.setdefault("minute_factors", {}).update(m_up)
        has_minute = True

    # ── 注册新列 ──
    truly_new = register_new_cols(schema, cols_encountered, write=True, descs=descs)

    # ── 元数据（日线） ──
    # 股票并集仍走 .json 合并；日期改为「按实际行情数据重建」（权威日历，
    # 只由行情数据决定；不再用 all_dates —— 其曾并入因子源日期 → 周末幽灵行）。
    _day_dates = rebuild_day_trade_dates()
    if _day_dates:
        all_dates = set(_day_dates)
    if sorted(all_stocks) != sorted(full_stocks):
        write_json_list(FULL_STOCK_LIST_FILE, sorted(all_stocks))
        log(f"  ℹ️ 日线 stock_list.json: {len(full_stocks)} → {len(all_stocks)} 只")

    # 分钟元数据
    if has_minute:
        update_minute_meta(minute_dates, minute_stocks)

    # ── 测试补列 ──
    if truly_new and test_stocks:
        for col in sorted(truly_new):
            tgt = new_target.get(col, "fundamental")
            full_dir = FULL_FUND_DAILY if tgt == "fundamental" else FULL_MARKET_DAILY
            test_dir = TEST_FUND_DAILY if tgt == "fundamental" else TEST_MARKET_DAILY
            n = copy_new_col_to_test(full_dir, test_dir, col, sorted(test_stocks))
            log(f"  ℹ️ {col}: 补入 {n} 只测试股票")

    save_state(state)

    log("")
    log("✅ 导入完成")
    if truly_new:
        log(f"   新增因子: {', '.join(sorted(truly_new))}（已注册 schema + factor_field_schema）")
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
                   help="重建权威交易日历（日线 per-stock 实际 ∪ 分钟 per-date）并写回三个 trade_dates.json")
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
        log("  ℹ️ 已重建权威交易日历。旧因子 parquet 中的周末脏行会在下次 /all 增量合并时被裁剪。")
        return 0
    return do_import(dry_run=False)


if __name__ == "__main__":
    sys.exit(main())