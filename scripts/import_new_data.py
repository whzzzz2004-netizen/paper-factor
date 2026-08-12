#!/usr/bin/env python3
"""
从本地「新建文件/」目录增量导入新增数据（行情 / 非行情 / 分钟 / 截面因子）。

用户把新增数据手动放到 新建文件/ 下，本脚本自动检测格式并导入。

支持的数据格式：
1. 日线数据（标准格式）：含 symbol+date 列，per-stock 合并
2. dailyData.parquet：全量日线单文件（symbol+date+数据列）
3. 分钟数据（分钟线/YYYYMMDD.parquet）：per-date 分钟，MultiIndex[instrument,datetime]
4. 截面因子（非行情/因子名.parquet）：index=日期, columns=股票代码, values=float64
5. 新建文件/基本面因子说明.csv：新因子描述（CSV 无表头，因子名,描述文本）

用法:
  python3 scripts/import_new_data.py --check           # 扫描 新建文件/，预览文件与列 schema
  python3 scripts/import_new_data.py --dry-run         # 打印将要导入的内容，不执行
  python3 scripts/import_new_data.py                   # 自动检测格式并导入
  python3 scripts/import_new_data.py --update-prompts-only  # 仅根据 schema.json 更新 prompt 标记块
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WH = "数据仓库"

# ── 数据目录（数据仓库分层） ──
FULL_MARKET_DAILY = PROJECT_ROOT / WH / "行情数据" / "日线" / "全量" / "stock_data" / "daily"
TEST_MARKET_DAILY = PROJECT_ROOT / WH / "行情数据" / "日线" / "测试" / "stock_data" / "daily"
FULL_FUND_DAILY = PROJECT_ROOT / WH / "非行情数据" / "全量" / "stock_data" / "daily"
TEST_FUND_DAILY = PROJECT_ROOT / WH / "非行情数据" / "测试" / "stock_data" / "daily"

# Barra 风险模型目录
BARRA_MODEL_DIR = PROJECT_ROOT / WH / "barra_model"
# 文件名含这些关键字的文件按 Barra 模型处理（放 新建文件/barra/ 或根目录均可）
BARRA_KEYWORDS = ["因子收益率表", "因子暴露表", "特质收益率表", "特质风险表", "风险因子协方差矩阵表"]

# 全量/测试元数据（位于 行情数据/日线 下）
FULL_META = FULL_MARKET_DAILY
FULL_TRADE_DATES_FILE = FULL_META / "trade_dates.json"
FULL_STOCK_LIST_FILE = FULL_META / "stock_list.json"
TEST_META = TEST_MARKET_DAILY
TEST_TRADE_DATES_FILE = TEST_META / "trade_dates.json"
TEST_STOCK_LIST_FILE = TEST_META / "stock_list.json"

# 分钟数据目录
FULL_MINUTE_BY_DATE = PROJECT_ROOT / WH / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute_by_date"
FULL_MINUTE_DIR = PROJECT_ROOT / WH / "行情数据" / "分钟线" / "全量" / "stock_data" / "minute"
TEST_MINUTE_BY_DATE = PROJECT_ROOT / WH / "行情数据" / "分钟线" / "测试" / "stock_data" / "minute_by_date"
TEST_MINUTE_DIR = PROJECT_ROOT / WH / "行情数据" / "分钟线" / "测试" / "stock_data" / "minute"
FULL_MINUTE_META = FULL_MINUTE_BY_DATE.parent  # stock_data/
FULL_MINUTE_TRADE_DATES = FULL_MINUTE_META / "trade_dates.json"
FULL_MINUTE_STOCK_LIST = FULL_MINUTE_META / "stock_list.json"
TEST_MINUTE_TRADE_DATES = TEST_MINUTE_BY_DATE.parent / "trade_dates.json"
TEST_MINUTE_STOCK_LIST = TEST_MINUTE_BY_DATE.parent / "stock_list.json"

SCHEMA_FILE = PROJECT_ROOT / "data" / "schema.json"
FACTOR_FIELD_SCHEMA_PATHS = [
    PROJECT_ROOT / WH / "行情数据" / "日线" / "全量" / "factor_field_schema.json",
    PROJECT_ROOT / WH / "行情数据" / "日线" / "测试" / "factor_field_schema.json",
]

NEW_DATA_DIR = PROJECT_ROOT / "新建文件"
DESC_FILE = NEW_DATA_DIR / "基本面因子说明.csv"

# ── 列分类（与 scripts/strip_fundamental_cols.py 一致） ──
MARKET_COLS = [
    "open", "close", "high", "low", "factor", "volume", "pct_chg", "pre_close",
    "turnover_rate", "EMA5", "EMA10", "EMA20", "jhjj_hsl", "net_pct_main",
    "net_pct_xl", "net_pct_l", "net_pct_m", "net_pct_s", "net_amount_main", "amount",
]
FUNDAMENTAL_COLS = [
    "roe", "roa", "pe_ttm", "pb", "revenue_yoy", "profit_yoy", "gross_margin",
    "net_margin", "debt_to_asset", "ocf_per_share", "market_cap",
    "circulating_market_cap", "total_shares", "float_shares", "adjusted_profit",
    "gross_profit", "total_holders", "holder_change_pct",
]
SYMBOL_COL_CANDIDATES = [
    "symbol", "stock_code", "code", "instrument", "wind_code", "ts_code",
    "sec_code", "股票代码", "代码", "证券代码",
]
DATE_COL_CANDIDATES = ["date", "trade_date", "tradeDate", "datetime", "time", "交易日期", "日期"]

MINUTE_EXPECTED_COLS = {"open", "high", "low", "close", "volume", "return", "factor", "vwap"}


# ── 基本面因子说明.csv 解析 ──
def _load_descriptions() -> dict:
    """读取新建文件/基本面因子说明.csv，返回 {因子名: 描述文本}。
    CSV 格式：第一列因子名（或 文件名.parquet），第二列描述文本（无表头，逗号分隔）。
    兼容 GBK / UTF-8 编码；兼容 2列（因子名,描述）与 3列（文件名,因子名,描述）。
    """
    if not DESC_FILE.exists():
        return {}
    import csv
    import io
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
            # 3列: 文件名,因子名,描述 — key 用文件 stem（去掉 .parquet）和因子名双匹配
            desc = row[2].strip()
            for key in (Path(name).stem, row[1].strip()):
                if key:
                    descs[key] = desc
        else:
            desc = row[1].strip() if len(row) > 1 else ""
            descs[name] = desc
    return descs


# ── schema 读写 ──
def _load_schema() -> dict:
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def _save_schema(s: dict):
    SCHEMA_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_factor_field_schema(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_factor_field_schema(path: Path, s: dict):
    path.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json_list(path: Path) -> list:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.keys())
    return []


def _write_json_list(path: Path, items: list):
    path.write_text(json.dumps(items), encoding="utf-8")


def _known_desc(c: str) -> str:
    """已知列的说明：优先 schema.json，其次 factor_field_schema 的 short_name，最后列名本身。"""
    s = _load_schema()
    if c in s["daily"]["columns"]:
        return s["daily"]["columns"][c]["description"]
    ff = _load_factor_field_schema(FACTOR_FIELD_SCHEMA_PATHS[0])
    if c in ff and ff[c].get("short_name"):
        return ff[c]["short_name"]
    return c


# ── 日期 / 股票代码 解析 ──
def _parse_dates(s: pd.Series) -> pd.Series:
    """把日期列解析为 datetime Series（兼容字符串/整数/浮点，YYYYMMDD 与 YYYY-MM-DD）。"""
    if s.dtype.kind in "iuf":
        s = s.apply(lambda v: str(int(v)) if pd.notna(v) and float(v) == int(float(v)) else np.nan)
    try:
        s = pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        s = pd.to_datetime(s, errors="coerce")
    if s.isna().all():
        s = pd.to_datetime(s.astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    return s


def _date_from_filename(name: str) -> Optional[pd.Timestamp]:
    """从文件名识别日期（YYYY-MM-DD 或 YYYYMMDD）。"""
    m = re.search(r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})", name)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except ValueError:
            pass
    m = re.search(r"(\d{8})", name)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)[:4]), month=int(m.group(1)[4:6]), day=int(m.group(1)[6:8]))
        except ValueError:
            pass
    return None


def _normalize_code(c) -> str:
    """股票代码归一化：去前导零，去交易所前后缀。如 000001→1, 600519→600519, sh600519→600519。"""
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
        return str(int(s))
    except ValueError:
        return s


def _code_from_filename(name: str) -> Optional[str]:
    """从文件名识别股票代码（排除日期文件名）。"""
    stem = Path(name).stem
    if _date_from_filename(stem) is not None:
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
    return str(int(s))


# ── 文件类型检测 ──
def _subdir_hint(path: Path) -> Optional[str]:
    """按所在子目录给出类型提示：日线/→market，非行情/→fundamental，分钟线/→minute，barra/→barra，根目录/→None。"""
    try:
        rel = path.relative_to(NEW_DATA_DIR).parts
    except ValueError:
        return None
    if len(rel) >= 2:
        if rel[0] == "日线":
            return "market"
        if rel[0] == "非行情":
            return "fundamental"
        if rel[0] == "分钟线":
            return "minute"
        if rel[0] == "barra":
            return "barra"
    return None


def _detect_file_type(path: Path, hint: Optional[str], columns: list) -> str:
    """检测文件类型: standard, daily_data, minute_by_date, cross_sectional_factor, barra_model。"""
    # Barra 风险模型：barra/ 子目录，或文件名含 Barra 关键字
    if hint == "barra" or any(k in path.name for k in BARRA_KEYWORDS):
        return "barra_model"

    # dailyData.parquet — 全量日线单文件
    if path.name.lower() == "dailydata.parquet":
        return "daily_data"

    # 分钟 per-date：在分钟线/子目录，含 instrument 列
    if hint == "minute" and "instrument" in columns:
        return "minute_by_date"

    # 截面因子：在非行情/子目录，无 symbol 列，唯一 date-like 列是 datetime（索引名）
    if hint == "fundamental":
        symbol_like = [c for c in columns if c in SYMBOL_COL_CANDIDATES]
        if not symbol_like:
            date_like = [c for c in columns if c in DATE_COL_CANDIDATES]
            # datetime 可能是 pandas 保存的索引名列，不是真正的数据列
            date_only_datetime = (len(date_like) == 1 and date_like[0] == "datetime") or len(date_like) == 0
            if date_only_datetime:
                non_std = [c for c in columns if c not in SYMBOL_COL_CANDIDATES
                           and c not in DATE_COL_CANDIDATES]
                numeric_cols = [c for c in non_std if re.match(r"^\d+$", str(c))]
                if numeric_cols and len(numeric_cols) == len(non_std):
                    return "cross_sectional_factor"

    return "standard"


# ── 列分类 ──
def _classify(data_cols: list, hint: Optional[str]):
    """把数据列分为行情/非行情/新列。返回 (market_cols, fund_cols, new_cols, new_target)。"""
    market = [c for c in data_cols if c in MARKET_COLS]
    fund = [c for c in data_cols if c in FUNDAMENTAL_COLS]
    new = [c for c in data_cols if c not in MARKET_COLS and c not in FUNDAMENTAL_COLS]
    target = _new_target(new, market, fund, hint)
    return market, fund, new, target


def _new_target(new: list, market: list, fund: list, hint: Optional[str]) -> str:
    """新列归类：子目录提示优先，其次多数归类，再其次默认非行情（输出警告）。"""
    if hint and hint in ("market", "fundamental"):
        return hint
    if hint == "minute":
        return "market"
    if len(fund) > len(market):
        return "fundamental"
    if len(market) > len(fund):
        return "market"
    if new:
        print("  ⚠️ 无法判断新列归属（根目录且无已知列参照），默认归为 非行情。"
              "如需归为行情，请把文件放到 新建文件/日线/", flush=True)
    return "fundamental"


# ── 文件扫描与检测 ──
def scan_new_data_dir() -> list:
    if not NEW_DATA_DIR.exists():
        return []
    exts = {".parquet", ".csv"}
    found = []
    for p in sorted(NEW_DATA_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and p.resolve() != DESC_FILE.resolve():
            found.append((p, _subdir_hint(p)))
    return found


def _read_header(path: Path):
    """返回 (列名, 行数)。parquet 用 schema；csv 读表头。"""
    if path.suffix.lower() == ".csv":
        df0 = pd.read_csv(path, nrows=0)
        n = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f) - 1
        return list(df0.columns), max(n, 0)
    pf = pq.ParquetFile(path)
    return pf.schema_arrow.names, pf.metadata.num_rows


def _load_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _prepare_frame(df: pd.DataFrame, path: Path):
    """把原始 df 整理为 (数据df[DatetimeIndex], 股票代码Series, symbol列名, 日期来源说明)。"""
    columns = list(df.columns)
    symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in columns), None)

    date_col = next((c for c in DATE_COL_CANDIDATES if c in columns), None)
    if date_col is not None:
        idx = _parse_dates(df[date_col])
        df = df.drop(columns=[date_col])
        df.index = pd.DatetimeIndex(idx, name="datetime")
        date_source = f"列:{date_col}"
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "datetime"
        date_source = "索引"
    else:
        fdate = _date_from_filename(path.name)
        if fdate is None:
            raise ValueError(f"无法识别日期（无 date 列/索引，文件名也不含日期）: {path.name}")
        df.index = pd.DatetimeIndex([pd.Timestamp(fdate)] * len(df), name="datetime")
        date_source = f"文件名:{path.name}"

    if symbol_col is not None:
        codes = df[symbol_col].apply(_normalize_code)
        df = df.drop(columns=[symbol_col])
    else:
        code = _code_from_filename(path.name)
        if code is None:
            raise ValueError(f"无法识别股票代码（无 symbol 列，文件名也不含代码）: {path.name}")
        codes = pd.Series([code] * len(df), index=df.index)
        symbol_col = "<文件名>"

    df = df.dropna(axis=1, how="all")
    return df, codes, symbol_col, date_source


class FileInfo:
    """单文件检测结果（check 只读表头；dry-run/import 额外 load）。"""

    def __init__(self, path: Path, hint: Optional[str]):
        self.path = path
        self.hint = hint
        self.file_type: str = "standard"
        self.error: Optional[str] = None
        self.columns: list = []
        self.row_count = 0
        self.symbol_col: Optional[str] = None
        self.date_source = ""
        self.market_cols: list = []
        self.fund_cols: list = []
        self.new_cols: list = []
        self.new_target: Optional[str] = None
        self.stocks: list = []
        self.date_min = None
        self.date_max = None
        self.df: Optional[pd.DataFrame] = None
        self.codes: Optional[pd.Series] = None
        # 截面因子专用
        self.factor_name: Optional[str] = None
        self.factor_desc: Optional[str] = None


def inspect_file(path: Path, hint: Optional[str], load: bool = False) -> FileInfo:
    info = FileInfo(path, hint)
    # Barra 模型文件：只按文件名/子目录判断，不读表头（CSV 可能上 GB，数行数会卡死）
    if hint == "barra" or any(k in path.name for k in BARRA_KEYWORDS):
        info.file_type = "barra_model"
        info.date_source = "Barra 模型"
        return info
    try:
        columns, nrows = _read_header(path)
        info.columns = columns
        info.row_count = nrows
        info.file_type = _detect_file_type(path, hint, columns)

        if info.file_type == "minute_by_date":
            info.date_source = "分钟 per-date"
            if load:
                info.df = _load_file(path)  # keep raw for minute import
                # extract date from filename
                fdate = _date_from_filename(path.name)
                if fdate is not None:
                    info.date_min = fdate
                    info.date_max = fdate
                # extract stock list from instrument index
                if info.df.index.names and "instrument" in info.df.index.names:
                    inst = info.df.index.get_level_values("instrument").unique()
                    info.stocks = sorted({str(int(s)) for s in inst if pd.notna(s)})
            return info

        if info.file_type == "cross_sectional_factor":
            info.factor_name = path.stem
            info.date_source = "截面因子索引"
            # load to check date range and stock list
            if load:
                info.df = _load_file(path)
                if isinstance(info.df.index, pd.DatetimeIndex):
                    info.date_min = info.df.index.min()
                    info.date_max = info.df.index.max()
                info.stocks = sorted({str(int(c)) for c in info.df.columns if re.match(r"^\d+$", str(c))})
            return info

        if info.file_type == "daily_data":
            info.date_source = "dailyData.parquet"

        # Standard / daily_data: use existing logic
        data_cols = [c for c in columns if c not in SYMBOL_COL_CANDIDATES and c not in DATE_COL_CANDIDATES]
        info.market_cols, info.fund_cols, info.new_cols, info.new_target = _classify(data_cols, hint)

        info.symbol_col = next((c for c in SYMBOL_COL_CANDIDATES if c in columns), None)
        if next((c for c in DATE_COL_CANDIDATES if c in columns), None):
            info.date_source = "列"
        elif "datetime" in columns:
            info.date_source = "索引(datetime)"
        elif _date_from_filename(path.name) is not None:
            info.date_source = "文件名日期"
        else:
            info.date_source = "自动检测"

        if load:
            df = _load_file(path)
            data_df, codes, symbol_col, date_source = _prepare_frame(df, path)
            info.df = data_df
            info.codes = codes
            info.symbol_col = symbol_col
            info.date_source = date_source
            stocks = sorted({s for s in codes.unique() if s})
            info.stocks = stocks
            if not data_df.empty:
                info.date_min = data_df.index.min()
                info.date_max = data_df.index.max()
    except Exception as e:
        info.error = str(e)
    return info


# ── Barra 模型导入 ──
def _import_barra_model(info: FileInfo) -> None:
    """把 Barra 模型文件复制到 数据仓库/barra_model/。"""
    BARRA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = BARRA_MODEL_DIR / info.path.name
    size_mb = info.path.stat().st_size / 1024 / 1024
    shutil.copy2(info.path, dst)
    print(f"  ✅ Barra 模型: {info.path.name} ({size_mb:.1f} MB) → {dst.relative_to(PROJECT_ROOT)}", flush=True)


# ── 合并导入（标准日线数据） ──
def _merge_stock(dst: Path, df: pd.DataFrame):
    """把一只股票的新数据合并进目标 per-stock parquet（按日期去重，新数据优先）。"""
    df = df.sort_index()
    if dst.exists():
        old = pd.read_parquet(dst)
        combined = pd.concat([old, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        combined.to_parquet(dst)
    else:
        df.to_parquet(dst)


def _copy_new_col_to_test(full_dir: Path, test_dir: Path, col: str, test_stocks: list) -> int:
    """把全量 parquet 的新列拷进测试 parquet（对齐固定测试交易日，只补列）。"""
    n = 0
    for stock in test_stocks:
        full_f = full_dir / f"{stock}.parquet"
        test_f = test_dir / f"{stock}.parquet"
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
        test.to_parquet(test_f)
        n += 1
    return n


# ── 分钟数据导入 ──
def _import_minute_file(info: FileInfo, full_dates: set, full_stocks: set) -> tuple:
    """导入分钟 per-date 文件。返回 (更新日期数, 更新股票数)。"""
    fdate = _date_from_filename(info.path.name)
    if fdate is None:
        print(f"  ⚠️ 无法从文件名识别日期: {info.path.name}")
        return 0, 0

    date_str = fdate.strftime("%Y%m%d")
    full_dates.add(date_str)

    # 1. 复制到 minute_by_date 全量目录
    dst = FULL_MINUTE_BY_DATE / info.path.name
    info.df.to_parquet(dst)
    print(f"  📅 minute_by_date: {date_str} ({len(info.df)} 行)", flush=True)

    # 2. 更新 per-stock minute 数据
    df = info.df
    stock_count = 0
    if df.index.names and "instrument" in df.index.names:
        instruments = df.index.get_level_values("instrument").unique()
        for inst in instruments:
            stock = str(int(inst))
            full_stocks.add(stock)
            sub = df.xs(inst, level="instrument").copy()
            sub.index.name = "datetime"
            _merge_stock(FULL_MINUTE_DIR / f"{stock}.parquet", sub)
            stock_count += 1

    return 1, stock_count


def _update_minute_meta(full_dates: set, full_stocks: set):
    """更新分钟数据 trade_dates.json 和 stock_list.json。"""
    old_dates = set(_load_json_list(FULL_MINUTE_TRADE_DATES))
    if sorted(full_dates) != sorted(old_dates):
        _write_json_list(FULL_MINUTE_TRADE_DATES, sorted(full_dates))
        print(f"  📅 分钟 trade_dates.json: {len(old_dates)} → {len(full_dates)} 天", flush=True)

    old_stocks = set(_load_json_list(FULL_MINUTE_STOCK_LIST))
    if sorted(full_stocks) != sorted(old_stocks):
        _write_json_list(FULL_MINUTE_STOCK_LIST, sorted(full_stocks))
        print(f"  📈 分钟 stock_list.json: {len(old_stocks)} → {len(full_stocks)} 只", flush=True)


# ── 截面因子导入 ──
def _import_cross_sectional_factor(info: FileInfo, descs: dict) -> list:
    """导入截面因子 parquet 到非行情 per-stock。返回新列名列表。"""
    factor_name = info.factor_name
    if factor_name is None:
        return []

    # 获取描述
    desc = descs.get(factor_name, "基本面因子说明.csv 未提供说明")

    df = info.df  # index=DatetimeIndex, columns=stock codes(int), values=float64
    # 转长格式
    long_df = df.stack().to_frame(name=factor_name)
    long_df.index.names = ["datetime", "stock"]
    long_df = long_df.reset_index(level="stock")
    long_df["stock"] = long_df["stock"].apply(lambda x: str(int(x)))

    # 按股票合并进全量非行情 per-stock
    stock_count = 0
    for stock, group in long_df.groupby("stock"):
        sub = group[[factor_name]].copy()
        sub.index.name = "datetime"
        _merge_stock(FULL_FUND_DAILY / f"{stock}.parquet", sub)
        stock_count += 1

    print(f"  截面因子 [{factor_name}]: {desc} — 合并 {stock_count} 只股票", flush=True)
    return [factor_name]


# ── 检查 / dry-run 输出 ──
def _type_label(ft: str) -> str:
    return {
        "standard": "标准",
        "daily_data": "dailyData",
        "minute_by_date": "分钟 per-date",
        "cross_sectional_factor": "截面因子",
        "barra_model": "Barra模型",
    }.get(ft, ft)


def check_new_data():
    files = scan_new_data_dir()
    if not files:
        print("新建文件/ 下没有可导入的文件（parquet/csv）")
        note = DESC_FILE
        if note.exists():
            print(f"提示: 可读 {DESC_FILE} 了解新列含义")
        descs = _load_descriptions()
        if descs:
            print(f"  基本面因子说明.csv 已注册 {len(descs)} 个因子描述")
        return
    print(f"扫描到 {len(files)} 个文件:")
    for path, hint in files:
        info = inspect_file(path, hint, load=False)
        rel = path.relative_to(PROJECT_ROOT)
        hint_txt = {"market": "日线", "fundamental": "非行情", "minute": "分钟线"}.get(hint, "根目录")
        type_txt = _type_label(info.file_type)
        print(f"\n  {rel}  [{hint_txt}] [{type_txt}]  ({path.stat().st_size / 1024:.0f} KB)")
        if info.error:
            print(f"    ❌ {info.error}")
            continue
        print(f"    行数: {info.row_count}")
        if info.file_type == "barra_model":
            print(f"    目标: 数据仓库/barra_model/")
            continue
        if info.file_type == "minute_by_date":
            fdate = _date_from_filename(path.name)
            print(f"    日期: {fdate.strftime('%Y%m%d') if fdate else '?'}")
            continue
        if info.file_type == "cross_sectional_factor":
            print(f"    因子名: {info.factor_name}")
            continue
        print(f"    股票列: {info.symbol_col or '无（按文件名识别）'} | 日期: {info.date_source}")
        print(f"    行情列({len(info.market_cols)}): {info.market_cols or '-'}")
        print(f"    非行情列({len(info.fund_cols)}): {info.fund_cols or '-'}")
        new_txt = f" → {'行情' if info.new_target == 'market' else '非行情'}" if info.new_cols else ""
        print(f"    新列({len(info.new_cols)}): {info.new_cols or '-'}{new_txt}")
    print("\n✅ 检查完成。运行 `python3 scripts/import_new_data.py` 导入。")


def print_plan(inspections: list):
    print("\n=== 将要导入的内容（dry-run） ===")
    for info in inspections:
        rel = info.path.relative_to(PROJECT_ROOT)
        if info.error:
            print(f"  ❌ {rel}: {info.error}")
            continue
        type_txt = _type_label(info.file_type)
        print(f"\n  {rel}  [{info.hint or '根目录'}] [{type_txt}]")
        if info.file_type == "barra_model":
            print(f"    复制到: 数据仓库/barra_model/ ({info.path.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        if info.file_type == "minute_by_date":
            fdate = _date_from_filename(info.path.name)
            print(f"    日期: {fdate.strftime('%Y%m%d') if fdate else '?'}")
            print(f"    股票数: {len(info.stocks)}")
            continue
        if info.file_type == "cross_sectional_factor":
            print(f"    因子名: {info.factor_name}")
            print(f"    日期范围: {info.date_min} ~ {info.date_max}")
            print(f"    股票数: {len(info.stocks)}")
            continue
        print(f"    股票数: {len(info.stocks)} | 日期: {info.date_min} ~ {info.date_max}")
        print(f"    行情列({len(info.market_cols)}): {info.market_cols or '-'}")
        print(f"    非行情列({len(info.fund_cols)}): {info.fund_cols or '-'}")
        new_txt = f" → {'行情' if info.new_target == 'market' else '非行情'}" if info.new_cols else ""
        print(f"    新列({len(info.new_cols)}): {info.new_cols or '-'}{new_txt}")


# ── 注册新列 ──
def _register_new_cols(schema: dict, cols_encountered: set, write: bool,
                       descs: Optional[dict] = None) -> list:
    """把遇到的数据列注册进 schema.json + factor_field_schema.json，返回真正的新列。"""
    schema_cols = schema["daily"]["columns"]
    descs = descs or {}
    truly_new = []
    for c in sorted(cols_encountered):
        if c not in schema_cols:
            schema_cols[c] = {"description": _known_desc(c), "source": "新建文件导入"}
            print(f"  📋 注册 schema 列: {c}", flush=True)
        for ffp in FACTOR_FIELD_SCHEMA_PATHS:
            ff = _load_factor_field_schema(ffp)
            if c not in ff:
                short_name = descs.get(c, c)
                note = descs.get(c, "字段含义待补充（见 基本面因子说明.csv）")
                ff[c] = {
                    "factor_name": c, "short_name": short_name, "formula": "",
                    "source": "新建文件导入", "note": note,
                }
                print(f"  📋 注册 factor_field_schema: {ffp.parent.name}/{c} → {short_name}", flush=True)
                if write:
                    _save_factor_field_schema(ffp, ff)
        if c not in MARKET_COLS and c not in FUNDAMENTAL_COLS:
            truly_new.append(c)
    if write:
        _save_schema(schema)
    return truly_new


# ── 主导入流程 ──
def do_import(dry_run: bool = False):
    files = scan_new_data_dir()
    if not files:
        print("新建文件/ 下没有可导入的文件（parquet/csv）")
        descs = _load_descriptions()
        if descs:
            print(f"  基本面因子说明.csv 已注册 {len(descs)} 个因子描述，但无数据文件需要导入。")
        return
    if not FULL_MARKET_DAILY.exists():
        print(f"❌ 全量行情目录不存在: {FULL_MARKET_DAILY}")
        return

    descs = _load_descriptions()
    schema = _load_schema()
    full_dates = set(_load_json_list(FULL_TRADE_DATES_FILE))
    full_stocks = set(_load_json_list(FULL_STOCK_LIST_FILE))
    test_stocks = set(_load_json_list(TEST_STOCK_LIST_FILE))

    inspections = [inspect_file(path, hint, load=True) for path, hint in files]
    if dry_run:
        print_plan(inspections)
        return

    all_dates = set(full_dates)
    all_stocks = set(full_stocks)
    cols_encountered = set()
    new_target = {}
    n_stock_updated = 0

    # 分钟数据：收集全量 meta
    minute_dates = set(_load_json_list(FULL_MINUTE_TRADE_DATES))
    minute_stocks = set(_load_json_list(FULL_MINUTE_STOCK_LIST))
    has_minute_data = False

    for info in inspections:
        rel = info.path.relative_to(PROJECT_ROOT)
        print(f"\n=== 导入: {rel} [{_type_label(info.file_type)}] ===", flush=True)
        if info.error:
            print(f"  ❌ 跳过（{info.error}）")
            continue

        # ── Barra 风险模型 ──
        if info.file_type == "barra_model":
            _import_barra_model(info)
            continue

        # ── 分钟 per-date ──
        if info.file_type == "minute_by_date":
            nd, ns = _import_minute_file(info, minute_dates, minute_stocks)
            has_minute_data = True
            n_stock_updated += ns
            continue

        # ── 截面因子 ──
        if info.file_type == "cross_sectional_factor":
            new_cols = _import_cross_sectional_factor(info, descs)
            cols_encountered.update(new_cols)
            for c in new_cols:
                new_target[c] = "fundamental"
            continue

        # ── 标准 / dailyData 日线数据 ──
        for c in info.new_cols:
            new_target[c] = info.new_target
        cols_encountered.update(info.market_cols + info.fund_cols + info.new_cols)

        mcols = info.market_cols + [c for c in info.new_cols if info.new_target == "market"]
        fcols = info.fund_cols + [c for c in info.new_cols if info.new_target == "fundamental"]
        n_merge = 0
        for stock in info.stocks:
            sub = info.df[info.codes == stock]
            if sub.empty:
                continue
            all_stocks.add(stock)
            for dt in sub.index:
                all_dates.add(pd.Timestamp(dt).strftime("%Y%m%d"))
            m_df = sub[[c for c in mcols if c in sub.columns]]
            f_df = sub[[c for c in fcols if c in sub.columns]]
            if not m_df.empty:
                _merge_stock(FULL_MARKET_DAILY / f"{stock}.parquet", m_df)
                n_merge += 1
            if not f_df.empty:
                _merge_stock(FULL_FUND_DAILY / f"{stock}.parquet", f_df)
                n_merge += 1
        print(f"  合并 {len(info.stocks)} 只股票 / {n_merge} 次写入", flush=True)
        n_stock_updated += len(info.stocks)

    # 注册新列（含基本面因子说明.csv 信息）
    truly_new = _register_new_cols(schema, cols_encountered, write=True, descs=descs)

    # 更新全量日线元数据（并集）
    if sorted(all_dates) != sorted(full_dates):
        _write_json_list(FULL_TRADE_DATES_FILE, sorted(all_dates))
        print(f"\n📅 日线 trade_dates.json: {len(full_dates)} → {len(all_dates)} 天", flush=True)
    if sorted(all_stocks) != sorted(full_stocks):
        _write_json_list(FULL_STOCK_LIST_FILE, sorted(all_stocks))
        print(f"📈 日线 stock_list.json: {len(full_stocks)} → {len(all_stocks)} 只", flush=True)

    # 更新分钟元数据
    if has_minute_data:
        _update_minute_meta(minute_dates, minute_stocks)

    # 测试数据固定 300 天：只补本次新列
    truly_new = sorted(set(truly_new))
    if truly_new and test_stocks:
        print("\n=== 测试数据补列（固定 300 天，只补新列，不加日期/股票） ===", flush=True)
        for col in truly_new:
            tgt = new_target.get(col, "market")
            if tgt == "market":
                n = _copy_new_col_to_test(FULL_MARKET_DAILY, TEST_MARKET_DAILY, col, sorted(test_stocks))
            else:
                n = _copy_new_col_to_test(FULL_FUND_DAILY, TEST_FUND_DAILY, col, sorted(test_stocks))
            print(f"  {col}: {n} 只测试股票已补列", flush=True)
    elif truly_new:
        print("\n⚠️ 测试 stock_list.json 为空，跳过测试数据补列")

    print(f"\n✅ 导入完成: {n_stock_updated} 只股票更新", flush=True)
    if truly_new:
        print(f"⚠️ NEW_COLUMNS_DETECTED: {truly_new}")
        print("Agent: 读 新建文件/基本面因子说明.csv 理解新列含义 → 更新 data/schema.json + "
              "两个 factor_field_schema.json → 运行 --update-prompts-only")
    else:
        print("无新列。如需刷新 prompt 标记块: python3 scripts/import_new_data.py --update-prompts-only")


# ── prompt 标记块更新（复用旧 sync_data.py 逻辑） ──
def update_prompt_files(schema: dict):
    daily_cols = schema["daily"]["columns"]
    minute_cols = schema["minute"]["columns"]
    ff_schema = _load_factor_field_schema(FACTOR_FIELD_SCHEMA_PATHS[0])

    def _fmt_line(col: str, src: dict) -> str:
        ff = ff_schema.get(col)
        if ff and ff.get("short_name") and ff["short_name"] != col:
            parts = [f"  - {col}: {ff['short_name']}"]
            note = ff.get("note", "")
            if note:
                parts.append(f"（{note}）")
            return "".join(parts)
        desc = src.get("description", col)
        if desc and desc != col:
            return f"  - {col}: {desc}"
        return f"  - {col}"

    daily_lines = [_fmt_line(k, v) for k, v in sorted(daily_cols.items())]
    minute_lines = [_fmt_line(k, v) for k, v in sorted(minute_cols.items())]
    daily_text = "\n".join(daily_lines)
    minute_text = "\n".join(minute_lines)

    updates = {
        "rdagent/components/coder/factor_coder/prompts.yaml": [
            ("<!-- DAILY_COLUMNS -->", "<!-- /DAILY_COLUMNS -->", daily_text),
            ("<!-- MINUTE_COLUMNS -->", "<!-- /MINUTE_COLUMNS -->", minute_text),
        ],
        ".claude/skills/factor/knowledge/daily.md": [
            ("<!-- DAILY_COLUMNS -->", "<!-- /DAILY_COLUMNS -->", daily_text),
        ],
        ".claude/skills/factor/knowledge/cross_section.md": [
            ("<!-- DAILY_COLUMNS -->", "<!-- /DAILY_COLUMNS -->", daily_text),
        ],
        ".claude/skills/factor/knowledge/deep_learning.md": [
            ("<!-- DAILY_COLUMNS -->", "<!-- /DAILY_COLUMNS -->", daily_text),
        ],
        ".claude/skills/factor/knowledge/minute.md": [
            ("<!-- MINUTE_COLUMNS -->", "<!-- /MINUTE_COLUMNS -->", minute_text),
        ],
        ".claude/skills/factor/knowledge/minute_cs.md": [
            ("<!-- MINUTE_COLUMNS -->", "<!-- /MINUTE_COLUMNS -->", minute_text),
        ],
    }

    for rel_path, markers in updates.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            print(f"  ⚠️ 文件不存在: {rel_path}", flush=True)
            continue
        content = full_path.read_text(encoding="utf-8")
        changed = False
        for start_marker, end_marker, new_text in markers:
            pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
            replacement = f"{start_marker}\n{new_text}\n{end_marker}"
            new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
            if count > 0:
                content = new_content
                changed = True
        if changed:
            full_path.write_text(content, encoding="utf-8")
            print(f"  ✅ 已更新: {rel_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="从本地 新建文件/ 目录导入新增行情/非行情数据")
    parser.add_argument("--check", action="store_true", help="扫描 新建文件/，预览文件与列 schema")
    parser.add_argument("--dry-run", action="store_true", help="打印将要导入的内容，不执行")
    parser.add_argument("--update-prompts-only", action="store_true", help="仅根据 schema.json 更新 prompt 标记块")
    args = parser.parse_args()

    if args.update_prompts_only:
        update_prompt_files(_load_schema())
        print("\n✅ prompt 标记块更新完成")
        return
    if args.check:
        check_new_data()
        return
    if args.dry_run:
        do_import(dry_run=True)
        return
    do_import(dry_run=False)


if __name__ == "__main__":
    main()