#!/usr/bin/env python3
"""Convert /mnt/remote_e/ market data into the H5 format used by paper-factor.

Reads:
  - Daily parquet from market_daily_daily_new/ (per-date files)
  - Minute parquet from market_minute_daily_new/ (per-date files)
  - Fundamental factor CSVs from 基本面因子/ (9 categories, 114 files)

Outputs:
  - git_ignore_folder/factor_implementation_source_data/     (full dataset)
  - git_ignore_folder/factor_implementation_source_data_debug/ (compact debug subset)
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REMOTE_DATA_ROOT = Path("/mnt/remote_e")
DAILY_DIR = REMOTE_DATA_ROOT / "market_daily_daily_new"
MINUTE_DIR = REMOTE_DATA_ROOT / "market_minute_daily_new"
FUNDAMENTAL_DIR = REMOTE_DATA_ROOT / "基本面因子"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_DATA_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"
DEBUG_DATA_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data_debug"

# Date range: last ~2 years
START_DATE = "2024-05-01"
END_DATE = "2026-05-20"

# Subset sizes for the full and debug bundles
FULL_MAX_DAYS = 252
FULL_MAX_INSTRUMENTS = 80
DEBUG_MAX_DAYS = 60
DEBUG_MAX_INSTRUMENTS = 20

META_FILENAME = "remote_data_meta.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _list_parquet_dates(directory: Path, start: str, end: str) -> list[str]:
    """Return sorted date strings (YYYYMMDD) of parquet files within [start, end]."""
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")
    dates = []
    for f in sorted(directory.glob("*.parquet")):
        d = f.stem
        if start_compact <= d <= end_compact:
            dates.append(d)
    return dates


def _to_multiindex(df: pd.DataFrame, date_col: str, symbol_col: str) -> pd.DataFrame:
    """Convert raw DataFrame to MultiIndex ['datetime', 'instrument'] with $-prefixed columns."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out[date_col].astype(str) if out[date_col].dtype == object else out[date_col])
    out["instrument"] = out[symbol_col].astype(str)
    out = out.drop(columns=[c for c in (symbol_col, date_col) if c in out.columns and c not in ("datetime", "instrument")])
    return out.set_index(["datetime", "instrument"]).sort_index()


# ---------------------------------------------------------------------------
# Daily conversion (memory-efficient: process file by file, keep only needed)
# ---------------------------------------------------------------------------
def convert_daily_data(instruments: set[str] | None = None) -> pd.DataFrame:
    """Read daily parquet files and return a MultiIndex DataFrame.

    If *instruments* is given, only rows whose symbol is in that set are kept.
    """
    dates = _list_parquet_dates(DAILY_DIR, START_DATE, END_DATE)
    if not dates:
        raise FileNotFoundError(f"No daily parquet files found in {DAILY_DIR} for {START_DATE}..{END_DATE}")

    print(f"  Reading {len(dates)} daily parquet files ...")
    frames: list[pd.DataFrame] = []
    for i, d in enumerate(dates):
        path = DAILY_DIR / f"{d}.parquet"
        df = pd.read_parquet(path)
        if instruments is not None:
            df = df[df["symbol"].astype(str).isin(instruments)]
        if df.empty:
            continue
        frames.append(df)
        if (i + 1) % 100 == 0:
            print(f"    ... read {i + 1}/{len(dates)} files")
            gc.collect()

    raw = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  Total daily rows: {len(raw):,}")

    # Rename and prefix columns, drop computed EMAs
    raw["datetime"] = pd.to_datetime(raw["date"].astype(str))
    raw["instrument"] = raw["symbol"].astype(str)
    raw = raw.drop(columns=["symbol", "date", "EMA5", "EMA10", "EMA20"])

    col_map = {
        "open": "$open",
        "close": "$close",
        "high": "$high",
        "low": "$low",
        "volume": "$volume",
        "factor": "$factor",
    }
    raw = raw.rename(columns=col_map)

    raw = raw.set_index(["datetime", "instrument"]).sort_index()

    # Compute derived columns
    # $pre_close: previous trading day's close per instrument
    raw["$pre_close"] = raw.groupby("instrument")["$close"].shift(1)
    # $pct_chg: daily return using factor-adjusted prices
    prev_adj = raw.groupby("instrument")["$close"].shift(1) * (
        raw.groupby("instrument")["$factor"].shift(1) / raw["$factor"]
    )
    raw["$pct_chg"] = ((raw["$close"] - prev_adj) / prev_adj).replace([np.inf, -np.inf], np.nan)

    return raw


# ---------------------------------------------------------------------------
# Minute conversion (memory-efficient: filter by instruments per file)
# ---------------------------------------------------------------------------
def convert_minute_data(instruments: set[str]) -> pd.DataFrame:
    """Read minute parquet files, keeping only rows for *instruments*."""
    dates = _list_parquet_dates(MINUTE_DIR, START_DATE, END_DATE)
    if not dates:
        raise FileNotFoundError(f"No minute parquet files found in {MINUTE_DIR} for {START_DATE}..{END_DATE}")

    print(f"  Reading {len(dates)} minute parquet files (filtered to {len(instruments)} instruments) ...")
    frames: list[pd.DataFrame] = []
    for i, d in enumerate(dates):
        path = MINUTE_DIR / f"{d}.parquet"
        df = pd.read_parquet(path, filters=[("symbol", "in", [int(s) for s in instruments if s.isdigit()])])
        if df.empty:
            continue
        frames.append(df)
        if (i + 1) % 100 == 0:
            print(f"    ... read {i + 1}/{len(dates)} files")
            gc.collect()

    if not frames:
        raise RuntimeError("No minute data loaded after filtering.")

    raw = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  Total minute rows: {len(raw):,}")

    raw["datetime"] = pd.to_datetime(raw["trade_date"])
    raw["instrument"] = raw["symbol"].astype(str)
    raw = raw.drop(columns=["symbol", "trade_date", "date"])

    col_map = {
        "open": "$open",
        "close": "$close",
        "high": "$high",
        "low": "$low",
        "volume": "$volume",
        "factor": "$factor",
        "return": "$return",
    }
    raw = raw.rename(columns=col_map)

    # Compute VWAP as typical price (no amount data available)
    raw["$vwap"] = (raw["$open"] + raw["$high"] + raw["$low"] + raw["$close"]) / 4.0

    raw = raw.set_index(["datetime", "instrument"]).sort_index()
    return raw


# ---------------------------------------------------------------------------
# Fundamental factor conversion (memory-efficient)
# ---------------------------------------------------------------------------
def convert_fundamental_factors(instruments: set[str] | None = None) -> pd.DataFrame:
    """Read all fundamental factor CSVs and return a DataFrame with MultiIndex.

    If *instruments* is given, only those symbol columns are kept.
    Returns MultiIndex ['datetime', 'instrument'] with columns like '$价值因子1'.

    Memory-efficient: reads each CSV, filters to target instruments/dates,
    stacks to long format, then joins column-by-column.
    """
    start_compact = START_DATE.replace("-", "")
    end_compact = END_DATE.replace("-", "")

    categories = sorted(
        d for d in os.listdir(FUNDAMENTAL_DIR)
        if os.path.isdir(os.path.join(FUNDAMENTAL_DIR, d)) and d != "log_record"
    )

    # First pass: read one CSV to determine the MultiIndex template
    # (all CSVs share the same trade_date × symbol grid)
    first_cat_dir = FUNDAMENTAL_DIR / categories[0]
    first_csv = sorted(f for f in os.listdir(first_cat_dir) if f.endswith(".csv"))[0]
    template = pd.read_csv(first_cat_dir / first_csv, usecols=["trade_date"])
    dates_mask = template["trade_date"].str.replace("-", "")
    dates_mask = (dates_mask >= start_compact) & (dates_mask <= end_compact)
    template_dates = template.loc[dates_mask, "trade_date"]
    n_dates = len(template_dates)
    print(f"  Date range: {n_dates} trading days")

    total_files = 0
    result: pd.DataFrame | None = None

    for cat in categories:
        cat_dir = FUNDAMENTAL_DIR / cat
        csv_files = sorted(f for f in os.listdir(cat_dir) if f.endswith(".csv"))
        print(f"  Processing category '{cat}': {len(csv_files)} files")

        for csv_file in csv_files:
            factor_name = csv_file.replace(".csv", "")
            col_name = f"${factor_name}"

            path = cat_dir / csv_file

            # Read only needed columns
            header_cols = pd.read_csv(path, nrows=0).columns.tolist()
            if instruments is not None:
                usecols = ["trade_date"] + [c for c in header_cols if c != "trade_date" and c in instruments]
            else:
                usecols = header_cols

            df = pd.read_csv(path, usecols=usecols)

            # Filter by date range
            dates = df["trade_date"].str.replace("-", "")
            mask = (dates >= start_compact) & (dates <= end_compact)
            df = df.loc[mask].copy()
            if df.empty:
                continue

            # Set trade_date as index, then stack to long format
            df = df.set_index("trade_date")
            df.index = pd.to_datetime(df.index)
            df.index.name = "datetime"
            # Stack: (datetime, instrument) -> value
            stacked = df.stack()
            stacked.index.names = ["datetime", "instrument"]
            stacked.name = col_name
            stacked.index = stacked.index.set_levels(
                stacked.index.levels[1].astype(str), level="instrument"
            )

            if result is None:
                result = stacked.to_frame()
            else:
                result[col_name] = stacked

            total_files += 1
            if total_files % 20 == 0:
                print(f"    ... loaded {total_files} factors, shape: {result.shape}")
                gc.collect()

    print(f"  Loaded {total_files} fundamental factor files")

    if result is None:
        return pd.DataFrame()

    print(f"  Fundamental factors shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Merge and write
# ---------------------------------------------------------------------------
def merge_daily_with_fundamentals(daily_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily OHLCV data with fundamental factors, then compute turnover_rate."""
    if fund_df.empty:
        return daily_df
    print("  Merging daily data with fundamental factors ...")
    merged = daily_df.join(fund_df, how="left")
    print(f"  Merged shape: {merged.shape}")

    # Compute real turnover_rate = volume / (流通股本 * 10000)
    # 股本因子2 = 流通股本(万股), volume is in shares
    float_share_col = "$股本因子2"
    if float_share_col in merged.columns:
        float_shares = pd.to_numeric(merged[float_share_col], errors="coerce").replace(0, np.nan)
        vol = pd.to_numeric(merged["$volume"], errors="coerce").fillna(0.0)
        merged["$turnover_rate"] = (vol / (float_shares * 10000)).replace([np.inf, -np.inf], np.nan)
        print(f"  Computed $turnover_rate from volume / 流通股本")
    else:
        # Fallback: proxy turnover_rate
        vol = pd.to_numeric(merged["$volume"], errors="coerce").fillna(0.0)
        scale = vol.groupby(level="datetime").transform("median").replace(0, np.nan)
        merged["$turnover_rate"] = (vol / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        print(f"  Warning: $股本因子2 not found, using proxy turnover_rate")

    return merged


def _build_subset(df: pd.DataFrame, max_days: int, max_instruments: int) -> pd.DataFrame:
    """Slice to a compact subset: last N days, first M instruments."""
    if df.empty:
        return df
    instruments = list(dict.fromkeys(df.index.get_level_values("instrument")))
    chosen_instruments = instruments[:max_instruments]
    mask = df.index.get_level_values("instrument").isin(chosen_instruments)
    return df.loc[mask].sort_index()


def _select_instruments_from_daily(daily_df: pd.DataFrame, max_instruments: int) -> set[str]:
    """Pick the first N instruments from the daily data."""
    instruments = list(dict.fromkeys(daily_df.index.get_level_values("instrument")))
    return set(instruments[:max_instruments])


def write_h5_bundle(
    folder: Path,
    daily: pd.DataFrame,
    minute: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    """Write daily and minute H5 files plus metadata to a folder."""
    folder.mkdir(parents=True, exist_ok=True)

    daily.to_hdf(folder / "daily_pv.h5", key="data")
    minute.to_hdf(folder / "minute_pv.h5", key="data")

    legacy = folder / "minute_quote.h5"
    if legacy.exists():
        legacy.unlink()

    (folder / "README.md").write_text(
        "# Factor source data\n\n"
        "Generated from /mnt/remote_e/ market data.\n\n"
        f"- `daily_pv.h5`: MultiIndex ['datetime', 'instrument']; "
        f"{daily.shape[0]:,} rows, {daily.shape[1]} columns\n"
        f"- `minute_pv.h5`: MultiIndex ['datetime', 'instrument']; "
        f"{minute.shape[0]:,} rows, {minute.shape[1]} columns\n",
        encoding="utf-8",
    )
    (folder / META_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Converting /mnt/remote_e/ data to paper-factor H5 format")
    print("=" * 60)

    # 1. Convert daily data (all instruments needed for universe selection)
    print("\n[1/5] Loading daily market data (all instruments) ...")
    daily_full = convert_daily_data()
    n_instruments = daily_full.index.get_level_values("instrument").nunique()
    print(f"  Daily data: {daily_full.shape[0]:,} rows, {daily_full.shape[1]} columns")
    print(f"  Date range: {daily_full.index.get_level_values('datetime').min()} .. "
          f"{daily_full.index.get_level_values('datetime').max()}")
    print(f"  Instruments: {n_instruments:,}")

    # 2. Select instrument universe for full and debug bundles
    print("\n[2/5] Selecting instrument universes ...")
    full_instruments = _select_instruments_from_daily(daily_full, FULL_MAX_INSTRUMENTS)
    debug_instruments = _select_instruments_from_daily(daily_full, DEBUG_MAX_INSTRUMENTS)
    print(f"  Full bundle: {len(full_instruments)} instruments")
    print(f"  Debug bundle: {len(debug_instruments)} instruments")

    # 3. Convert fundamental factors (only for full instruments)
    print("\n[3/5] Converting fundamental factors ...")
    fund_df = convert_fundamental_factors(instruments=full_instruments)

    # 4. Merge daily with fundamentals, then build subsets
    print("\n[4/5] Building daily datasets ...")
    daily_merged = merge_daily_with_fundamentals(daily_full, fund_df)
    del fund_df
    gc.collect()

    full_daily = _build_subset(daily_merged, FULL_MAX_DAYS, FULL_MAX_INSTRUMENTS)
    debug_daily = _build_subset(daily_merged, DEBUG_MAX_DAYS, DEBUG_MAX_INSTRUMENTS)
    del daily_merged, daily_full
    gc.collect()
    print(f"  Full daily: {full_daily.shape}")
    print(f"  Debug daily: {debug_daily.shape}")

    # 5. Convert minute data (only for full instruments, then subset)
    print("\n[5/5] Converting minute market data ...")
    minute_full = convert_minute_data(instruments=full_instruments)
    debug_minute = _build_subset(minute_full, DEBUG_MAX_DAYS, DEBUG_MAX_INSTRUMENTS)
    # Re-subset full minute to max days
    full_minute = _build_subset(minute_full, FULL_MAX_DAYS, FULL_MAX_INSTRUMENTS)
    del minute_full
    gc.collect()
    print(f"  Full minute: {full_minute.shape}")
    print(f"  Debug minute: {debug_minute.shape}")

    # Build metadata
    metadata = {
        "source": "remote_e",
        "updated_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily_rows": int(full_daily.shape[0]),
        "daily_columns": int(full_daily.shape[1]),
        "minute_rows": int(full_minute.shape[0]),
        "instruments_full": len(full_instruments),
        "instruments_debug": len(debug_instruments),
        "fundamental_factors": full_daily.shape[1] - 9,  # OHLCV + factor + pre_close + pct_chg + turnover_rate = 9 base columns
    }

    # Write bundles
    print(f"\nWriting full dataset to {FULL_DATA_DIR} ...")
    write_h5_bundle(FULL_DATA_DIR, full_daily, full_minute, metadata)

    print(f"Writing debug dataset to {DEBUG_DATA_DIR} ...")
    write_h5_bundle(DEBUG_DATA_DIR, debug_daily, debug_minute, metadata | {"debug": True})

    print("\n" + "=" * 60)
    print("Done! Data conversion complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
