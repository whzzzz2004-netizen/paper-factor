"""Utilities for loading locally mined factor pool files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from rdagent.log import rdagent_logger as logger

FACTOR_POOL_DIR = Path.cwd() / "git_ignore_folder" / "factor_outputs"
_SNAPSHOT_PREFIX_RE = re.compile(r"^\d{8}_\d{6}(?:_[0-9a-f]{8})?__?.+$")


def _is_snapshot_file(path: Path) -> bool:
    return _SNAPSHOT_PREFIX_RE.match(path.stem) is not None


def _infer_time_granularity(df: pd.DataFrame) -> str:
    if df.empty or "datetime" not in df.index.names:
        return "unknown"
    dt_index = pd.to_datetime(df.index.get_level_values("datetime"))
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


def _normalize_factor_df(path: Path, time_granularity: str | None = "daily") -> pd.DataFrame | None:
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df.empty:
        logger.warning(f"Skip empty factor pool file: {path}")
        return None
    if not isinstance(df.index, pd.MultiIndex) or set(["datetime", "instrument"]) - set(df.index.names):
        logger.warning(f"Skip factor pool file with invalid index: {path}; index_names={list(df.index.names)}")
        return None
    if df.shape[1] != 1:
        logger.warning(f"Skip factor pool file with {df.shape[1]} columns: {path}")
        return None
    if df.index.has_duplicates:
        logger.warning(f"Factor pool file has duplicated index values; keeping last value: {path}")
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    if time_granularity is not None:
        actual_granularity = _infer_time_granularity(df)
        if actual_granularity != time_granularity:
            logger.warning(
                f"Skip factor pool file with time granularity {actual_granularity}: {path}; "
                f"expected={time_granularity}"
            )
            return None
    return df


def load_factor_pool(
    factor_pool_path: str | Path | None = None,
    factor_top_k: int | None = None,
    factor_offset: int = 0,
    time_granularity: str | None = "daily",
) -> pd.DataFrame:
    """Load latest factor parquet files from the factor pool into a Qlib StaticDataLoader dataframe."""
    pool_dir = Path(factor_pool_path) if factor_pool_path is not None else FACTOR_POOL_DIR
    if not pool_dir.exists():
        raise FileNotFoundError(f"Factor pool directory does not exist: {pool_dir}")

    parquet_files = sorted(p for p in pool_dir.glob("*.parquet") if not _is_snapshot_file(p))
    if factor_offset:
        parquet_files = parquet_files[factor_offset:]
    if factor_top_k is not None:
        parquet_files = parquet_files[:factor_top_k]
    if not parquet_files:
        raise FileNotFoundError(f"No latest factor parquet files found under {pool_dir}")

    factor_dfs: list[pd.DataFrame] = []
    for path in parquet_files:
        df = _normalize_factor_df(path, time_granularity=time_granularity)
        if df is not None:
            factor_dfs.append(df)

    if not factor_dfs:
        raise ValueError(f"No valid factor parquet files found under {pool_dir}")

    combined = pd.concat(factor_dfs, axis=1, join="inner").sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    combined = combined.dropna(how="all")
    combined.columns = pd.MultiIndex.from_product([["feature"], combined.columns.astype(str)])
    logger.info(f"Loaded {len(combined.columns)} factors from {pool_dir}; rows={len(combined)}")
    return combined
