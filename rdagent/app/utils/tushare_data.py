from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd()
FACTOR_DATA_DIR = ROOT / "git_ignore_folder" / "factor_implementation_source_data"
FACTOR_DATA_DEBUG_DIR = ROOT / "git_ignore_folder" / "factor_implementation_source_data_debug"
META_FILENAME = "tushare_data_meta.json"
PROFILE_META_FILENAME = "jq_data_meta.json"


@dataclass(frozen=True)
class TushareUpdateConfig:
    token: str
    years: int = 1
    max_stocks: int = 80
    max_futures: int = 20
    max_minute_symbols: int = 10
    debug_days: int = 60
    debug_instruments: int = 20
    minute_freq: str = "1min"
    request_sleep: float = 0.5


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(int(raw), 0)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return default


def config_from_env() -> TushareUpdateConfig | None:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        return None
    return TushareUpdateConfig(
        token=token,
        years=_env_int("TUSHARE_YEARS", 1),
        max_stocks=_env_int("TUSHARE_MAX_STOCKS", 80),
        max_futures=_env_int("TUSHARE_MAX_FUTURES", 20),
        max_minute_symbols=_env_int("TUSHARE_MAX_MINUTE_SYMBOLS", 10),
        debug_days=_env_int("TUSHARE_DEBUG_DAYS", 60),
        debug_instruments=_env_int("TUSHARE_DEBUG_INSTRUMENTS", 20),
        minute_freq=os.environ.get("TUSHARE_MINUTE_FREQ", "1min").strip() or "1min",
        request_sleep=_env_float("TUSHARE_REQUEST_SLEEP", 0.5),
    )


def _date_bounds(years: int) -> tuple[str, str]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=max(years, 1))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _to_datetime_index(df: pd.DataFrame, date_col: str, instrument_col: str = "ts_code") -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out[date_col].astype(str))
    out["instrument"] = out[instrument_col].astype(str)
    out = out.drop(columns=[c for c in ("ts_code", "trade_date", "trade_time", "date") if c in out.columns])
    return out.set_index(["datetime", "instrument"]).sort_index()


def _prefix_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.rename(columns=mapping).copy()
    index_columns = {"ts_code", "trade_date", "trade_time", "date", "datetime", "instrument"}
    rename_extra = {
        col: f"${col}"
        for col in out.columns
        if col not in index_columns and not str(col).startswith("$")
    }
    return out.rename(columns=rename_extra)


def _call_with_retries(func: Any, *, sleep_s: float, retries: int = 3, **kwargs: Any) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            result = func(**kwargs)
            if result is None:
                return pd.DataFrame()
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if "频率超限" in str(exc) or "权限" in str(exc):
                raise
            if attempt + 1 < retries:
                time.sleep(max(sleep_s, 0.2) * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def _stock_universe(pro: Any, max_stocks: int, sleep_s: float) -> pd.DataFrame:
    fields = "ts_code,symbol,name,area,industry,market,list_date"
    df = _call_with_retries(
        pro.stock_basic,
        exchange="",
        list_status="L",
        fields=fields,
        sleep_s=sleep_s,
    )
    if df.empty:
        return df
    df = df.sort_values(["market", "ts_code"], kind="stable")
    return df.head(max_stocks).reset_index(drop=True)


def _future_universe(pro: Any, max_futures: int, sleep_s: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fields = "ts_code,symbol,name,exchange,list_date,delist_date"
    today = pd.Timestamp.today().strftime("%Y%m%d")
    for exchange in ("CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX"):
        try:
            df = _call_with_retries(pro.fut_basic, exchange=exchange, fields=fields, sleep_s=sleep_s)
        except Exception:  # noqa: BLE001
            continue
        if not df.empty:
            frames.append(df)
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    futures = pd.concat(frames, ignore_index=True)
    if "delist_date" in futures.columns:
        futures = futures[(futures["delist_date"].fillna("99999999").astype(str) >= today)]
    return futures.sort_values(["exchange", "ts_code"], kind="stable").head(max_futures).reset_index(drop=True)


def _fetch_stock_daily(pro: Any, universe: pd.DataFrame, start_date: str, end_date: str, sleep_s: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    universe_by_code = universe.set_index("ts_code", drop=False) if not universe.empty else pd.DataFrame()
    for ts_code in universe["ts_code"].astype(str):
        daily = _call_with_retries(
            pro.daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            sleep_s=sleep_s,
        )
        if daily.empty:
            continue
        adj = _call_with_retries(
            pro.adj_factor,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            sleep_s=sleep_s,
        )
        basic = _call_with_retries(
            pro.daily_basic,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=(
                "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,"
                "ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
            ),
            sleep_s=sleep_s,
        )
        merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left") if not adj.empty else daily
        if not basic.empty:
            merged = merged.merge(basic, on=["ts_code", "trade_date"], how="left")
        merged["asset_type"] = "stock"
        if not universe_by_code.empty and ts_code in universe_by_code.index:
            static = universe_by_code.loc[ts_code]
            for col in ["symbol", "name", "area", "industry", "market", "list_date"]:
                if col in static.index:
                    merged[col] = static[col]
        frames.append(merged)
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = _prefix_columns(
        raw,
        {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "pre_close": "$pre_close",
            "change": "$change",
            "pct_chg": "$pct_chg",
            "vol": "$volume",
            "amount": "$amount",
            "adj_factor": "$factor",
            "turnover_rate": "$turnover_rate",
            "turnover_rate_f": "$turnover_rate_f",
        },
    )
    return _to_datetime_index(raw, "trade_date")


def _fetch_future_daily(pro: Any, universe: pd.DataFrame, start_date: str, end_date: str, sleep_s: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ts_code in universe.get("ts_code", pd.Series(dtype=str)).astype(str):
        daily = _call_with_retries(
            pro.fut_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            sleep_s=sleep_s,
        )
        if daily.empty:
            continue
        daily["asset_type"] = "future"
        frames.append(daily)
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = _prefix_columns(
        raw,
        {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "pre_close": "$pre_close",
            "settle": "$settle",
            "pre_settle": "$pre_settle",
            "change1": "$change1",
            "change2": "$change2",
            "vol": "$volume",
            "amount": "$amount",
            "oi": "$open_interest",
            "oi_chg": "$open_interest_chg",
        },
    )
    if "$factor" not in raw.columns:
        raw["$factor"] = 1.0
    return _to_datetime_index(raw, "trade_date")


def _fetch_minute_for_codes(
    ts_module: Any,
    pro: Any,
    codes: list[tuple[str, str]],
    start_date: str,
    end_date: str,
    cfg: TushareUpdateConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_dt = pd.to_datetime(start_date).strftime("%Y-%m-%d 09:00:00")
    end_dt = pd.to_datetime(end_date).strftime("%Y-%m-%d 16:00:00")
    for ts_code, asset in codes:
        try:
            df = _call_with_retries(
                ts_module.pro_bar,
                ts_code=ts_code,
                api=pro,
                asset=asset,
                freq=cfg.minute_freq,
                start_date=start_dt,
                end_date=end_dt,
                sleep_s=cfg.request_sleep,
            )
        except Exception as exc:  # noqa: BLE001
            if "频率超限" in str(exc) or "权限" in str(exc):
                break
            continue
        if df.empty:
            continue
        df["asset_type"] = "stock" if asset == "E" else "future"
        frames.append(df)
        time.sleep(cfg.request_sleep)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    date_col = "trade_time" if "trade_time" in raw.columns else "datetime" if "datetime" in raw.columns else "trade_date"
    raw = _prefix_columns(
        raw,
        {
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "pre_close": "$pre_close",
            "change": "$change",
            "pct_chg": "$pct_chg",
            "vol": "$volume",
            "amount": "$amount",
            "oi": "$open_interest",
        },
    )
    minute = _to_datetime_index(raw, date_col)
    return minute


def _normalize_daily_columns(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    for col in ("$open", "$high", "$low", "$close", "$volume"):
        if col not in out.columns:
            out[col] = np.nan
    if "$factor" not in out.columns:
        out["$factor"] = 1.0
    if "$turnover_rate" not in out.columns:
        out["$turnover_rate"] = np.nan
    if "$turnover" not in out.columns:
        out["$turnover"] = out["$turnover_rate"]
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _normalize_minute_columns(minute: pd.DataFrame) -> pd.DataFrame:
    out = minute.copy()
    for col in ("$open", "$high", "$low", "$close", "$volume"):
        if col not in out.columns:
            out[col] = np.nan
    if "$vwap" not in out.columns and {"$amount", "$volume"}.issubset(out.columns):
        volume = pd.to_numeric(out["$volume"], errors="coerce").replace(0, np.nan)
        out["$vwap"] = pd.to_numeric(out["$amount"], errors="coerce") / volume
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _write_data_bundle(
    folder: Path,
    daily: pd.DataFrame,
    minute_pv: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    daily.to_hdf(folder / "daily_pv.h5", key="data")
    minute_pv.to_hdf(folder / "minute_pv.h5", key="data")
    legacy_quote_path = folder / "minute_quote.h5"
    if legacy_quote_path.exists():
        legacy_quote_path.unlink()
    (folder / "README.md").write_text(_readme_text(daily, minute_pv), encoding="utf-8")
    (folder / META_FILENAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / PROFILE_META_FILENAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _readme_text(daily: pd.DataFrame, minute_pv: pd.DataFrame) -> str:
    return (
        "# Factor source data\n\n"
        "Generated from Tushare and stored with key `data`.\n\n"
        f"- `daily_pv.h5`: MultiIndex ['datetime', 'instrument']; columns {list(daily.columns)}\n"
        f"- `minute_pv.h5`: Tushare minute bar data; MultiIndex ['datetime', 'instrument']; columns {list(minute_pv.columns)}\n"
    )


def _debug_subset(df: pd.DataFrame, max_days: int, max_instruments: int) -> pd.DataFrame:
    if df.empty:
        return df
    dates = df.index.get_level_values("datetime").normalize().unique().sort_values()
    instruments = list(dict.fromkeys(df.index.get_level_values("instrument")))
    chosen_dates = dates[-max_days:]
    chosen_instruments = instruments[:max_instruments]
    dt = df.index.get_level_values("datetime").normalize()
    inst = df.index.get_level_values("instrument")
    mask = dt.isin(chosen_dates) & inst.isin(chosen_instruments)
    return df.loc[mask].sort_index()


def _meta_current(path: Path) -> bool:
    meta_path = path / META_FILENAME
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return meta.get("updated_on") == pd.Timestamp.today().strftime("%Y-%m-%d")


def update_tushare_factor_data(cfg: TushareUpdateConfig) -> dict[str, Any]:
    try:
        import tushare as ts
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tushare is not installed. Install the project requirements first.") from exc

    pro = ts.pro_api(cfg.token)
    start_date, end_date = _date_bounds(cfg.years)
    stocks = _stock_universe(pro, cfg.max_stocks, cfg.request_sleep)
    futures = _future_universe(pro, cfg.max_futures, cfg.request_sleep)

    daily_frames = [
        frame
        for frame in (
            _fetch_stock_daily(pro, stocks, start_date, end_date, cfg.request_sleep),
            _fetch_future_daily(pro, futures, start_date, end_date, cfg.request_sleep),
        )
        if not frame.empty
    ]
    if not daily_frames:
        raise RuntimeError("Tushare returned no daily stock or future data.")
    daily = _normalize_daily_columns(pd.concat(daily_frames, axis=0, sort=False).sort_index())

    minute_codes = [(code, "E") for code in stocks.get("ts_code", pd.Series(dtype=str)).astype(str).tolist()]
    minute_codes.extend((code, "FT") for code in futures.get("ts_code", pd.Series(dtype=str)).astype(str).tolist())
    minute_codes = minute_codes[: cfg.max_minute_symbols]
    minute_pv = _fetch_minute_for_codes(ts, pro, minute_codes, start_date, end_date, cfg)
    if minute_pv.empty:
        raise RuntimeError("Tushare returned no minute bar data.")
    minute_pv = _normalize_minute_columns(minute_pv)

    metadata = {
        "source": "tushare",
        "updated_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "start_date": start_date,
        "end_date": end_date,
        "years": cfg.years,
        "stock_count": int(stocks.shape[0]),
        "future_count": int(futures.shape[0]),
        "daily_rows": int(daily.shape[0]),
        "minute_rows": int(minute_pv.shape[0]),
        "minute_freq": cfg.minute_freq,
    }

    _write_data_bundle(FACTOR_DATA_DIR, daily, minute_pv, metadata)
    _write_data_bundle(
        FACTOR_DATA_DEBUG_DIR,
        _debug_subset(daily, cfg.debug_days, cfg.debug_instruments),
        _debug_subset(minute_pv, cfg.debug_days, cfg.debug_instruments),
        metadata | {"debug": True},
    )
    return metadata


def auto_update_tushare_data_if_configured(force: bool = False) -> str | None:
    cfg = config_from_env()
    if cfg is None:
        return None
    required = ["daily_pv.h5", "minute_pv.h5"]
    files_ready = all((FACTOR_DATA_DIR / name).exists() and (FACTOR_DATA_DEBUG_DIR / name).exists() for name in required)
    if not force and files_ready and _meta_current(FACTOR_DATA_DIR):
        return "Using current Tushare factor data."
    metadata = update_tushare_factor_data(cfg)
    return (
        "Updated Tushare factor data "
        f"({metadata['stock_count']} stocks, {metadata['future_count']} futures, "
        f"{metadata['daily_rows']} daily rows, {metadata['minute_rows']} minute rows)."
    )
