from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rdagent.log import rdagent_logger as logger


ROOT = Path.cwd()
GIT_IGNORE_DIR = ROOT / "git_ignore_folder"
PAPERS_DIR = ROOT / "papers"
TEMPLATE_DIR = ROOT / "rdagent" / "scenarios" / "qlib" / "experiment" / "factor_data_template"
FACTOR_DATA_DIR = ROOT / "git_ignore_folder" / "factor_implementation_source_data"
FACTOR_DATA_DEBUG_DIR = ROOT / "git_ignore_folder" / "factor_implementation_source_data_debug"
JQ_META_FILENAME = "jq_data_meta.json"
FULL_SAMPLE_DAYS = 252
DEBUG_SAMPLE_DAYS = 60
FULL_SAMPLE_INSTRUMENTS = 80
DEBUG_SAMPLE_INSTRUMENTS = 20


def _copy_if_missing(src: Path, dst: Path, *, force: bool = False) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not force:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return True


def _is_valid_daily_factor_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_hdf(path, key="data")
    except Exception:
        return False
    required_columns = {"$open", "$close", "$high", "$low", "$volume", "$factor"}
    return (
        df is not None
        and not df.empty
        and isinstance(df.index, pd.MultiIndex)
        and set(df.index.names) == {"datetime", "instrument"}
        and required_columns.issubset(df.columns)
    )


def _build_compact_daily_dataset(
    source_path: Path,
    target_path: Path,
    *,
    max_days: int,
    max_instruments: int,
) -> bool:
    if not _is_valid_daily_factor_file(source_path):
        return False
    df = pd.read_hdf(source_path, key="data").sort_index()
    if df.empty:
        return False

    dates = df.index.get_level_values("datetime").unique().sort_values()
    instruments = list(dict.fromkeys(df.index.get_level_values("instrument")))
    chosen_dates = dates[-max_days:]
    chosen_instruments = instruments[:max_instruments]
    compact_df = df.loc[pd.IndexSlice[chosen_dates, chosen_instruments], :].sort_index()
    if compact_df.empty:
        return False
    compact_df = _ensure_daily_turnover_columns(compact_df)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    compact_df.to_hdf(target_path, key="data")
    return True


def _ensure_daily_turnover_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "$turnover_rate" not in result.columns:
        volume = pd.to_numeric(result.get("$volume"), errors="coerce").fillna(0.0)
        scale = volume.groupby(level="datetime").transform("median").replace(0, np.nan)
        result["$turnover_rate"] = (volume / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    if "$turnover" not in result.columns:
        result["$turnover"] = result["$turnover_rate"]
    return result


def _jq_credentials() -> tuple[str, str]:
    username = os.environ.get("JQDATA_USERNAME", "").strip()
    password = os.environ.get("JQDATA_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError(
            "JQDATA_USERNAME or JQDATA_PASSWORD is not set. "
            "Put them in .env before running init_workspace so real JQData can be downloaded."
        )
    return username, password


def _meta_current(path: Path) -> bool:
    meta_path = path / JQ_META_FILENAME
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return meta.get("updated_on") == pd.Timestamp.today().strftime("%Y-%m-%d")


def _normalize_instrument(code: str) -> str:
    code = str(code)
    return code


def _to_multiindex_dataframe(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    out = df.copy()
    out = out.reset_index()
    datetime_col = next((c for c in out.columns if "date" in str(c).lower() or "time" in str(c).lower()), out.columns[0])
    out["datetime"] = pd.to_datetime(out[datetime_col])
    out["instrument"] = _normalize_instrument(instrument)
    return out.set_index(["datetime", "instrument"]).sort_index()


def _fetch_jq_turnover_rate(jq_module, stock: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        valuation = jq_module.get_valuation(
            stock,
            start_date=start_date.date(),
            end_date=end_date.date(),
            fields=["day", "turnover_ratio"],
        )
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if valuation is None or valuation.empty:
        return pd.DataFrame()
    valuation = valuation.rename(columns={"day": "datetime", "turnover_ratio": "$turnover_rate"})
    valuation["datetime"] = pd.to_datetime(valuation["datetime"])
    valuation["instrument"] = _normalize_instrument(stock)
    valuation["$turnover"] = valuation["$turnover_rate"]
    return valuation.set_index(["datetime", "instrument"]).sort_index()[["$turnover_rate", "$turnover"]]


def _fetch_jq_daily_data(jq_module, stock: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    fields = ["open", "high", "low", "close", "volume", "money", "factor", "avg", "paused"]
    daily = jq_module.get_price(
        stock,
        start_date=start_date.date(),
        end_date=end_date.date(),
        fq="post",
        frequency="daily",
        fields=fields,
    )
    if daily is None or daily.empty:
        return pd.DataFrame()
    out = _to_multiindex_dataframe(daily, stock)
    out = out.rename(
        columns={
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "money": "$amount",
            "factor": "$factor",
            "avg": "$avg",
            "paused": "$paused",
        }
    )
    turnover = _fetch_jq_turnover_rate(jq_module, stock, start_date, end_date)
    if not turnover.empty:
        out = out.join(turnover, how="left")
    return _ensure_daily_turnover_columns(out.replace([np.inf, -np.inf], np.nan))


def _fetch_jq_minute_data(jq_module, stock: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    fields = ["open", "high", "low", "close", "volume", "money", "factor", "avg", "paused"]
    minute = jq_module.get_price(
        stock,
        start_date=start_date.strftime("%Y-%m-%d 09:30:00"),
        end_date=end_date.strftime("%Y-%m-%d 15:00:00"),
        fq="post",
        frequency="minute",
        fields=fields,
    )
    if minute is None or minute.empty:
        return pd.DataFrame()
    out = _to_multiindex_dataframe(minute, stock)
    out = out.rename(
        columns={
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "money": "$amount",
            "factor": "$factor",
            "avg": "$vwap",
            "paused": "$paused",
        }
    )
    if "$vwap" not in out.columns and {"$amount", "$volume"}.issubset(out.columns):
        volume = pd.to_numeric(out["$volume"], errors="coerce").replace(0, np.nan)
        out["$vwap"] = pd.to_numeric(out["$amount"], errors="coerce") / volume
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _write_jq_bundle(folder: Path, daily: pd.DataFrame, minute_pv: pd.DataFrame, metadata: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    daily.to_hdf(folder / "daily_pv.h5", key="data")
    minute_pv.to_hdf(folder / "minute_pv.h5", key="data")
    legacy_quote_path = folder / "minute_quote.h5"
    if legacy_quote_path.exists():
        legacy_quote_path.unlink()
    (folder / "README.md").write_text(
        (
            "# Factor source data\n\n"
            "Generated from JQData and stored with key `data`.\n\n"
            f"- `daily_pv.h5`: MultiIndex ['datetime', 'instrument']; columns {list(daily.columns)}\n"
            f"- `minute_pv.h5`: MultiIndex ['datetime', 'instrument']; columns {list(minute_pv.columns)}\n"
        ),
        encoding="utf-8",
    )
    (folder / JQ_META_FILENAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _debug_subset(df: pd.DataFrame, max_days: int, max_instruments: int) -> pd.DataFrame:
    if df.empty:
        return df
    instruments = list(dict.fromkeys(df.index.get_level_values("instrument")))
    chosen_instruments = instruments[:max_instruments]
    inst = df.index.get_level_values("instrument")
    # Keep the full history in debug mode and only shrink the instrument universe.
    # Sequence models such as GRU/TCN need a continuous time span to form train/test windows.
    mask = inst.isin(chosen_instruments)
    return df.loc[mask].sort_index()


def _download_real_jq_factor_data(force: bool = False) -> list[str]:
    if (
        not force
        and _is_valid_daily_factor_file(FACTOR_DATA_DIR / "daily_pv.h5")
        and (FACTOR_DATA_DIR / "minute_pv.h5").exists()
        and _meta_current(FACTOR_DATA_DIR)
    ):
        return ["Using current JQData factor data."]

    try:
        import jqdatasdk as jq_module
    except ModuleNotFoundError as exc:
        raise RuntimeError("jqdatasdk is not installed. Install project requirements first.") from exc

    username, password = _jq_credentials()
    jq_module.auth(username, password)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=1)
    securities = jq_module.get_all_securities(types=["stock"], date=end_date.date())
    if securities is None or securities.empty:
        raise RuntimeError("JQData returned no stock universe.")
    securities = securities[securities["end_date"] >= start_date]
    securities = securities.sort_index(kind="stable").head(FULL_SAMPLE_INSTRUMENTS)
    stocks = list(securities.index.astype(str))
    if not stocks:
        raise RuntimeError("No eligible stocks were returned from JQData.")

    daily_frames: list[pd.DataFrame] = []
    minute_frames: list[pd.DataFrame] = []
    for stock in stocks:
        stock_start = max(pd.Timestamp(securities.loc[stock, "start_date"]), start_date)
        stock_end = min(pd.Timestamp(securities.loc[stock, "end_date"]), end_date)
        daily_df = _fetch_jq_daily_data(jq_module, stock, stock_start, stock_end)
        minute_df = _fetch_jq_minute_data(jq_module, stock, stock_start, stock_end)
        if not daily_df.empty:
            daily_frames.append(daily_df)
        if not minute_df.empty:
            minute_frames.append(minute_df)

    if not daily_frames:
        raise RuntimeError("JQData returned no daily data.")
    if not minute_frames:
        raise RuntimeError("JQData returned no minute data.")

    daily = pd.concat(daily_frames).sort_index()
    minute_pv = pd.concat(minute_frames).sort_index()
    metadata = {
        "source": "jqdatasdk",
        "updated_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "years": 1,
        "stock_count": len(stocks),
        "daily_rows": int(daily.shape[0]),
        "minute_rows": int(minute_pv.shape[0]),
    }
    _write_jq_bundle(FACTOR_DATA_DIR, daily, minute_pv, metadata)
    _write_jq_bundle(
        FACTOR_DATA_DEBUG_DIR,
        _debug_subset(daily, DEBUG_SAMPLE_DAYS, DEBUG_SAMPLE_INSTRUMENTS),
        _debug_subset(minute_pv, DEBUG_SAMPLE_DAYS, DEBUG_SAMPLE_INSTRUMENTS),
        metadata | {"debug": True},
    )
    return [
        f"Prepared real JQData daily factor data: {FACTOR_DATA_DIR / 'daily_pv.h5'}",
        f"Prepared real JQData minute factor data: {FACTOR_DATA_DIR / 'minute_pv.h5'}",
        f"Prepared real JQData debug daily factor data: {FACTOR_DATA_DEBUG_DIR / 'daily_pv.h5'}",
        f"Prepared real JQData debug minute factor data: {FACTOR_DATA_DEBUG_DIR / 'minute_pv.h5'}",
    ]


def _generate_synthetic_daily_factor_file(
    target_path: Path,
    *,
    n_days: int,
    n_instruments: int,
) -> None:
    rng = np.random.default_rng(42)
    end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=n_days)
    instruments = [f"SH{600000 + i:06d}" for i in range(n_instruments)]
    rows: list[pd.DataFrame] = []

    for idx, instrument in enumerate(instruments):
        base_price = 8.0 + idx * 0.3
        daily_returns = rng.normal(0.0004, 0.018, size=n_days)
        close = base_price * np.cumprod(1 + daily_returns)
        open_ = np.concatenate(([close[0] * (1 - rng.normal(0, 0.004))], close[:-1])) * (
            1 + rng.normal(0, 0.003, size=n_days)
        )
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.02, size=n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.02, size=n_days))
        volume = rng.lognormal(mean=14.0, sigma=0.5, size=n_days) * (1 + 0.15 * np.sin(np.linspace(0, 6, n_days)))
        factor = np.clip(1 + np.cumsum(rng.normal(0.0, 0.0008, size=n_days)), 0.7, 1.3)
        instrument_index = pd.MultiIndex.from_product([dates, [instrument]], names=["datetime", "instrument"])
        rows.append(
            pd.DataFrame(
                {
                    "$open": open_.astype(np.float32),
                    "$close": close.astype(np.float32),
                    "$high": high.astype(np.float32),
                    "$low": low.astype(np.float32),
                    "$volume": volume.astype(np.float32),
                    "$factor": factor.astype(np.float32),
                },
                index=instrument_index,
            )
        )

    synthetic_df = pd.concat(rows).sort_index()
    synthetic_df = _ensure_daily_turnover_columns(synthetic_df)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_hdf(target_path, key="data")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _minute_day_count(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_hdf(path, key="data")
    if df.empty or "datetime" not in df.index.names:
        return 0
    dt = pd.to_datetime(df.index.get_level_values("datetime"))
    return int(dt.normalize().nunique())


def _minute_timestamps_for_day(day: pd.Timestamp) -> pd.DatetimeIndex:
    morning = pd.date_range(day.normalize() + pd.Timedelta(hours=9, minutes=30), periods=120, freq="min")
    afternoon = pd.date_range(day.normalize() + pd.Timedelta(hours=13), periods=120, freq="min")
    return morning.append(afternoon)


def _build_intraday_profile(minutes_per_day: int = 240) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, minutes_per_day)
    volume_curve = 0.55 + 0.45 * np.abs(x)
    volume_curve = volume_curve / volume_curve.sum()
    drift_curve = np.linspace(0.0, 1.0, minutes_per_day)
    return volume_curve, drift_curve


def _generate_minute_sample_data(
    daily_path: Path,
    minute_pv_path: Path,
    minute_quote_path: Path,
    max_days: int,
    max_instruments: int,
) -> None:
    daily_df = pd.read_hdf(daily_path, key="data").sort_index()
    instruments = list(dict.fromkeys(daily_df.index.get_level_values("instrument")))
    chosen_instruments = instruments[:max_instruments]
    available_dates = daily_df.index.get_level_values("datetime").unique().sort_values()
    chosen_dates = available_dates[-max_days:]
    sampled_daily = daily_df.loc[pd.IndexSlice[chosen_dates, chosen_instruments], :].copy()

    volume_curve, drift_curve = _build_intraday_profile()
    minute_frames: list[pd.DataFrame] = []

    for (dt, instrument), row in sampled_daily.iterrows():
        base_open = float(row["$open"])
        base_close = float(row["$close"])
        base_high = float(row["$high"])
        base_low = float(row["$low"])
        base_volume = max(float(row["$volume"]), 0.0)
        minute_index = pd.MultiIndex.from_product(
            [_minute_timestamps_for_day(pd.Timestamp(dt)), [instrument]],
            names=["datetime", "instrument"],
        )

        trend = base_open + (base_close - base_open) * drift_curve
        oscillation = 0.12 * (base_high - base_low + 1e-6) * np.sin(np.linspace(0.0, 4.0 * np.pi, len(drift_curve)))
        mid = np.clip(trend + oscillation, base_low, base_high)

        prev_close = np.concatenate(([base_open], mid[:-1]))
        close = mid
        high = np.clip(np.maximum(prev_close, close) + 0.05 * (base_high - base_low + 1e-6), base_low, base_high)
        low = np.clip(np.minimum(prev_close, close) - 0.05 * (base_high - base_low + 1e-6), base_low, base_high)

        minute_volume = np.maximum(np.round(base_volume * volume_curve), 0.0)
        if minute_volume.sum() > 0:
            minute_volume[-1] += base_volume - minute_volume.sum()
        vwap = (prev_close + high + low + close) / 4.0

        minute_frames.append(
            pd.DataFrame(
                {
                    "$open": prev_close,
                    "$close": close,
                    "$high": high,
                    "$low": low,
                    "$volume": minute_volume,
                    "$vwap": vwap,
                },
                index=minute_index,
            )
        )

    pd.concat(minute_frames).sort_index().to_hdf(minute_pv_path, key="data")
    if minute_quote_path.exists():
        minute_quote_path.unlink()


def _ensure_minute_data_files() -> None:
    for folder, max_days, max_instruments in [
        (FACTOR_DATA_DIR, 252, 80),
        (FACTOR_DATA_DEBUG_DIR, 60, 20),
    ]:
        daily_path = folder / "daily_pv.h5"
        minute_pv_path = folder / "minute_pv.h5"
        minute_quote_path = folder / "minute_quote.h5"
        if not daily_path.exists():
            continue
        if minute_pv_path.exists() and _minute_day_count(minute_pv_path) >= max_days:
            continue
        _generate_minute_sample_data(daily_path, minute_pv_path, minute_quote_path, max_days, max_instruments)


def _ensure_workspace_dirs() -> list[Path]:
    dirs = [
        GIT_IGNORE_DIR,
        GIT_IGNORE_DIR / "factor_outputs",
        GIT_IGNORE_DIR / "research_store" / "knowledge",
        GIT_IGNORE_DIR / "research_store" / "knowledge_v2" / "paper_improvement",
        GIT_IGNORE_DIR / "research_store" / "knowledge_v2" / "error_cases",
        GIT_IGNORE_DIR / "factor_implementation_source_data",
        GIT_IGNORE_DIR / "factor_implementation_source_data_debug",
        GIT_IGNORE_DIR / "RD-Agent_workspace",
        GIT_IGNORE_DIR / "traces",
        GIT_IGNORE_DIR / "static",
        PAPERS_DIR,
        PAPERS_DIR / "inbox",
        PAPERS_DIR / "factor_improvement",
    ]
    for path in dirs:
        _ensure_dir(path)
    return dirs


def _ensure_env_file(force: bool = False) -> str:
    env_path = ROOT / ".env"
    env_example_path = ROOT / ".env.example"
    if env_path.exists() and not force:
        return f"Kept existing env file: {env_path}"
    if env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        return f"Created env file from template: {env_path}"
    return "Skipped env file creation because .env.example is missing."


def _ensure_factor_data(force: bool = False) -> list[str]:
    _ensure_dir(FACTOR_DATA_DIR)
    _ensure_dir(FACTOR_DATA_DEBUG_DIR)
    from rdagent.app.utils.tushare_data import auto_update_tushare_data_if_configured

    tushare_message = auto_update_tushare_data_if_configured(force=force)
    if tushare_message is not None:
        return [tushare_message]

    if os.environ.get("PAPER_FACTOR_USE_JQDATA", "").strip().lower() in {"1", "true", "yes", "on"}:
        return _download_real_jq_factor_data(force=force)

    if (
        not force
        and _is_valid_daily_factor_file(FACTOR_DATA_DIR / "daily_pv.h5")
        and (FACTOR_DATA_DIR / "minute_pv.h5").exists()
        and _is_valid_daily_factor_file(FACTOR_DATA_DEBUG_DIR / "daily_pv.h5")
        and (FACTOR_DATA_DEBUG_DIR / "minute_pv.h5").exists()
    ):
        return ["Using existing bundled/local factor source data."]

    template_debug = TEMPLATE_DIR / "daily_pv_debug.h5"
    template_full = TEMPLATE_DIR / "daily_pv_all.h5"
    source_template = template_debug if template_debug.exists() else template_full
    if not source_template.exists():
        raise RuntimeError(
            "Bundled factor data template is missing. "
            "Set PAPER_FACTOR_USE_JQDATA=1 with JQDATA_USERNAME/JQDATA_PASSWORD to download real data."
        )

    shutil.copy(source_template, FACTOR_DATA_DIR / "daily_pv.h5")
    shutil.copy(source_template, FACTOR_DATA_DEBUG_DIR / "daily_pv.h5")
    for folder in (FACTOR_DATA_DIR, FACTOR_DATA_DEBUG_DIR):
        _generate_minute_sample_data(
            folder / "daily_pv.h5",
            folder / "minute_pv.h5",
            folder / "minute_quote.h5",
            max_days=60,
            max_instruments=20,
        )

    metadata = {
        "source": "bundled_compact_sample",
        "updated_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "start_date": "sample",
        "end_date": "sample",
        "years": "sample",
        "stock_count": "sample",
    }
    for folder in (FACTOR_DATA_DIR, FACTOR_DATA_DEBUG_DIR):
        (folder / JQ_META_FILENAME).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        readme = folder / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Factor source data\n\n"
                "Generated from the bundled compact sample. "
                "Set `PAPER_FACTOR_USE_JQDATA=1` during `paper-factor init --force` to download real JQData.\n",
                encoding="utf-8",
            )

    return [
        f"Prepared bundled daily factor data: {FACTOR_DATA_DIR / 'daily_pv.h5'}",
        f"Prepared bundled minute factor data: {FACTOR_DATA_DIR / 'minute_pv.h5'}",
        f"Prepared bundled debug daily factor data: {FACTOR_DATA_DEBUG_DIR / 'daily_pv.h5'}",
        f"Prepared bundled debug minute factor data: {FACTOR_DATA_DEBUG_DIR / 'minute_pv.h5'}",
    ]


def _ingest_factor_improvement_knowledge() -> list[str]:
    from rdagent.app.qlib_rd_loop.paper_improvement import ingest_factor_improvement_papers

    report_folder = PAPERS_DIR / "factor_improvement"
    pdf_count = len(list(report_folder.rglob("*.pdf")))
    if pdf_count == 0:
        return [f"No factor-improvement papers found under {report_folder}."]

    updated_count = ingest_factor_improvement_papers(report_folder=str(report_folder))
    return [
        f"Ingested {updated_count} factor-improvement paper(s) into the paper-improvement knowledge base."
    ]


def init_workspace(
    force: bool = False,
    *,
    ingest_factor_improvement: bool = False,
) -> dict[str, list[str] | str]:
    created_dirs = [str(path) for path in _ensure_workspace_dirs()]
    env_message = _ensure_env_file(force=force)
    data_actions = _ensure_factor_data(force=force)
    if ingest_factor_improvement:
        data_actions.extend(_ingest_factor_improvement_knowledge())

    summary = {
        "created_dirs": created_dirs,
        "env": env_message,
        "data": data_actions,
        "next_steps": [
            "Review .env and fill in your API keys if needed.",
            "Run `rdagent health_check --no-check-docker` to verify API and port configuration.",
            "Run `rdagent daily_factor` or `rdagent minute_factor` to start mining factors.",
        ],
    }
    logger.info("Workspace initialization finished.")
    return summary


def validate_workspace_ready(*, require_factor_data: bool = True) -> dict[str, list[str] | str]:
    created_dirs = [str(path) for path in _ensure_workspace_dirs()]
    env_message = _ensure_env_file(force=False)

    data_messages: list[str] = []
    if require_factor_data:
        checks = [
            (FACTOR_DATA_DIR / "daily_pv.h5", _is_valid_daily_factor_file),
            (FACTOR_DATA_DIR / "minute_pv.h5", lambda path: path.exists()),
            (FACTOR_DATA_DEBUG_DIR / "daily_pv.h5", _is_valid_daily_factor_file),
            (FACTOR_DATA_DEBUG_DIR / "minute_pv.h5", lambda path: path.exists()),
        ]
        missing = [str(path) for path, checker in checks if not checker(path)]
        if missing:
            raise RuntimeError(
                "Factor source data is not ready. Missing or invalid files:\n"
                + "\n".join(f"- {path}" for path in missing)
                + "\nRun `paper-factor init --force` first."
            )
        data_messages.append("Using existing local factor source data.")

    return {
        "created_dirs": created_dirs,
        "env": env_message,
        "data": data_messages,
    }
