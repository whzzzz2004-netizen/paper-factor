import re
import shutil
import os
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.utils.env import QTDockerEnv

REAL_MINUTE_PV_FILENAME = "minute_pv.h5"
REAL_MINUTE_QUOTE_FILENAME = "minute_quote.h5"

DAILY_COLUMN_GROUPS: dict[str, dict[str, str]] = {
    "价格": {"$open": "开盘价", "$high": "最高价", "$low": "最低价", "$close": "收盘价", "$pre_close": "前收盘价"},
    "涨跌": {"$change": "涨跌额", "$pct_chg": "涨跌幅"},
    "成交量额": {"$volume": "成交量(股)", "$amount": "成交额(元)"},
    "换手": {"$turnover_rate": "换手率", "$turnover_rate_f": "换手率(自由流通股本)", "$turnover": "换手率(别名)", "$volume_ratio": "量比"},
    "估值": {"$pe": "市盈率", "$pe_ttm": "市盈率(TTM)", "$pb": "市净率", "$ps": "市销率", "$ps_ttm": "市销率(TTM)"},
    "股息": {"$dv_ratio": "股息率", "$dv_ttm": "股息率(TTM)"},
    "市值股本": {"$total_mv": "总市值", "$circ_mv": "流通市值", "$total_share": "总股本", "$float_share": "流通股本", "$free_share": "自由流通股本"},
    "复权": {"$factor": "复权因子"},
    "期货": {"$pre_settle": "前结算价", "$settle": "结算价", "$change1": "涨跌1", "$change2": "涨跌2", "$open_interest": "持仓量", "$open_interest_chg": "持仓量变化"},
    "元信息": {"$symbol": "股票代码", "$name": "股票名称", "$area": "地区", "$industry": "行业", "$market": "市场", "$list_date": "上市日期", "$asset_type": "资产类型"},
}

MINUTE_COLUMN_GROUPS: dict[str, dict[str, str]] = {
    "价格": {"$open": "分钟开盘价", "$close": "分钟收盘价", "$high": "分钟最高价", "$low": "分钟最低价"},
    "成交": {"$volume": "分钟成交量", "$vwap": "分钟均价"},
}


def _format_column_groups(groups: dict[str, dict[str, str]], available_columns: list[str]) -> str:
    """Format column groups into a compact, readable description, skipping empty groups."""
    available = set(available_columns)
    lines = []
    for category, cols in groups.items():
        present = {k: v for k, v in cols.items() if k in available}
        if not present:
            continue
        items = ", ".join(f"{k}({v})" for k, v in present.items())
        lines.append(f"  {category}: {items}")
    return "\n".join(lines)


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


def _build_intraday_profile(base_daily_df: pd.DataFrame, minutes_per_day: int = 240) -> tuple[np.ndarray, np.ndarray]:
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
    if daily_df.empty:
        raise ValueError(f"Cannot generate minute sample data from empty source: {daily_path}")

    instruments = list(dict.fromkeys(daily_df.index.get_level_values("instrument")))
    chosen_instruments = instruments[:max_instruments]
    available_dates = daily_df.index.get_level_values("datetime").unique().sort_values()
    chosen_dates = available_dates[-max_days:]
    sampled_daily = daily_df.loc[pd.IndexSlice[chosen_dates, chosen_instruments], :].copy()

    volume_curve, drift_curve = _build_intraday_profile(sampled_daily)
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
        mid = trend + oscillation
        mid = np.clip(mid, base_low, base_high)

        prev_close = np.concatenate(([base_open], mid[:-1]))
        close = mid
        high = np.maximum(prev_close, close) + 0.05 * (base_high - base_low + 1e-6)
        low = np.minimum(prev_close, close) - 0.05 * (base_high - base_low + 1e-6)
        high = np.clip(high, base_low, base_high)
        low = np.clip(low, base_low, base_high)

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

    minute_pv_df = pd.concat(minute_frames).sort_index()
    minute_pv_df.to_hdf(minute_pv_path, key="data")
    if minute_quote_path.exists():
        minute_quote_path.unlink()


def ensure_sample_minute_data_files() -> None:
    folder_specs = [
        (Path(FACTOR_COSTEER_SETTINGS.data_folder), 252, 80),
        (Path(FACTOR_COSTEER_SETTINGS.data_folder_debug), 60, 20),
    ]
    for folder, max_days, max_instruments in folder_specs:
        daily_path = folder / "daily_pv.h5"
        real_minute_pv_path = folder / REAL_MINUTE_PV_FILENAME
        real_minute_quote_path = folder / REAL_MINUTE_QUOTE_FILENAME
        if not daily_path.exists():
            continue

        real_ready = real_minute_pv_path.exists() and _minute_day_count(real_minute_pv_path) >= max_days
        if real_ready:
            continue

        _generate_minute_sample_data(
            daily_path=daily_path,
            minute_pv_path=real_minute_pv_path,
            minute_quote_path=real_minute_quote_path,
            max_days=max_days,
            max_instruments=max_instruments,
        )


def generate_data_folder_from_qlib():
    template_path = Path(__file__).parent / "factor_data_template"
    qtde = QTDockerEnv()
    qtde.prepare()

    # Run the Qlib backtest
    execute_log = qtde.check_output(
        local_path=str(template_path),
        entry=f"python generate.py",
    )

    assert (Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5").exists(), (
        "daily_pv_all.h5 is not generated. It means rdagent/scenarios/qlib/experiment/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
        + execute_log
    )
    assert (Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5").exists(), (
        "daily_pv_debug.h5 is not generated. It means rdagent/scenarios/qlib/experiment/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
        + execute_log
    )

    Path(FACTOR_COSTEER_SETTINGS.data_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "README.md",
    )

    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "README.md",
    )
    ensure_sample_minute_data_files()


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string("""
# {{file_name}}

## File Type
{{type_desc}}

## Content Overview
{{content}}
""")

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### Data Structure\n"
        df_info += (
            f"- Index: MultiIndex with levels {df.index.names}\n"
            if isinstance(df.index, pd.MultiIndex)
            else f"- Index: {df.index.name}\n"
        )

        df_info += "\n### Columns\n"
        available_columns = list(df.columns)

        column_groups = DAILY_COLUMN_GROUPS
        if "minute_pv" in p.name.lower():
            column_groups = MINUTE_COLUMN_GROUPS

        df_info += _format_column_groups(column_groups, available_columns)

        unknown = [c for c in available_columns if not any(c in g for g in column_groups.values())]
        if unknown:
            df_info += "\n  其他: " + ", ".join(unknown) + "\n"

        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += "\n### Sample Data\n"
            df_info += f"Showing data for instrument {one_instrument}:\n"
            df_info += str(df_on_one_instrument.head(5))

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )

    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown Documentation",
                content=content,
            )

    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def resolve_factor_data_mode() -> str:
    mode = os.environ.get("RDAGENT_FACTOR_DATA_MODE", "all").strip().lower()
    if mode not in {"all", "daily", "minute"}:
        return "all"
    return mode


def factor_mode_instruction(mode: str | None = None) -> str:
    selected_mode = resolve_factor_data_mode() if mode is None else mode
    data_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    has_real_minute_files = (data_folder / REAL_MINUTE_PV_FILENAME).exists()
    minute_source_sentence = (
        f"Prefer real minute-level source files such as {REAL_MINUTE_PV_FILENAME}."
        if has_real_minute_files
        else f"Use minute-level source files named {REAL_MINUTE_PV_FILENAME}."
    )
    instructions = {
        "daily": (
            "Current factor mining mode: daily_factor.\n"
            "Prefer daily source files such as daily_pv.h5.\n"
            "Produce one daily factor value per trading day and instrument."
        ),
        "minute": (
            "Current factor mining mode: minute_factor.\n"
            + minute_source_sentence
            + "\n"
            "You may aggregate intraday minute bar information into one daily factor value per trading day and instrument.\n"
            "Do not output minute-level factor values."
        ),
        "all": (
            "Current factor mining mode: generic.\n"
            "You may choose from the available source files, but the final result must be one daily factor value per trading day and instrument."
        ),
    }
    return instructions[selected_mode]


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    if (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).exists()
    ):
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib()
    ensure_sample_minute_data_files()
    mode = resolve_factor_data_mode()
    pattern_by_mode = {
        "all": r"^(.+\.h5|README\.md)$" if fname_reg == ".*" else fname_reg,
        "daily": r"^(daily_.*\.h5|README\.md)$",
        "minute": r"^(minute_pv\.h5|README\.md)$",
    }
    content_l = []
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(pattern_by_mode[mode], p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return factor_mode_instruction(mode) + "\n\n" + "\n----------------- file splitter -------------\n".join(content_l)


def get_compact_data_folder_intro() -> str:
    mode = resolve_factor_data_mode()
    data_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)

    def _h5_columns(file_name: str) -> list[str]:
        path = data_folder / file_name
        if path.exists():
            try:
                return list(pd.read_hdf(path, key="data").columns)
            except Exception:  # noqa: BLE001
                pass
        return []

    def _daily_intro() -> str:
        cols = _h5_columns("daily_pv.h5")
        if not cols:
            return "- daily_pv.h5: 日线数据，字段不可用"
        return "- daily_pv.h5: 日线数据，MultiIndex ['datetime', 'instrument']\n" + _format_column_groups(
            DAILY_COLUMN_GROUPS, cols
        )

    def _minute_intro() -> str:
        cols = _h5_columns("minute_pv.h5")
        if not cols:
            return "- minute_pv.h5: 分钟数据，字段不可用"
        return "- minute_pv.h5: 分钟数据，MultiIndex ['datetime', 'instrument']\n" + _format_column_groups(
            MINUTE_COLUMN_GROUPS, cols
        )

    file_intros = {"daily": _daily_intro, "minute": _minute_intro, "all": lambda: f"{_daily_intro()}\n{_minute_intro()}"}

    return factor_mode_instruction(mode) + "\n\nAvailable source files:\n" + file_intros[mode]()
