#!/usr/bin/env python3
"""import_new_data.py 的最小回归测试。

每个用例在临时目录里搭一个小数据仓库，不碰真实数据。
跑法：python3 scripts/test_import_new_data.py
"""

import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "drive_files" / "scripts" / "import_new_data.py"


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


def _daily(values: dict, dates: list) -> pd.DataFrame:
    df = pd.DataFrame(values, index=pd.to_datetime(dates))
    df.index.name = "datetime"
    return df


def _minute(values: dict, stamps: list, stock: str = "000001") -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([[stock], pd.to_datetime(stamps)],
                                     names=["instrument", "datetime"])
    return pd.DataFrame(values, index=idx)


class Sandbox:
    """一个临时 data root，目录结构与真实仓库一致。"""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pf_import_test_"))
        (self.root / "scripts").mkdir()
        shutil.copy(SOURCE, self.root / "scripts" / "import_new_data.py")

        self.market_full = self.root / "数据仓库/行情数据/日线/全量/stock_data/daily"
        self.market_test = self.root / "数据仓库/行情数据/日线/测试/stock_data/daily"
        self.fund_full = self.root / "数据仓库/非行情数据/全量/stock_data/daily"
        self.fund_test = self.root / "数据仓库/非行情数据/测试/stock_data/daily"
        self.minute_full = self.root / "数据仓库/行情数据/分钟线/全量/stock_data"
        self.minute_test = self.root / "数据仓库/行情数据/分钟线/测试/stock_data"
        self.raw = self.root / "原始数据"
        for p in (
            self.market_full, self.market_test, self.fund_full, self.fund_test,
            self.minute_full / "minute_by_date", self.minute_test / "minute_by_date",
            self.raw / "日线", self.raw / "分钟线",
            self.raw / "非行情/日线", self.raw / "非行情/分钟线",
        ):
            p.mkdir(parents=True, exist_ok=True)

        _write_json(self.root / "schema.json", {"daily": {"columns": {}},
                                                "minute": {"columns": {}}})
        for p in (
            self.root / "数据仓库/行情数据/日线/全量/factor_field_schema.json",
            self.root / "数据仓库/行情数据/日线/测试/factor_field_schema.json",
        ):
            _write_json(p, {})
        self.state = {"daily_market": None, "daily_factors": {}, "minute_factors": {}}
        self.save_state()
        self.set_lists(self.market_full, [], [])
        self.set_lists(self.market_test, [], [])
        self.set_lists(self.fund_full, [], [])
        self.set_lists(self.fund_test, [], [])
        self.set_minute_lists([], [])

        os.environ["PAPER_FACTOR_DATA_ROOT"] = str(self.root)
        spec = importlib.util.spec_from_file_location(
            f"import_new_data_{id(self)}", self.root / "scripts/import_new_data.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    # ── 写数据 ────────────────────────────────────────────────
    def write_market(self, stock: str, df: pd.DataFrame, test: bool = True):
        _write_parquet(df, self.market_full / f"{stock}.parquet")
        if test:
            _write_parquet(df, self.market_test / f"{stock}.parquet")

    def write_fund(self, stock: str, df: pd.DataFrame, test: bool = True):
        _write_parquet(df, self.fund_full / f"{stock}.parquet")
        if test:
            _write_parquet(df, self.fund_test / f"{stock}.parquet")

    def write_minute(self, date: str, df: pd.DataFrame, test: bool = True):
        _write_parquet(df, self.minute_full / "minute_by_date" / f"{date}.parquet")
        if test:
            _write_parquet(df, self.minute_test / "minute_by_date" / f"{date}.parquet")

    def set_lists(self, daily_dir: Path, stocks: list, dates: list):
        _write_json(daily_dir / "stock_list.json", stocks)
        _write_json(daily_dir / "trade_dates.json", dates)

    def set_minute_lists(self, dates: list, stocks: list):
        for p in (self.minute_full, self.minute_full / "minute_by_date",
                  self.minute_test, self.minute_test / "minute_by_date"):
            _write_json(p / "trade_dates.json", dates)
            _write_json(p / "stock_list.json", stocks)

    def save_state(self):
        _write_json(self.root / "数据仓库/import_state.json", self.state)

    def run(self):
        rc = self.mod.do_import()
        # do_import 会从磁盘重新加载并写回 state；测试里同步一份最新值。
        self.state = json.loads((self.root / "数据仓库/import_state.json").read_text())
        return rc

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# 用例
# ═══════════════════════════════════════════════════════════════════
def test_daily_new_stock_and_column():
    sb = Sandbox()
    try:
        sb.state["daily_market"] = "2026-01-02"
        sb.save_state()
        base = _daily({"open": [1.0, 1.1], "close": [1.0, 1.1]},
                      ["2026-01-01", "2026-01-02"])
        sb.write_market("000001", base)
        sb.set_lists(sb.market_full, ["000001"], ["20260101", "20260102"])
        sb.set_lists(sb.market_test, ["000001"], ["20260101", "20260102"])

        raw = pd.DataFrame({
            "symbol": ["000001", "000003", "000003"],
            "trade_date": ["20260103", "20260101", "20260102"],
            "open": [2.0, 3.0, 3.1],
            "close": [2.0, 3.0, 3.1],
            "new_col": [9.0, 8.0, 8.1],
        })
        _write_parquet(raw, sb.raw / "日线/dailyData.parquet")

        sb.run()
        new_stock = pd.read_parquet(sb.market_full / "000003.parquet")
        assert new_stock.index.tolist() == [pd.Timestamp("2026-01-01"),
                                            pd.Timestamp("2026-01-02")], new_stock.index
        assert "000003" in json.loads((sb.market_full / "stock_list.json").read_text())
        test_cols = pd.read_parquet(sb.market_test / "000001.parquet").columns
        assert "new_col" in test_cols, test_cols
        schema = json.loads((sb.root / "schema.json").read_text())
        assert "new_col" in schema["daily"]["columns"]
        print("✅ 日线：新股票历史/stock_list/新列/测试补列")
    finally:
        sb.close()


def test_daily_new_column_backfills_history():
    sb = Sandbox()
    try:
        sb.state["daily_market"] = "2026-01-02"
        sb.save_state()
        base = _daily({"open": [1.0, 1.1], "close": [1.0, 1.1]},
                      ["2026-01-01", "2026-01-02"])
        sb.write_market("000001", base)
        sb.set_lists(sb.market_full, ["000001"], ["20260101", "20260102"])
        sb.set_lists(sb.market_test, ["000001"], ["20260101", "20260102"])

        raw = pd.DataFrame({
            "symbol": ["000001"] * 3,
            "trade_date": ["20260101", "20260102", "20260103"],
            "open": [9.0, 9.1, 2.0],
            "close": [9.0, 9.1, 2.0],
            "new_col": [5.0, 6.0, 7.0],
        })
        _write_parquet(raw, sb.raw / "日线/dailyData.parquet")

        sb.run()
        full = pd.read_parquet(sb.market_full / "000001.parquet")
        assert full.loc["2026-01-01", "new_col"] == 5.0   # 新列回填历史
        assert full.loc["2026-01-01", "open"] == 1.0     # 旧列不被旧日期覆盖
        assert full.loc["2026-01-03", "open"] == 2.0     # 旧列更新新日期

        test = pd.read_parquet(sb.market_test / "000001.parquet")
        assert test["new_col"].tolist() == [5.0, 6.0]
        print("✅ 日线：新列回填全部历史，已有列只更新新日期")
    finally:
        sb.close()


def test_minute_new_column_existing_date():
    sb = Sandbox()
    try:
        stamps = ["2026-01-01 09:31", "2026-01-01 09:32"]
        base = _minute({"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1],
                        "volume": [10, 10]}, stamps)
        sb.write_minute("20260101", base)
        sb.set_minute_lists(["20260101"], ["000001"])
        raw = base.copy()
        raw["my_factor"] = [5.0, 6.0]
        _write_parquet(raw, sb.raw / "分钟线/20260101.parquet")

        sb.run()
        for path in (sb.minute_full / "minute_by_date/20260101.parquet",
                     sb.minute_test / "minute_by_date/20260101.parquet"):
            df = pd.read_parquet(path)
            assert {"open", "close", "volume", "my_factor"} <= set(df.columns), df.columns
            assert df["my_factor"].tolist() == [5.0, 6.0]
        schema = json.loads((sb.root / "schema.json").read_text())
        assert "my_factor" in schema["minute"]["columns"]
        assert "my_factor" not in schema["daily"]["columns"]
        print("✅ 分钟：已有日期合并新列（测试+全量），schema 进 minute")
    finally:
        sb.close()


def test_minute_flat_format_drops_date_columns():
    sb = Sandbox()
    try:
        stamps = ["2026-01-01 09:31", "2026-01-01 09:32"]
        base = _minute({"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1],
                        "volume": [10, 10]}, stamps)
        sb.write_minute("20260101", base)
        sb.set_minute_lists(["20260101"], ["000001"])
        flat = pd.DataFrame({
            "symbol": ["000001", "000001"],
            "trade_date": stamps,
            "date": ["2026-01-01", "2026-01-01"],
            "open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1],
            "volume": [10, 10], "my_factor": [5.0, 6.0],
        })
        _write_parquet(flat, sb.raw / "分钟线/20260101.parquet")

        sb.run()
        df = pd.read_parquet(sb.minute_full / "minute_by_date/20260101.parquet")
        assert "my_factor" in df.columns
        assert "date" not in df.columns
        schema = json.loads((sb.root / "schema.json").read_text())
        assert "my_factor" in schema["minute"]["columns"]
        assert "date" not in schema["minute"]["columns"]
        print("✅ 分钟：扁平格式的 trade_date/date 被丢弃，不误判为新列")
    finally:
        sb.close()


def test_minute_new_date_keeps_base_columns():
    sb = Sandbox()
    try:
        base = _minute({"open": [1], "high": [1], "low": [1], "close": [1], "volume": [10]},
                       ["2026-01-01 09:31"])
        sb.write_minute("20260101", base)
        sb.set_minute_lists(["20260101"], ["000001"])
        raw = _minute({"open": [2], "high": [2], "low": [2], "close": [2], "volume": [20],
                       "my_factor": [7.0]}, ["2026-01-02 09:31"])
        _write_parquet(raw, sb.raw / "分钟线/20260102.parquet")

        sb.run()
        df = pd.read_parquet(sb.minute_full / "minute_by_date/20260102.parquet")
        assert {"open", "high", "low", "close", "volume", "my_factor"} <= set(df.columns), df.columns
        print("✅ 分钟：新日期+新列写完整 DataFrame（不丢 OHLCV）")
    finally:
        sb.close()


def test_minute_partial_new_date_rejected():
    sb = Sandbox()
    try:
        sb.set_minute_lists([], [])
        _write_parquet(_minute({"my_factor": [9.0]}, ["2026-01-02 09:31"]),
                       sb.raw / "分钟线/20260102.parquet")
        sb.run()
        assert not (sb.minute_full / "minute_by_date/20260102.parquet").exists()
        print("✅ 分钟：只有新列、没有基础 OHLCV 的新日期被拒绝，不生成半成品")
    finally:
        sb.close()


def test_cross_sectional_duplicate_dates():
    sb = Sandbox()
    try:
        base = _daily({"open": [1.0, 1.1], "close": [1.0, 1.1]},
                      ["2026-01-01", "2026-01-02"])
        sb.write_market("000001", base)
        sb.write_fund("000001", _daily({"analyst_coverage": [1.0, 2.0]},
                                       ["2026-01-01", "2026-01-02"]))
        sb.set_lists(sb.market_full, ["000001"], ["20260101", "20260102"])
        sb.set_lists(sb.market_test, ["000001"], ["20260101", "20260102"])
        sb.set_lists(sb.fund_full, ["000001"], ["20260101", "20260102"])
        sb.set_lists(sb.fund_test, ["000001"], ["20260101", "20260102"])
        sb.state["daily_market"] = "2026-01-02"
        sb.save_state()

        src = pd.DataFrame({"000001": [1.0, 2.0, 3.0]},
                           index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"]))
        _write_parquet(src, sb.raw / "非行情/日线/f_new.parquet")

        sb.run()
        df = pd.read_parquet(sb.fund_full / "000001.parquet")
        assert df.loc["2026-01-02", "f_new"] == 3.0
        assert "f_new" in pd.read_parquet(sb.fund_test / "000001.parquet").columns
        schema = json.loads((sb.root / "schema.json").read_text())
        assert "f_new" in schema["daily"]["columns"]
        print("✅ 截面因子：重复日期去重、注册 schema、补测试")
    finally:
        sb.close()


def test_calendar_is_not_merged():
    sb = Sandbox()
    try:
        sb.write_market("000001", _daily({"close": [1, 1]},
                                         ["2026-01-01", "2026-01-02"]), test=False)
        sb.set_lists(sb.market_full, ["000001"], ["20260101", "20260102"])
        for d in ("20260102", "20260103"):
            sb.write_minute(d, _minute({"close": [1.0]}, [f"{d[:4]}-{d[4:6]}-{d[6:]} 09:31"]),
                            test=False)
        sb.mod.rebuild_authoritative_trade_dates()
        day = json.loads((sb.market_full / "trade_dates.json").read_text())
        minute = json.loads((sb.minute_full / "trade_dates.json").read_text())
        assert day == ["20260101", "20260102"], day
        assert minute == ["20260102", "20260103"], minute
        print("✅ 日历：日线/分钟分开，分钟日期不污染日线")
    finally:
        sb.close()


def test_minute_factor_stops_at_missing_date():
    sb = Sandbox()
    try:
        for d in ("20260101", "20260103"):
            sb.write_minute(d, _minute({"close": [1.0]}, [f"{d[:4]}-{d[4:6]}-{d[6:]} 09:31"]))
        sb.set_minute_lists(["20260101", "20260103"], ["000001"])

        src = pd.DataFrame({"000001": [1.0, 2.0, 3.0]},
                           index=pd.to_datetime(["2026-01-01 09:31", "2026-01-02 09:31",
                                                 "2026-01-03 09:31"]))
        _write_parquet(src, sb.raw / "非行情/分钟线/mf.parquet")

        sb.run()
        assert sb.state["minute_factors"]["mf"] == "2026-01-01 09:31:00", sb.state
        assert "mf" in pd.read_parquet(sb.minute_full / "minute_by_date/20260101.parquet").columns
        assert "mf" not in pd.read_parquet(sb.minute_full / "minute_by_date/20260103.parquet").columns

        # 补上缺失的基础日期后应继续处理 01-02、01-03
        sb.write_minute("20260102",
                        _minute({"close": [1.0]}, ["2026-01-02 09:31"]))
        sb.run()
        assert sb.state["minute_factors"]["mf"] == "2026-01-03 09:31:00", sb.state
        for d in ("20260102", "20260103"):
            assert "mf" in pd.read_parquet(
                sb.minute_full / f"minute_by_date/{d}.parquet").columns
        schema = json.loads((sb.root / "schema.json").read_text())
        assert "mf" in schema["minute"]["columns"]
        assert "mf" not in schema["daily"]["columns"]
        print("✅ 分钟因子：缺基础日期时暂停，补齐后继续；schema 进 minute")
    finally:
        sb.close()


def main():
    tests = [
        test_daily_new_stock_and_column,
        test_daily_new_column_backfills_history,
        test_minute_new_column_existing_date,
        test_minute_flat_format_drops_date_columns,
        test_minute_new_date_keeps_base_columns,
        test_minute_partial_new_date_rejected,
        test_cross_sectional_duplicate_dates,
        test_calendar_is_not_merged,
        test_minute_factor_stops_at_missing_date,
    ]
    for fn in tests:
        fn()
    print(f"\n全部通过：{len(tests)} 个用例")


if __name__ == "__main__":
    main()
