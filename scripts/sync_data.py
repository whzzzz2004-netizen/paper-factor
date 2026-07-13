#!/usr/bin/env python3
"""
增量数据同步工具。

从远程 E 盘 (SMB) 同步增量数据，转换为挖因子所需的 per-stock/per-date parquet 格式。
自动检测新增列 → 更新 schema.json → 更新 prompt 文件。

用法:
  python3 scripts/sync_data.py            # 自动同步增量
  python3 scripts/sync_data.py --full     # 强制全量同步
  python3 scripts/sync_data.py --dry-run  # 只列变更不执行
  python3 scripts/sync_data.py --check    # 只检查远程有无新数据
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("FACTOR_DATA_DIR", str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data")))
DATA_DIR_1000 = Path(os.environ.get("FACTOR_DATA_DIR_1000", str(PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data_1000")))
SCHEMA_FILE = PROJECT_ROOT / "data" / "schema.json"
TMP_DIR = Path(tempfile.gettempdir()) / "sync_data_tmp"

# factor_field_schema.json 路径（两个数据目录都要更新）
FACTOR_FIELD_SCHEMA_PATHS = [
    DATA_DIR / "factor_field_schema.json",
    DATA_DIR_1000 / "factor_field_schema.json",
]

# ── SMB 配置 ──
SMB_HOST = "192.168.1.13"
SMB_SHARE = "E"
SMB_USER = "pc"
SMB_PASS = "123456"

REMOTE_DAILY_DIR = "market_daily_daily_new"
REMOTE_MINUTE_DIR = "market_minute_daily_new"
REMOTE_DAILY_FULL = "dailyData.parquet"
REMOTE_INDUSTRY = "jq_swIndu_comp.csv"
REMOTE_JHJJ_HSL = "jhjjHsl.csv"

# ── 需要自动更新的 prompt 文件（含模板标记位） ──
# 每个条目: (file_path, start_marker, end_marker, template_fn)
# template_fn(schema_cols) → 要插入的文本
PROMPT_TARGETS: list[tuple[str, str, str, callable]] = []

# ── 本轮检测到的新列（跨函数收集） ──
_new_cols_all: dict[str, list[str]] = {"daily": [], "minute": []}


def _load_schema() -> dict:
    return json.loads(SCHEMA_FILE.read_text())


def _save_schema(s: dict):
    SCHEMA_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")


def _load_factor_field_schema(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_factor_field_schema(path: Path, s: dict):
    path.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")


def update_factor_field_schema(new_cols: set[str], source: str):
    """将新列追加到所有数据目录下的 factor_field_schema.json"""
    for path in FACTOR_FIELD_SCHEMA_PATHS:
        schema = _load_factor_field_schema(path)
        changed = False
        for c in sorted(new_cols):
            if c not in schema:
                schema[c] = {
                    "factor_name": c,
                    "short_name": c,
                    "formula": "",
                    "source": source,
                    "note": f"来源: {source}，字段含义待补充",
                }
                changed = True
        if changed:
            _save_factor_field_schema(path, schema)
            print(f"  ✅ 已更新: {path.parent.name}/factor_field_schema.json (+{len(new_cols)} 列)", flush=True)


def _smb_cmd(cmd: str) -> subprocess.CompletedProcess:
    """执行 smbclient 命令，返回 CompletedProcess"""
    full_cmd = [
        "smbclient", f"//{SMB_HOST}/{SMB_SHARE}",
        "-U", f"{SMB_USER}%{SMB_PASS}",
        "-c", cmd,
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=120)


def _smb_list(dir_path: str) -> list[tuple[str, int]]:
    """列出远程目录中的文件 (name, size)"""
    r = _smb_cmd(f"cd {dir_path}; ls")
    files = []
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] not in (".", ".."):
            name = parts[0]
            try:
                size = int(parts[2])
                files.append((name, size))
            except ValueError:
                continue
    return files


def _smb_download(remote_path: str, local_path: Path):
    """从远程下载文件"""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    r = _smb_cmd(f"cd {Path(remote_path).parent}; get {Path(remote_path).name} {local_path}")
    if r.returncode != 0:
        raise RuntimeError(f"下载失败: {remote_path} ({r.stderr.strip()})")


def _get_local_date_set(data_subdir: str) -> set[str]:
    """获取本地已有的日期集合（统一转 YYYYMMDD）"""
    dates_file = DATA_DIR / data_subdir
    if dates_file.exists():
        raw = json.loads(dates_file.read_text())
        return {d.replace("-", "") for d in raw}  # YYYY-MM-DD → YYYYMMDD
    return set()


def _get_remote_date_set(remote_dir: str) -> set[str]:
    """获取远程按日期命名的 parquet 文件集合"""
    files = _smb_list(remote_dir)
    dates = set()
    for name, _ in files:
        if name.endswith(".parquet") and len(name) == 16:  # YYYYMMDD.parquet
            dates.add(name[:8])
    return dates


def _ensure_local_structure():
    """确保本地目录结构完整"""
    for sub in [
        "stock_data/daily", "stock_data/minute_by_date",
        "stock_data/minutes",
    ]:
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


def sync_daily_incremental(dry_run: bool = False) -> list[str]:
    """同步增量日线数据（按日期 parquet → per-stock parquet）"""
    print("\n=== 日线增量同步 ===", flush=True)

    local_dates = _get_local_date_set("stock_data/daily/trade_dates.json")
    remote_dates = _get_remote_date_set(REMOTE_DAILY_DIR)
    new_dates = sorted(remote_dates - local_dates)

    if not new_dates:
        print("  无新增日期", flush=True)
        return []

    print(f"  发现 {len(new_dates)} 个新日期: {new_dates[:5]}...", flush=True)
    if dry_run:
        return new_dates

    # 下载新文件并转换
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    new_stock_data: dict[str, pd.DataFrame] = {}
    all_stocks = set()

    for date_str in new_dates:
        remote_file = f"{REMOTE_DAILY_DIR}/{date_str}.parquet"
        local_tmp = TMP_DIR / f"{date_str}.parquet"
        try:
            _smb_download(remote_file, local_tmp)
        except RuntimeError as e:
            print(f"  ⚠️ 下载失败 {date_str}: {e}", flush=True)
            continue

        df = pd.read_parquet(local_tmp)
        for _, row in df.iterrows():
            stock = str(row["symbol"]).zfill(6)
            all_stocks.add(stock)
            if stock not in new_stock_data:
                new_stock_data[stock] = []
            new_stock_data[stock].append({
                "open": row.get("open", np.nan),
                "close": row.get("close", np.nan),
                "high": row.get("high", np.nan),
                "low": row.get("low", np.nan),
                "factor": row.get("factor", np.nan),
                "volume": row.get("volume", np.nan),
                "amount": row.get("volume", np.nan) * row.get("close", np.nan),
            })

        # 检测新列（对比 schema）
        remote_cols = set(df.columns) - {"symbol", "date"}
        schema_cols = set(_load_schema()["daily"]["columns"].keys())
        new_cols = remote_cols - schema_cols
        if new_cols:
            print(f"  检测到新列: {new_cols}", flush=True)
            s = _load_schema()
            for c in new_cols:
                s["daily"]["columns"][c] = {"description": c, "source": "remote_daily"}
            _save_schema(s)
            update_factor_field_schema(new_cols, "remote_daily")
            _new_cols_all["daily"].extend(sorted(new_cols))

        local_tmp.unlink(missing_ok=True)
    if not new_stock_data:
        print("  无有效数据", flush=True)
        return []

    # 追加到 per-stock parquet
    for stock, rows in new_stock_data.items():
        new_df = pd.DataFrame(rows)
        new_df.index = pd.to_datetime(date_str)  # placeholder, will be properly set
        stock_file = DATA_DIR / "stock_data" / "daily" / f"{stock}.parquet"
        if stock_file.exists():
            old = pd.read_parquet(stock_file)
            combined = pd.concat([old, new_df])
            combined.to_parquet(stock_file)
        else:
            new_df.to_parquet(stock_file)

    # 更新日期列表
    all_dates = sorted(local_dates | set(new_dates))
    (DATA_DIR / "stock_data/daily/trade_dates.json").write_text(json.dumps(all_dates))
    (DATA_DIR / "stock_data/daily/stock_list.json").write_text(json.dumps(sorted(all_stocks)))

    print(f"  ✅ 完成: {len(new_dates)} 天, {len(new_stock_data)} 只有增量", flush=True)
    return new_dates


def sync_daily_full(dry_run: bool = False):
    """从 dailyData.parquet 全量同步（含财务数据）"""
    print("\n=== 日线全量同步 ===", flush=True)
    if dry_run:
        print("  [dry-run] 将下载 dailyData.parquet 并转换", flush=True)
        return

    print("  下载 dailyData.parquet (376MB)...", flush=True)
    local_full = TMP_DIR / "dailyData.parquet"
    _smb_download(REMOTE_DAILY_FULL, local_full)

    print("  读取中...", flush=True)
    df = pd.read_parquet(local_full)

    print(f"  共 {len(df)} 行, 列: {list(df.columns)}", flush=True)

    # 检测新列
    schema = _load_schema()
    remote_cols = set(df.columns) - {"symbol", "date"}
    schema_cols = set(schema["daily"]["columns"].keys())
    new_cols = remote_cols - schema_cols
    if new_cols:
        print(f"  检测到新列: {new_cols}", flush=True)
        for c in new_cols:
            schema["daily"]["columns"][c] = {"description": c, "source": "dailyData.parquet"}
        _save_schema(schema)
        update_factor_field_schema(new_cols, "dailyData.parquet")
        _new_cols_all["daily"].extend(sorted(new_cols))

    print("  逐股票写入...", flush=True)
    for stock in df["symbol"].unique():
        stock_str = str(stock).zfill(6)
        sub = df[df["symbol"] == stock].copy()
        sub = sub.drop(columns=["symbol"])
        if "date" in sub.columns:
            sub = sub.set_index("date")
        sub.index = pd.to_datetime(sub.index)

        if "amount" not in sub.columns:
            sub["amount"] = sub["volume"] * sub["close"]

        stock_file = DATA_DIR / "stock_data" / "daily" / f"{stock_str}.parquet"
        if stock_file.exists():
            old = pd.read_parquet(stock_file)
            # 只追加新日期
            existing_dates = set(old.index.date)
            new_rows = sub[~sub.index.date.isin(existing_dates)]
            if not new_rows.empty:
                combined = pd.concat([old, new_rows])
                combined.to_parquet(stock_file)
        else:
            sub.to_parquet(stock_file)

    local_full.unlink(missing_ok=True)
    print("  ✅ 日线全量同步完成", flush=True)


def sync_industry(dry_run: bool = False):
    """同步行业分类数据"""
    print("\n=== 行业分类同步 ===", flush=True)
    if dry_run:
        print("  [dry-run] 将下载 jq_swIndu_comp.csv", flush=True)
        return

    local_csv = TMP_DIR / "jq_swIndu_comp.csv"
    _smb_download(REMOTE_INDUSTRY, local_csv)
    df = pd.read_csv(local_csv)

    # 格式：code, name, industry
    industry_dict = {}
    for _, row in df.iterrows():
        code = str(row.get("code", "")).zfill(6)
        indu = row.get("industry_name", row.get("industry", ""))
        if code and indu:
            industry_dict[code] = indu

    out_file = DATA_DIR / "stock_data" / "daily" / "industry.json"
    out_file.write_text(json.dumps(industry_dict, indent=2, ensure_ascii=False))
    print(f"  ✅ 行业分类: {len(industry_dict)} 只股票, {len(set(industry_dict.values()))} 个行业", flush=True)
    local_csv.unlink(missing_ok=True)


def sync_jhjj_hsl(dry_run: bool = False):
    """同步集合竞价换手率"""
    print("\n=== 集合竞价换手率同步 ===", flush=True)
    if dry_run:
        print("  [dry-run] 将下载 jhjjHsl.csv", flush=True)
        return

    local_csv = TMP_DIR / "jhjjHsl.csv"
    _smb_download(REMOTE_JHJJ_HSL, local_csv)
    df = pd.read_csv(local_csv)

    # 检测新列
    schema = _load_schema()
    remote_cols = set(df.columns) - {"symbol", "date"}
    schema_cols = set(schema["daily"]["columns"].keys())
    new_cols = remote_cols - schema_cols
    if new_cols:
        print(f"  检测到新列: {new_cols}", flush=True)
        for c in new_cols:
            schema["daily"]["columns"][c] = {"description": c, "source": "jhjjHsl.csv"}
        _save_schema(schema)
        update_factor_field_schema(new_cols, "jhjjHsl.csv")
        _new_cols_all["daily"].extend(sorted(new_cols))

    # 逐股票合并
    for stock in df["symbol"].unique():
        stock_str = str(stock).zfill(6)
        sub = df[df["symbol"] == stock].copy()
        sub = sub.drop(columns=["symbol"])
        if "date" in sub.columns:
            sub = sub.set_index("date")
        sub.index = pd.to_datetime(sub.index)

        stock_file = DATA_DIR / "stock_data" / "daily" / f"{stock_str}.parquet"
        if not stock_file.exists():
            continue
        old = pd.read_parquet(stock_file)
        for col in sub.columns:
            if col in old.columns:
                old[col] = old[col].fillna(sub[col])
            else:
                old[col] = sub[col]
        old.to_parquet(stock_file)

    print(f"  ✅ jhjj_hsl 合并完成: {len(df)} 行", flush=True)
    local_csv.unlink(missing_ok=True)


def sync_minute_incremental(dry_run: bool = False) -> list[str]:
    """同步分钟线增量（按日期 parquet → 直接存 per-date parquet）"""
    print("\n=== 分钟线增量同步 ===", flush=True)

    local_dates = _get_local_date_set("stock_data/minute_by_date/trade_dates.json")
    remote_dates = _get_remote_date_set(REMOTE_MINUTE_DIR)
    new_dates = sorted(remote_dates - local_dates)

    if not new_dates:
        print("  无新增日期", flush=True)
        return []

    print(f"  发现 {len(new_dates)} 个新日期: {new_dates[:5]}...", flush=True)
    if dry_run:
        return new_dates

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    all_stocks = set()

    for date_str in new_dates:
        remote_file = f"{REMOTE_MINUTE_DIR}/{date_str}.parquet"
        local_dst = DATA_DIR / "stock_data" / "minute_by_date" / f"{date_str}.parquet"
        try:
            _smb_download(remote_file, local_dst)
        except RuntimeError as e:
            print(f"  ⚠️ 下载失败 {date_str}: {e}", flush=True)
            continue

        # 检测新列
        df = pq.read_schema(local_dst)
        remote_cols = set(df.names) - {"symbol", "date", "datetime", "instrument"}
        schema = _load_schema()
        schema_cols = set(schema["minute"]["columns"].keys())
        new_cols = remote_cols - schema_cols
        if new_cols:
            print(f"  分钟线新列: {new_cols}", flush=True)
            for c in new_cols:
                schema["minute"]["columns"][c] = {"description": c, "source": "remote_minute"}
            _save_schema(schema)
            update_factor_field_schema(new_cols, "remote_minute")
            _new_cols_all["minute"].extend(sorted(new_cols))

        # 提取股票列表
        df = pd.read_parquet(local_dst, columns=["symbol"] if "symbol" in df.names else ["instrument"])
        for col in ["symbol", "instrument"]:
            if col in df.columns:
                all_stocks.update(str(s).zfill(6) for s in df[col].unique())
                break

    # 更新元数据
    all_dates = sorted(local_dates | set(new_dates))
    (DATA_DIR / "stock_data/minute_by_date/trade_dates.json").write_text(json.dumps(all_dates))
    (DATA_DIR / "stock_data/minute_by_date/stock_list.json").write_text(json.dumps(sorted(all_stocks)))

    print(f"  ✅ 完成: {len(new_dates)} 天, {len(all_stocks)} 只", flush=True)
    return new_dates


def update_prompt_files(schema: dict):
    """用模板标记位更新 prompt 文件"""
    daily_cols = schema["daily"]["columns"]
    minute_cols = schema["minute"]["columns"]

    # 加载 factor_field_schema 获取更丰富的字段描述
    ff_schema = _load_factor_field_schema(FACTOR_FIELD_SCHEMA_PATHS[0])

    def _fmt_line(col: str, src: dict) -> str:
        """生成单列描述行，优先使用 factor_field_schema 的丰富信息"""
        ff = ff_schema.get(col) or ff_schema.get(f"${col}")
        if ff and ff.get("short_name") and ff["short_name"] != col:
            parts = [f"  - {col}: {ff['short_name']}"]
            note = ff.get("note", "")
            if note:
                parts.append(f"（{note}）")
            return "".join(parts)
        # fallback: 用 data/schema.json 的描述
        desc = src.get("description", col)
        if desc and desc != col:
            return f"  - {col}: {desc}"
        return f"  - {col}"

    daily_lines = [_fmt_line(k, v) for k, v in sorted(daily_cols.items())]
    minute_lines = [_fmt_line(k, v) for k, v in sorted(minute_cols.items())]

    daily_text = "\n".join(daily_lines)
    minute_text = "\n".join(minute_lines)

    # 按文件名分组（仅保留实际存在的文件）
    updates = {
        "rdagent/components/coder/factor_coder/prompts.yaml": [
            ("<!-- DAILY_COLUMNS -->", "<!-- /DAILY_COLUMNS -->", daily_text),
            ("<!-- MINUTE_COLUMNS -->", "<!-- /MINUTE_COLUMNS -->", minute_text),
        ],
        # factor skill knowledge files
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
        content = full_path.read_text()
        changed = False
        for start_marker, end_marker, new_text in markers:
            pattern = re.escape(start_marker) + r".*?" + re.escape(end_marker)
            replacement = f"{start_marker}\n{new_text}\n{end_marker}"
            new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
            if count > 0:
                content = new_content
                changed = True
        if changed:
            full_path.write_text(content)
            print(f"  ✅ 已更新: {rel_path}", flush=True)


def check_remote() -> dict:
    """检查远程数据状态"""
    print("\n=== 远程数据检查 ===", flush=True)

    # 检查连通性
    try:
        files = _smb_list(".")
        print(f"  ✅ 远程连接正常 (E盘)", flush=True)
    except Exception as e:
        print(f"  ❌ 远程连接失败: {e}", flush=True)
        return {"connected": False}

    # 检查日线
    daily_files = _smb_list(REMOTE_DAILY_DIR)
    daily_count = len([f for f in daily_files if f[0].endswith(".parquet")])
    print(f"  日线: {daily_count} 个文件", flush=True)

    # 检查分钟
    minute_files = _smb_list(REMOTE_MINUTE_DIR)
    minute_count = len([f for f in minute_files if f[0].endswith(".parquet")])
    print(f"  分钟: {minute_count} 个文件", flush=True)

    # 检查 dailyData
    full_files = {f[0]: f[1] for f in _smb_list(".")}
    daily_data_size = full_files.get(REMOTE_DAILY_FULL, 0)
    print(f"  dailyData.parquet: {daily_data_size//1024//1024}MB", flush=True)

    industry_exists = full_files.get(REMOTE_INDUSTRY)
    print(f"  行业分类: {'存在' if industry_exists else '不存在'}", flush=True)

    # 对比本地
    local_daily_dates = _get_local_date_set("stock_data/daily/trade_dates.json")
    remote_daily_dates = _get_remote_date_set(REMOTE_DAILY_DIR)
    new_daily = remote_daily_dates - local_daily_dates
    if new_daily:
        print(f"  日线待同步: {len(new_daily)} 天 ({min(new_daily)} ~ {max(new_daily)})", flush=True)
    else:
        print(f"  日线已是最新 ({len(local_daily_dates)} 天)", flush=True)

    local_minute_dates = _get_local_date_set("stock_data/minute_by_date/trade_dates.json")
    remote_minute_dates = _get_remote_date_set(REMOTE_MINUTE_DIR)
    new_minute = remote_minute_dates - local_minute_dates
    if new_minute:
        print(f"  分钟待同步: {len(new_minute)} 天 ({min(new_minute)} ~ {max(new_minute)})", flush=True)
    else:
        print(f"  分钟已是最新 ({len(local_minute_dates)} 天)", flush=True)

    return {
        "connected": True,
        "daily_count": daily_count,
        "minute_count": minute_count,
        "new_daily": len(new_daily),
        "new_minute": len(new_minute),
    }


def _rebuild_limit_up():
    """从全量日线数据重新生成 limit_up_daily.parquet（涨停板剔除列表）"""
    lu_path = DATA_DIR / "limit_up_daily.parquet"
    stock_dir = DATA_DIR / "stock_data" / "daily"
    stock_list_file = stock_dir / "stock_list.json"
    if not stock_list_file.exists():
        print("  ⚠️  无 stock_list.json，跳过涨停列表生成", flush=True)
        return
    stock_list = json.loads(stock_list_file.read_text())
    print(f"  生成涨停列表 ({len(stock_list)} stocks)...", flush=True)
    rows = []
    for i, s in enumerate(stock_list):
        try:
            df = pd.read_parquet(stock_dir / f"{s}.parquet", columns=["pct_chg"])
        except Exception:
            continue
        # 涨停条件：pct_chg >= 9.5%（A股主板10%、科创/创业板20%，9.5%保底）
        mask = df["pct_chg"] >= 9.5
        if mask.any():
            for dt in df.index[mask]:
                rows.append({"datetime": dt, "instrument": s, "pct_chg": float(df.loc[dt, "pct_chg"])})
        if (i + 1) % 1000 == 0 or (i + 1) == len(stock_list):
            print(f"    扫描: {i+1}/{len(stock_list)}", flush=True)
    if not rows:
        print("  ⚠️  无涨停记录", flush=True)
        return
    result = pd.DataFrame(rows).sort_values(["datetime", "instrument"]).reset_index(drop=True)
    # 同时保存到两个数据目录（全量 + 测试300）
    result.to_parquet(lu_path)
    lu_path_1000 = DATA_DIR_1000 / "limit_up_daily.parquet"
    lu_path_1000.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(lu_path_1000)
    print(f"  ✅ 涨停列表: {len(result)} 条, 日期 {result['datetime'].min().date()} ~ {result['datetime'].max().date()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="增量数据同步工具")
    parser.add_argument("--full", action="store_true", help="强制全量同步")
    parser.add_argument("--dry-run", action="store_true", help="仅列出变更不执行")
    parser.add_argument("--check", action="store_true", help="仅检查远程状态")
    parser.add_argument("--skip-prompts", action="store_true", help="跳过 prompt 文件更新")
    parser.add_argument("--update-prompts-only", action="store_true", help="仅更新 prompt 文件（不同步数据）")
    args = parser.parse_args()

    t0 = time.time()

    if args.check:
        check_remote()
        return

    if args.update_prompts_only:
        update_prompt_files(_load_schema())
        print(f"\n✅ prompt 文件更新完成, {time.time()-t0:.0f}s", flush=True)
        return

    if args.dry_run:
        sync_daily_incremental(dry_run=True)
        sync_minute_incremental(dry_run=True)
        print(f"\n[dry-run] 完成, {time.time()-t0:.0f}s", flush=True)
        return

    _ensure_local_structure()
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 行业分类（如果本地没有）
        if not (DATA_DIR / "stock_data/daily/industry.json").exists():
            sync_industry()
        elif args.full:
            sync_industry()

        # 2. 日线增量
        sync_daily_incremental()

        # 3. 日线全量财务（如果本地缺少 dailyData 中的列）
        schema = _load_schema()
        sample_stock = list((DATA_DIR / "stock_data/daily").glob("*.parquet"))
        if sample_stock:
            sample_cols = set(pq.read_schema(sample_stock[0]).names)
            daily_data_cols = {k for k, v in schema["daily"]["columns"].items()
                              if v.get("source") == "dailyData.parquet"}
            missing_fin_cols = daily_data_cols - sample_cols
            if missing_fin_cols or args.full:
                print(f"  缺财务列: {missing_fin_cols}", flush=True)
                sync_daily_full(dry_run=args.dry_run)

        # 4. jhjj_hsl（如果本地没有该列）
        if sample_stock:
            sample_cols = set(pq.read_schema(sample_stock[0]).names)
            if "jhjj_hsl" not in sample_cols or args.full:
                sync_jhjj_hsl(dry_run=args.dry_run)

        # 5. 分钟线增量
        sync_minute_incremental()

        # 6. 如果有新列 → 输出给 agent
        all_new = sorted(set(_new_cols_all["daily"] + _new_cols_all["minute"]))
        if all_new:
            print(f"\n⚠️ NEW_COLUMNS_DETECTED: {all_new}", flush=True)
            print("Agent: 读 data/数据说明.txt 理解新列含义 → 更新 schema.json + factor_field_schema.json → 运行 --update-prompts-only", flush=True)

        # 7. 更新 prompt 文件
        if not args.skip_prompts:
            update_prompt_files(_load_schema())

        # 8. 从日线数据重新生成涨停列表 (limit_up_daily.parquet)
        _rebuild_limit_up()

        print(f"\n✅ 全部完成, {time.time()-t0:.0f}s", flush=True)

    finally:
        # 清理临时目录
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
