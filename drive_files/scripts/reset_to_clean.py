#!/usr/bin/env python3
"""
reset_to_clean.py — 把旧版数据仓库清理到"干净起点"（配合新 getdata 流程使用）。

背景：getdata 机制已更新为"域自治、各自增量、无 ffill"。
若目标机（老板电脑）的 数据仓库/ 还是旧版（含大量空基本面列、价值因子1/2、旧 schema 注册、
旧 import_state），请先在目标机运行本脚本做一次"清场"，再跑 `python getdata.py` 从头重建。

本脚本做四件事（全部可安全重跑）：
  1. 删除 原始数据/非行情/日线/ 下不再需要的旧因子源文件（价值因子1/2 及其它非在册因子）
  2. 清空 数据仓库/非行情数据/{全量,测试}/stock_data/daily/*.parquet 的列（保留日期索引）
  3. 重置 schema.json / factor_field_schema.json：仅保留 新因子描述.csv 里描述的因子
  4. 重置 import_state.json：清空 daily_factors（让 getdata 全量重导入）

用法:
  python reset_to_clean.py [--dry-run]

注意：此脚本不删除行情/分钟数据；只处理"非行情截面因子"区域。
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_ROOT = Path(r"D:\paper-factor-data")
WH = DATA_ROOT / "数据仓库"
RAW = DATA_ROOT / "原始数据"
RAW_FUND_DAILY = RAW / "非行情" / "日线"
FUND_FULL = WH / "非行情数据" / "全量" / "stock_data" / "daily"
FUND_TEST = WH / "非行情数据" / "测试" / "stock_data" / "daily"
SCHEMA = DATA_ROOT / "schema.json"
FF_PATHS = [
    WH / "行情数据" / "日线" / "全量" / "factor_field_schema.json",
    WH / "行情数据" / "日线" / "测试" / "factor_field_schema.json",
]
STATE = WH / "import_state.json"
DESC_FILE = RAW / "新因子描述.csv"

ALL_STANDARD_FACTOR_NAMES = None  # 由 update_from_csv() 填充（标准因子清单）


def log(msg):
    print(msg, flush=True)


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def current_source_factors() -> list:
    """返回 原始数据/非行情/日线/ 下当前有源文件的因子名列表。"""
    if not RAW_FUND_DAILY.exists():
        return []
    return [p.stem for p in RAW_FUND_DAILY.glob("*.parquet")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印将做什么，不实际执行")
    args = ap.parse_args()
    dry = args.dry_run

    # ── 1. 删除旧因子源文件（价值因子1/2）──
    stale = ["价值因子1", "价值因子2", "test_", "AB"]
    removed_src = []
    for p in RAW_FUND_DAILY.glob("*.parquet"):
        if any(p.stem.startswith(s) for s in stale):
            removed_src.append(p.name)
    if removed_src:
        log(f"1) [删除旧因子源文件] {len(removed_src)} 个: {removed_src}")
        if not dry:
            for name in removed_src:
                (RAW_FUND_DAILY / name).unlink(missing_ok=True)
    else:
        log("1) 原始数据/非行情/日线/ 没有需要删除的旧源文件")

    # ── 2. 清空非行情 per-stock 列 ──
    for base, tag in [(FUND_FULL, "全量"), (FUND_TEST, "测试")]:
        files = glob.glob(str(base / "*.parquet"))
        cnt = 0
        for f in files:
            try:
                df = pd.read_parquet(f)
                if len(df) and len(df.columns):
                    if not dry:
                        df[[]].to_parquet(f)
                    cnt += 1
            except Exception:
                continue
        log(f"【2) 清空{tag} per-stock】 {cnt}/{len(files)} 文件列已清空（保留日期索引）")

    # ── 3. 重置 schema / factor_field_schema ──
    #     只保留：新因子描述.csv 里描述的所有因子
    descs = {}
    if DESC_FILE.exists():
        raw = DESC_FILE.read_bytes()
        text = next((raw.decode(e) for e in ("utf-8-sig", "utf-8", "gb18030", "gbk")
                     if _try_decode(raw, e)), None) or ""
        import csv, io
        for row in csv.reader(io.StringIO(text)):
            if row and row[0].strip():
                descs[row[0].strip()] = row[1].strip() if len(row) > 1 else ""

    keep = set(descs.keys())
    log(f"【3】按新因子描述.csv 保留因子: {sorted(keep) or '（无）'}")

    schema = load_json(SCHEMA) or {}
    daily_cfg = schema.setdefault("daily", {}).setdefault("columns", {})
    # 仅保留 keep 里的列（若列不在 keep 中但描述里有也保留）
    for c in list(daily_cfg.keys()):
        if c not in keep:
            del daily_cfg[c]
    # 也把 keep 里还没有的列补上基本注册
    for c in keep:
        if c not in daily_cfg:
            daily_cfg[c] = {"description": descs.get(c, c), "source": "原始数据导入"}
    if not dry:
        save_json(SCHEMA, schema)
    log(f"      schema.json daily.columns → {sorted(daily_cfg)}")

    for ffp in FACTOR_FIELD_SCHEMA_PATHS:
        ff = load_json(ffp) or {}
        for c in list(ff.keys()):
            if c not in keep:
                del ff[c]
        for c in keep:
            if c not in ff:
                ff[c] = {"factor_name": c, "short_name": descs.get(c, c), "formula": "",
                         "source": "原始数据导入", "note": descs.get(c, "（见 新因子描述.csv）")}
        if not dry:
            save_json(ffp, ff)
        log(f"      factor_field_schema.json（{ffp.parent.name}）→ {sorted(ff)}")

    # ── 4. 重置 import_state —— 清空 daily_factors ──
    st = load_json(STATE) or {}
    st["daily_factors"] = {}
    if not dry:
        save_json(STATE, st)
    log(f"   import_state.json daily_factors → {{}}")

    log("")
    if dry:
        log("（dry-run 完成，未实际改动。）")
    else:
        log("✅ reset 完成。下一步：清空后运行 `python getdata.py --check` 或直接 `python getdata.py` 从头重建。")
    return 0


def _try_decode(raw, enc):
    try:
        raw.decode(enc)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())