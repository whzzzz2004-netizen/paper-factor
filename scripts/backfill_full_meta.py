#!/usr/bin/env python3
"""一次性脚本：为已生成的全量因子 meta.json 回填缺失的描述字段。

从测试输出目录（literature_reports）复制 factor_description / factor_formulation /
source_report_title / source_report_path / source_excerpt，并从 extracted_reports
提取 JSON 的 cols 补全 variables。

用法:
  python scripts/backfill_full_meta.py              # 扫描所有日期
  python scripts/backfill_full_meta.py --date 20260801  # 只处理指定日期
  python scripts/backfill_full_meta.py --dry-run    # 只报告，不写回
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FULL_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "文献因子_全量"
LIT_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports"
EXTRACT_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "extracted_reports"

DESC_KEYS = (
    "factor_description", "factor_formulation", "variables",
    "source_report_title", "source_report_path", "source_excerpt",
)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def needs_backfill(meta: dict) -> bool:
    for k in DESC_KEYS:
        v = meta.get(k)
        if not v:
            return True
    return False


def find_test_meta(date_str: str, report: str, factor: str) -> dict:
    candidates = [
        LIT_BASE / date_str / report / factor / f"{factor}.meta.json",
        LIT_BASE / report / factor / f"{factor}.meta.json",
    ]
    for c in candidates:
        m = load_json(c)
        if m:
            return m
    return {}


def find_extracted_factor(date_str: str, report: str, factor: str) -> dict:
    candidates = []
    if date_str:
        candidates.append(EXTRACT_BASE / date_str / f"{report}.extracted.json")
    candidates.append(EXTRACT_BASE / f"{report}.extracted.json")
    for c in candidates:
        data = load_json(c)
        if isinstance(data, list):
            factors = data
        elif isinstance(data, dict):
            factors = data.get("factors", [])
        else:
            continue
        if isinstance(factors, list):
            for f in factors:
                if f.get("name") == factor:
                    return f
    return {}


def backfill_meta(meta: dict, date_str: str, report: str, factor: str) -> dict:
    src = find_test_meta(date_str, report, factor)
    # 测试 meta 缺的字段（含 variables）从 extracted_reports 补全
    ex = find_extracted_factor(date_str, report, factor)
    for k, src_key in (("factor_description", "description"),
                       ("factor_formulation", "formulation"),
                       ("source_excerpt", "source_excerpt")):
        if not src.get(k) and ex.get(src_key):
            src[k] = ex[src_key]
    if not src.get("variables") and ex.get("cols"):
        src["variables"] = {c: c for c in ex["cols"]}
    if not src.get("source_report_title"):
        src["source_report_title"] = report

    merged = dict(meta)
    for k in DESC_KEYS:
        if not merged.get(k) and src.get(k):
            merged[k] = src[k]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="只处理指定日期 (YYYYMMDD)")
    parser.add_argument("--dry-run", action="store_true", help="只报告不写回")
    args = parser.parse_args()

    # 收集 (date_str, report, factor, meta_path) 列表
    targets = []
    if args.date:
        base = FULL_BASE / args.date
        date_str = args.date
        scan_root = base
    else:
        base = FULL_BASE
        date_str = ""
        scan_root = base

    # 兼容两种结构: 全量/<report>/<factor> 和 全量/<date>/<report>/<factor>
    def _scan(report_dir: Path, cur_date: str):
        for factor_dir in sorted(report_dir.iterdir()):
            if not factor_dir.is_dir():
                continue
            meta_path = factor_dir / f"{factor_dir.name}.meta.json"
            if meta_path.exists():
                targets.append((cur_date, report_dir.name, factor_dir.name, meta_path))

    for report_dir in sorted(scan_root.iterdir()):
        if not report_dir.is_dir():
            continue
        if report_dir.name.isdigit() and len(report_dir.name) == 8:
            # 日期子目录结构
            for sub in sorted(report_dir.iterdir()):
                if sub.is_dir():
                    _scan(sub, report_dir.name)
        else:
            _scan(report_dir, date_str)

    fixed = 0
    already_ok = 0
    failed = 0
    for date_str, report, factor, meta_path in targets:
        meta = load_json(meta_path)
        if meta is None:
            failed += 1
            continue
        if not needs_backfill(meta):
            already_ok += 1
            continue
        merged = backfill_meta(meta, date_str, report, factor)
        if merged == meta:
            continue
        fixed += 1
        if not args.dry_run:
            meta_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        missing = [k for k in DESC_KEYS if not merged.get(k)]
        print(f"{'[DRY] ' if args.dry_run else ''}补全 {date_str}/{report}/{factor}"
              f"{'  仍缺: ' + ','.join(missing) if missing else ''}")

    print(f"\n完成: 共 {len(targets)} 个因子，补全 {fixed}，已完整 {already_ok}，读取失败 {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
