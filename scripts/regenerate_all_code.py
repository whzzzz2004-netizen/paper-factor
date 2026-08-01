#!/usr/bin/env python3
"""
一键遍历全部因子，用当前模板重新生成 .code.py + 修复 meta.json。

用法:
  python scripts/regenerate_all_code.py                          # 重新生成所有因子
  python scripts/regenerate_all_code.py --report 研报名          # 只处理指定研报
  python scripts/regenerate_all_code.py --factor 因子名          # 只处理指定因子
  python scripts/regenerate_all_code.py --type minute_cs         # 只处理指定类型
  python scripts/regenerate_all_code.py --dry-run                # 只列出，不执行
  python scripts/regenerate_all_code.py --fix-meta-only          # 只修复 meta.json，不重新生成代码

支持的类型: daily, minute, cross_section, minute_cs, deep_learning
"""
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace

BASE = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "literature_reports" / "20260727"
FULL_BASE = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "文献因子_全量" / "20260727"

# ── 检测因子类型 ──

def detect_factor_type(code_text: str) -> str:
    """从 .code.py 代码文本检测因子类型"""
    if "calc_factor_minute_raw" in code_text and "cross_section_transform" in code_text:
        return "minute_cs"
    if "calc_factors_one_day" in code_text:
        return "minute"  # 必须在 calc_factor_series 之前：分钟模板可能含有该字符串
    if "calc_factor_single_stock" in code_text:
        return "daily"
    if "calc_factor_series" in code_text:
        return "daily"  # vectorized daily
    if "calc_factor_cross_section" in code_text:
        return "cross_section"
    if "train_model" in code_text and "predict" in code_text:
        return "deep_learning"
    return "unknown"


# ── 提取用户代码 ──

def extract_user_code_minute_cs(code: str) -> str | None:
    """Extract user code between N_WORKERS line and auto-detection section."""
    start_marker = "N_WORKERS = int(os.environ.get("
    end_marker = "# ── 自动列推断"
    start_idx = code.find(start_marker)
    if start_idx < 0:
        return None
    end_of_start = code.find("\n", start_idx)
    start_idx = end_of_start + 1
    end_idx = code.find(end_marker, start_idx)
    if end_idx < 0:
        return None
    return code[start_idx:end_idx].strip()


def extract_user_code_daily(code: str) -> str | None:
    """Extract calc_factor_single_stock or calc_factor_series function."""
    for func_name in ["calc_factor_single_stock", "calc_factor_series"]:
        idx = code.find(f"def {func_name}(")
        if idx >= 0:
            # Find where this function ends (next top-level def)
            rest = code[idx:]
            lines = rest.split("\n")
            func_lines = []
            func_indent = None
            for i, line in enumerate(lines):
                if i == 0:
                    func_lines.append(line)
                    func_indent = len(line) - len(line.lstrip())
                    continue
                if not line.strip():
                    func_lines.append(line)
                    continue
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                if (stripped.startswith("def ") or stripped.startswith("@")) and current_indent <= func_indent:
                    break
                if current_indent <= func_indent and not stripped.startswith("#"):
                    break
                func_lines.append(line)
            return "\n".join(func_lines)
    return None


def extract_user_code_cross_section(code: str) -> str | None:
    """Extract calc_factor_cross_section function."""
    idx = code.find("def calc_factor_cross_section(")
    if idx < 0:
        return None
    rest = code[idx:]
    lines = rest.split("\n")
    func_lines = []
    func_indent = None
    for i, line in enumerate(lines):
        if i == 0:
            func_lines.append(line)
            func_indent = len(line) - len(line.lstrip())
            continue
        if not line.strip():
            func_lines.append(line)
            continue
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if (stripped.startswith("def ") or stripped.startswith("@")) and current_indent <= func_indent:
            break
        if current_indent <= func_indent and not stripped.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def extract_user_code_minute(code: str) -> str | None:
    """Extract calc_factors_one_day function."""
    idx = code.find("def calc_factors_one_day(")
    if idx < 0:
        return None
    rest = code[idx:]
    lines = rest.split("\n")
    func_lines = []
    func_indent = None
    for i, line in enumerate(lines):
        if i == 0:
            func_lines.append(line)
            func_indent = len(line) - len(line.lstrip())
            continue
        if not line.strip():
            func_lines.append(line)
            continue
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if (stripped.startswith("def ") or stripped.startswith("@")) and current_indent <= func_indent:
            break
        if current_indent <= func_indent and not stripped.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def extract_user_code_dl(code: str) -> str | None:
    """Extract train_model and predict functions."""
    parts = []
    for func_name in ["train_model", "predict", "predict_batch"]:
        idx = code.find(f"def {func_name}(")
        if idx >= 0:
            rest = code[idx:]
            lines = rest.split("\n")
            func_lines = []
            func_indent = None
            for i, line in enumerate(lines):
                if i == 0:
                    func_lines.append(line)
                    func_indent = len(line) - len(line.lstrip())
                    continue
                if not line.strip():
                    func_lines.append(line)
                    continue
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                if (stripped.startswith("def ") or stripped.startswith("@")) and current_indent <= func_indent:
                    break
                if current_indent <= func_indent and not stripped.startswith("#"):
                    break
                func_lines.append(line)
            parts.append("\n".join(func_lines))
    return "\n\n".join(parts) if parts else None


def extract_lookback(code: str) -> int:
    """Extract LOOKBACK_DAYS value."""
    m = re.search(r'LOOKBACK_DAYS\s*=\s*min\(max\(1,\s*(\d+)\),\s*\d+\)', code)
    if m:
        return int(m.group(1))
    m = re.search(r'LOOKBACK_DAYS\s*=\s*max\(1,\s*(\d+)\)', code)
    if m:
        return int(m.group(1))
    m = re.search(r'LOOKBACK_DAYS\s*=\s*(\d+)', code)
    if m:
        return int(m.group(1))
    return 20


def extract_load_cols(code: str) -> list | None:
    """Extract _LOAD_COLS definition."""
    m = re.search(r'_LOAD_COLS\s*=\s*(\[[^\]]*\])', code)
    if m:
        try:
            cols = ast.literal_eval(m.group(1))
            return cols if cols else None
        except:
            pass
    return None


# ── 模板映射 ──

TEMPLATE_MAP = {
    "minute_cs": FactorFBWorkspace.MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE,
    "daily": FactorFBWorkspace.DAILY_FRAMEWORK_TEMPLATE,
    "cross_section": FactorFBWorkspace.CROSS_SECTION_FRAMEWORK_TEMPLATE,
    "minute": FactorFBWorkspace.MINUTE_FRAMEWORK_TEMPLATE,
    "deep_learning": FactorFBWorkspace.DEEP_LEARNING_FRAMEWORK_TEMPLATE,
}

EXTRACTOR_MAP = {
    "minute_cs": extract_user_code_minute_cs,
    "daily": extract_user_code_daily,
    "cross_section": extract_user_code_cross_section,
    "minute": extract_user_code_minute,
    "deep_learning": extract_user_code_dl,
}


# ── 修复 meta.json ──

def fix_meta_json(meta_path: Path) -> bool:
    """修复 meta.json 中缺失的字段。从 literature_reports 对应因子复制。"""
    if not meta_path.exists():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    # 检查哪些字段缺失
    required = ["factor_description", "factor_formulation", "source_excerpt", "source_report_title", "source_report_path"]
    missing = [k for k in required if not meta.get(k)]
    if not missing:
        return False  # 无需修复

    # 尝试从 literature_reports 对应因子复制
    factor_name = meta.get("factor_name", meta_path.stem.replace(".meta", ""))
    report_name = meta.get("report_name") or meta_path.parent.parent.name

    src_path = BASE / report_name / factor_name / f"{factor_name}.meta.json"
    if not src_path.exists():
        # 尝试其他命名风格
        for variant in [factor_name.lower(), factor_name.upper(), factor_name.capitalize()]:
            for p in (BASE / report_name).rglob(f"{variant}.meta.json"):
                src_path = p
                break
            if src_path.exists():
                break

    if src_path.exists():
        try:
            src_meta = json.loads(src_path.read_text(encoding="utf-8"))
            for k in missing:
                if src_meta.get(k):
                    meta[k] = src_meta[k]
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception:
            pass

    return False


# ── 主逻辑 ──

def regenerate_factor(code_path: Path, dry_run: bool = False) -> dict:
    """重新生成单个因子的 .code.py"""
    factor_name = code_path.parent.name
    report_name = code_path.parent.parent.name
    result = {"report": report_name, "factor": factor_name, "status": None, "error": None}

    code = code_path.read_text(encoding="utf-8")
    ftype = detect_factor_type(code)

    if ftype == "unknown":
        result["status"] = "skipped"
        result["error"] = "unknown type"
        return result

    template = TEMPLATE_MAP.get(ftype)
    extractor = EXTRACTOR_MAP.get(ftype)
    if not template or not extractor:
        result["status"] = "skipped"
        result["error"] = "no template/extractor"
        return result

    user_code = extractor(code)
    if not user_code:
        result["status"] = "failed"
        result["error"] = "cannot extract user code"
        return result

    # Validate syntax
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        result["status"] = "failed"
        result["error"] = f"syntax error: {e}"
        return result

    lookback = extract_lookback(code)
    load_cols = extract_load_cols(code)

    if dry_run:
        result["status"] = "dry_run"
        result["type"] = ftype
        result["lookback"] = lookback
        return result

    # Rebuild
    try:
        wrapped = FactorFBWorkspace._build_factor_code(template, user_code, lookback, load_cols)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"build failed: {e}"
        return result

    # Write to literature_reports
    lit_path = BASE / report_name / factor_name / f"{factor_name}.code.py"
    lit_path.write_text(wrapped, encoding="utf-8")

    # Write to 文献因子_全量
    full_path = FULL_BASE / report_name / factor_name / f"{factor_name}.code.py"
    full_written = False
    if full_path.parent.exists():
        full_path.write_text(wrapped, encoding="utf-8")
        full_written = True

    result["status"] = "success"
    result["type"] = ftype
    result["lookback"] = lookback
    result["full_written"] = full_written
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="一键重新生成所有因子 .code.py")
    parser.add_argument("--report", help="只处理指定研报（模糊匹配）")
    parser.add_argument("--factor", help="只处理指定因子（模糊匹配）")
    parser.add_argument("--type", help="只处理指定类型: daily/minute/cross_section/minute_cs/deep_learning")
    parser.add_argument("--dry-run", action="store_true", help="只列出，不执行")
    parser.add_argument("--fix-meta-only", action="store_true", help="只修复 meta.json，不重新生成代码")
    args = parser.parse_args()

    if args.fix_meta_only:
        # 只修复 meta.json
        meta_fixed = 0
        for base in [BASE, FULL_BASE]:
            if not base.exists():
                continue
            for m in sorted(base.rglob("*.meta.json")):
                if fix_meta_json(m):
                    print(f"  ✅ 修复 meta: {m.relative_to(base.parent.parent)}")
                    meta_fixed += 1
        print(f"\n修复了 {meta_fixed} 个 meta.json")
        return

    # 收集所有 .code.py
    all_files = []
    for cp in sorted(BASE.rglob("*.code.py")):
        if ".code.code.py" in cp.name:
            continue
        report_name = cp.parent.parent.name
        factor_name = cp.parent.name

        if args.report and args.report not in report_name:
            continue
        if args.factor and args.factor not in factor_name:
            continue

        code = cp.read_text(encoding="utf-8")
        ftype = detect_factor_type(code)
        if args.type and ftype != args.type:
            continue
        if ftype == "unknown":
            continue

        all_files.append((cp, ftype))

    if not all_files:
        print("无待处理因子")
        return

    print(f"找到 {len(all_files)} 个因子:")
    for cp, ftype in all_files:
        print(f"  [{ftype:>12}] {cp.parent.parent.name}/{cp.parent.name}")

    if args.dry_run:
        return

    # 执行
    success = fail = skipped = 0
    for cp, ftype in all_files:
        print(f"\n📄 [{ftype}] {cp.parent.parent.name}/{cp.parent.name}")
        result = regenerate_factor(cp)
        if result["status"] == "success":
            print(f"  ✅ {result['factor']}: lookback={result['lookback']}, full={result.get('full_written')}")
            success += 1
        elif result["status"] == "skipped":
            print(f"  ⏭️  {result['factor']}: {result.get('error', '')}")
            skipped += 1
        else:
            print(f"  ❌ {result['factor']}: {result.get('error', '')}")
            fail += 1

    # 修复 meta.json
    meta_fixed = 0
    for base in [BASE, FULL_BASE]:
        if not base.exists():
            continue
        for m in sorted(base.rglob("*.meta.json")):
            if fix_meta_json(m):
                meta_fixed += 1

    print(f"\n{'='*50}")
    print(f"生成: {success} 成功, {fail} 失败, {skipped} 跳过")
    if meta_fixed:
        print(f"修复: {meta_fixed} 个 meta.json")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()