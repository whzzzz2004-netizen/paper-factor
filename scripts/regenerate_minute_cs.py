#!/usr/bin/env python3
"""
Regenerate all minute_cs .code.py files with the new multiprocessing.Pool template.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace

BASE = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "literature_reports" / "20260726"
FULL_BASE = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "文献因子_全量" / "20260726"

# All minute_cs factors (verified by calc_factor_minute_raw grep)
MINUTE_CS_FACTORS = [
    ("高频选股因子梳理", "5MinRealizedSkewness"),
    ("高频选股因子梳理", "5MinRealizedKurtosis"),
    ("高频选股因子梳理", "CallAuctionVolumeRatio"),
    ("高频选股因子梳理", "PriceImpactProportionAnomaly"),
    ("高频选股因子梳理", "TimeProportionAnomaly"),
    ("高频选股因子梳理", "VolumeProportionAnomaly"),
    ("日内交易特征稳定性", "SDRKURT"),
    ("日内交易特征稳定性", "SDRSKEW"),
    ("日内交易特征稳定性", "SDRVOL"),
    ("日内交易特征稳定性", "SDVHHI"),
    ("日内交易特征稳定性", "SDVKURT"),
    ("日内交易特征稳定性", "SDVSKEW"),
    ("日内交易特征稳定性", "SDVVOL"),
    ("跳跃Beta与连续Beta", "ContinuousBeta"),
    ("跳跃Beta与连续Beta", "JumpBeta"),
    ("跳跃Beta与连续Beta", "continuous_beta"),
    ("跳跃Beta与连续Beta", "jump_beta"),
]


def extract_user_code(code: str) -> str:
    """Extract user code between N_WORKERS line and auto-detection section."""
    start_marker = "N_WORKERS = int(os.environ.get("
    end_marker = "# ── 自动列推断"

    start_idx = code.find(start_marker)
    if start_idx < 0:
        return None

    # Find the end of the N_WORKERS line
    end_of_start = code.find("\n", start_idx)
    start_idx = end_of_start + 1  # Start after the N_WORKERS line

    end_idx = code.find(end_marker, start_idx)
    if end_idx < 0:
        return None

    # Extract everything between, strip empty lines
    user_code = code[start_idx:end_idx].strip()
    return user_code


def extract_lookback(code: str) -> int:
    """Extract LOOKBACK_DAYS value."""
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


def regenerate_factor(code_path: Path) -> bool:
    """Regenerate a minute_cs .code.py with the new template."""
    code = code_path.read_text(encoding="utf-8")

    # Extract user code
    user_code = extract_user_code(code)
    if user_code is None:
        print(f"  ❌ Cannot extract user code from {code_path}")
        return False

    # Validate that user code has the required functions
    if "def calc_factor_minute_raw" not in user_code:
        print(f"  ❌ Missing calc_factor_minute_raw in user code")
        return False

    # Extract lookback and load_cols
    lookback = extract_lookback(code)
    load_cols = extract_load_cols(code)

    factor_name = code_path.parent.name

    # Validate syntax
    try:
        ast.parse(user_code)
    except SyntaxError as e:
        print(f"  ❌ Syntax error in user code: {e}")
        return False

    # Rebuild with new template
    wrapped = FactorFBWorkspace._build_factor_code(
        FactorFBWorkspace.MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE,
        user_code, lookback, load_cols
    )

    # Write to literature_reports
    report_name = code_path.parent.parent.name
    lit_path = BASE / report_name / factor_name / f"{factor_name}.code.py"
    lit_path.write_text(wrapped, encoding="utf-8")

    # Write to 文献因子_全量
    full_path = FULL_BASE / report_name / factor_name / f"{factor_name}.code.py"
    full_written = False
    if full_path.parent.exists():
        full_path.write_text(wrapped, encoding="utf-8")
        full_written = True

    print(f"  ✅ {report_name}/{factor_name}: lookback={lookback}, full={full_written}")
    return True


def main():
    success = fail = 0
    for report, factor in MINUTE_CS_FACTORS:
        code_path = BASE / report / factor / f"{factor}.code.py"
        if not code_path.exists():
            print(f"  ⏭️  {report}/{factor}: file not found")
            fail += 1
            continue
        print(f"\n📄 {report}/{factor}")
        if regenerate_factor(code_path):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"Done: {success} regenerated, {fail} failed")


if __name__ == "__main__":
    main()