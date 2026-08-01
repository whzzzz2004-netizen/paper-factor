#!/usr/bin/env python3
"""
Convert daily factors from calc_factor_single_stock (per-date) to calc_factor_series (vectorized).
Also regenerates the .code.py with the new template.

Usage:
  python scripts/convert_daily_to_vectorized.py [--factor FACTOR_NAME] [--force]
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace

BASE = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "literature_reports" / "20260726"
FULL_BASE = Path(__file__).parent.parent / "git_ignore_folder" / "factor_outputs" / "文献因子_全量" / "20260726"


def extract_user_func(code: str, func_name: str) -> str:
    """Extract a function definition by name from code."""
    idx = code.find(f"def {func_name}(")
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
        # A new top-level def/dectorator at same or less indent = end of function
        if (stripped.startswith("def ") or stripped.startswith("@")) and current_indent <= func_indent:
            break
        # A non-empty, non-comment line at the base indent level ends the function
        if current_indent <= func_indent and not stripped.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def is_buggy(func_code: str) -> bool:
    """Check if a calc_factor_series function has unreachable code (buggy conversion)."""
    if "s = np.nan" not in func_code:
        return False
    lines = func_code.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "s = np.nan" and i + 2 < len(lines):
            if "s.name =" in lines[i+1] and lines[i+2].strip() == "return s":
                return True
    return False


def fix_buggy_series(func_code: str) -> str:
    """
    Fix a buggy calc_factor_series that has early s=np.nan + return s + unreachable code.
    Strategy: find the last real `s = <expr>` block, keep only that + the function signature.
    """
    lines = func_code.split("\n")
    func_indent = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
    indent_str = " " * func_indent

    # Find the last real computation: `s = <not np.nan>` at func body indent
    last_real_block_start = None
    for i in range(len(lines) - 2):
        stripped = lines[i].strip()
        curr_indent = len(lines[i]) - len(stripped)
        if stripped.startswith("s = ") and stripped != "s = np.nan" and curr_indent > func_indent:
            if i + 2 < len(lines) and "s.name =" in lines[i+1] and lines[i+2].strip() == "return s":
                last_real_block_start = i

    if last_real_block_start is None:
        # No real computation found - return a NaN Series
        name_match = re.search(r"""s\.name\s*=\s*['"](\w+)['"]""", func_code)
        name = name_match.group(1) if name_match else "factor"
        return f"def calc_factor_series(df, stock):\n{indent_str}return pd.Series(np.nan, index=df.index, name='{name}')"

    # Keep only: function signature + last real block
    kept = [lines[0]]  # function signature
    kept.extend(lines[last_real_block_start:])
    # Strip trailing empty lines
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def convert_single_stock_to_series(func_code: str) -> str:
    """Convert a calc_factor_single_stock function to calc_factor_series."""
    new_code = func_code.replace(
        "def calc_factor_single_stock(df, trade_date, stock):",
        "def calc_factor_series(df, stock):"
    )
    # Remove docstring
    new_code = re.sub(r'""".*?"""', '', new_code, count=1, flags=re.DOTALL)
    new_code = re.sub(r"'''.*?'''", '', new_code, count=1, flags=re.DOTALL)

    # Find all return {"name": value} statements
    pattern = r'return\s*\{\s*["\'](\w+)["\']\s*:\s*([^}]+?)\s*\}'
    matches = list(re.finditer(pattern, new_code))
    if not matches:
        return new_code

    for m in reversed(matches):
        is_last = (m.start() == matches[-1].start())
        name = m.group(1)
        expr = m.group(2).strip()
        if is_last:
            replacement = f"s = {expr}\n    s.name = '{name}'\n    return s"
        else:
            replacement = f"return pd.Series(np.nan, index=df.index, name='{name}')"
        new_code = new_code[:m.start()] + replacement + new_code[m.end():]

    # Fix .iloc[-1] patterns: in vectorized mode, we work with the full Series, not just the last row
    # df['col'].iloc[-1] → df['col']  (full Series, not scalar)
    # df.iloc[-1] → df  (full DataFrame, not last row)
    new_code = re.sub(r"(\w+)\['\w+'\]\.iloc\[-1\]", lambda m: m.group(0).replace(".iloc[-1]", ""), new_code)
    new_code = re.sub(r"(\w+)\.iloc\[-1\]", lambda m: m.group(0).replace(".iloc[-1]", ""), new_code)

    return new_code


def convert_factor(code_path: Path) -> bool:
    code = code_path.read_text(encoding="utf-8")
    has_old = "def calc_factor_single_stock" in code
    has_new = "def calc_factor_series" in code
    if not has_old and not has_new:
        print(f"  ⏭️  No factor function found")
        return False

    # Extract lookback and load_cols
    lookback_m = re.search(r'LOOKBACK_DAYS\s*=\s*(\d+)', code)
    lookback = int(lookback_m.group(1)) if lookback_m else 20
    cols_m = re.search(r'_LOAD_COLS\s*=\s*(\[[^\]]*\])', code)
    load_cols = None
    if cols_m:
        try:
            load_cols = ast.literal_eval(cols_m.group(1))
        except:
            pass

    # Extract factor name from file name or code
    factor_name = code_path.parent.name

    if has_old:
        func_code = extract_user_func(code, "calc_factor_single_stock")
        if not func_code:
            print(f"  ❌ Cannot extract calc_factor_single_stock")
            return False
        series_code = convert_single_stock_to_series(func_code)
    elif has_new:
        func_code = extract_user_func(code, "calc_factor_series")
        if not func_code:
            print(f"  ❌ Cannot extract calc_factor_series")
            return False
        if is_buggy(func_code):
            print(f"  🔧 Fixing buggy calc_factor_series...")
            series_code = fix_buggy_series(func_code)
        else:
            print(f"  ⏭️  Already correct")
            return False

    # Validate syntax
    try:
        ast.parse(series_code)
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        print(f"  Code:\n{series_code}")
        return False

    # Rebuild with new template
    wrapped = FactorFBWorkspace._build_factor_code(
        FactorFBWorkspace.DAILY_FRAMEWORK_TEMPLATE, series_code, lookback, load_cols
    )

    # Write to literature_reports
    rel_path = code_path.relative_to(BASE)
    report_name = rel_path.parent.parent.name
    factor_dir_name = rel_path.parent.name
    lit_path = BASE / report_name / factor_dir_name / f"{factor_name}.code.py"
    lit_path.write_text(wrapped, encoding="utf-8")

    # Write to 文献因子_全量
    full_path = FULL_BASE / report_name / factor_dir_name / f"{factor_name}.code.py"
    full_written = False
    if full_path.parent.exists():
        full_path.write_text(wrapped, encoding="utf-8")
        full_written = True

    print(f"  ✅ {report_name}/{factor_name}: lookback={lookback}, full={full_written}")
    return True


def main():
    args = sys.argv[1:]
    factor_filter = None
    if args and args[0] == "--factor":
        factor_filter = args[1] if len(args) > 1 else None

    # Collect all daily factors
    daily_files = []
    for cp in sorted(BASE.rglob("*.code.py")):
        if ".code.code.py" in cp.name:
            continue
        content = cp.read_text(encoding="utf-8")
        if "calc_factor_single_stock" not in content and "calc_factor_series" not in content:
            continue
        if factor_filter and factor_filter not in cp.name:
            continue
        daily_files.append(cp)

    print(f"Found {len(daily_files)} daily factors to process")

    success = fail = 0
    for cp in daily_files:
        print(f"\n📄 {cp.parent.parent.name}/{cp.parent.name}")
        if convert_factor(cp):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"Done: {success} converted, {fail} failed")


if __name__ == "__main__":
    main()