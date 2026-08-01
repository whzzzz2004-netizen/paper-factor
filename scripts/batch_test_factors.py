#!/usr/bin/env python3
"""
Batch test 4 representative factors (one per template type) with the new templates.
Uses test data (300 stocks) and verifies non-null ratio >= 1%.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_DATA_DIR = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data_1000"
LITERATURE_BASE = PROJECT_ROOT / "git_ignore_folder" / "factor_outputs" / "literature_reports" / "20260727"

FACTORS = [
    # (type, report, factor)
    ("daily_single", "华泰单因子测试之动量类因子", "return_1m"),
    ("cross_section", "小隐于野大隐于市低波因子的进阶", "vol20"),
    ("minute", "日内交易特征稳定性与股票收益", "SDRSKEW"),
    ("minute_cs", "基于资金推动力的价量张力因子构建", "弹力势差"),
]


def test_factor(ftype, report, factor_name):
    code_path = LITERATURE_BASE / report / factor_name / f"{factor_name}.code.py"
    if not code_path.exists():
        return {"success": False, "error": f"code file not found: {code_path}"}

    env = os.environ.copy()
    env["FACTOR_DATA_DIR"] = str(TEST_DATA_DIR)
    env["FACTOR_N_WORKERS"] = "4"

    tmpdir = Path(tempfile.mkdtemp(prefix=f"factor_test_{factor_name}_"))
    t0 = time.time()

    try:
        proc = subprocess.run(
            [sys.executable, str(code_path)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        elapsed = time.time() - t0
        result_path = tmpdir / "result.parquet"
        # The factor code saves to the code directory, not tempdir
        # Also check the code directory
        code_result = code_path.parent / f"{factor_name}.parquet"

        result_info = {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed": round(elapsed, 1),
            "tmpdir": str(tmpdir),
            "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
        }

        # Check for result parquet
        if code_result.exists():
            import pandas as pd
            df = pd.read_parquet(code_result)
            nn = int(df.notna().sum().sum())
            total = df.size
            nn_ratio = round(nn / total, 4) if total > 0 else 0
            result_info["result_exists"] = True
            result_info["shape"] = list(df.shape)
            result_info["non_null_ratio"] = nn_ratio
            result_info["non_null_count"] = nn
            result_info["total_count"] = total
            result_info["passed"] = nn_ratio >= 0.01
        else:
            # Check if it was saved to tmpdir
            tmp_result = tmpdir / f"{factor_name}.parquet"
            if tmp_result.exists():
                import pandas as pd
                df = pd.read_parquet(tmp_result)
                nn = int(df.notna().sum().sum())
                total = df.size
                nn_ratio = round(nn / total, 4) if total > 0 else 0
                result_info["result_exists"] = True
                result_info["shape"] = list(df.shape)
                result_info["non_null_ratio"] = nn_ratio
                result_info["non_null_count"] = nn
                result_info["total_count"] = total
                result_info["passed"] = nn_ratio >= 0.01
                # Copy to code dir
                import shutil
                shutil.copy2(tmp_result, code_result)
            else:
                result_info["result_exists"] = False
                result_info["non_null_ratio"] = 0
                result_info["passed"] = False

        return result_info

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "success": False,
            "error": "timeout",
            "elapsed": round(elapsed, 1),
            "tmpdir": str(tmpdir),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "success": False,
            "error": str(e),
            "elapsed": round(elapsed, 1),
            "tmpdir": str(tmpdir),
        }


def main():
    results = {}
    all_passed = True

    for ftype, report, factor_name in FACTORS:
        print(f"\n{'='*60}")
        print(f"Testing [{ftype}] {report}/{factor_name}...")
        print(f"{'='*60}")

        result = test_factor(ftype, report, factor_name)
        results[f"{ftype}/{factor_name}"] = result

        if result.get("passed"):
            print(f"  ✅ PASSED: nn_ratio={result.get('non_null_ratio', 'N/A')}, "
                  f"shape={result.get('shape', 'N/A')}, {result.get('elapsed')}s")
        elif result.get("result_exists"):
            print(f"  ⚠️  LOW: nn_ratio={result.get('non_null_ratio')} (< 1%), "
                  f"shape={result.get('shape')}, {result.get('elapsed')}s")
            all_passed = False
        else:
            print(f"  ❌ FAILED: {result.get('error', 'no result')}, {result.get('elapsed')}s")
            if result.get("stdout_tail"):
                print(f"  stdout: {result['stdout_tail'][-500:]}")
            if result.get("stderr_tail"):
                print(f"  stderr: {result['stderr_tail'][-500:]}")
            all_passed = False

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, result in results.items():
        if result.get("passed"):
            print(f"  ✅ {key}: non_null={result.get('non_null_ratio')}, {result.get('elapsed')}s")
        elif result.get("result_exists"):
            print(f"  ⚠️  {key}: non_null={result.get('non_null_ratio')} (<1%), {result.get('elapsed')}s")
        else:
            print(f"  ❌ {key}: FAILED ({result.get('error', 'no result')})")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())