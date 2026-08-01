#!/usr/bin/env python3
"""
为日线 .code.py 增加防御性处理，防止单只股票异常导致 joblib 线程死锁。

改动 (对 _compute_stock):
  1. _val = _series.loc[td] 之后加 isinstance(_val, pd.Series) 兜底 (重复日期取最后一行)
  2. 整个函数体包 try/except Exception → return [] (单只股票异常不阻塞整批)

同时更新 factor.py 的 DAILY_FRAMEWORK_TEMPLATE 保持一致。
"""
import shutil
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RUN_DIR = PROJ / "git_ignore_folder" / "factor_outputs" / "文献因子_全量" / "20260730"
FACTOR_PY = PROJ / "rdagent" / "components" / "coder" / "factor_coder" / "factor.py"

VAL_GUARD = ('if isinstance(_val, pd.Series):  # 重复日期兜底：取最后一行')


def is_daily(code: str) -> bool:
    return '"stock_data" / "daily"' in code and 'def _compute_stock' in code


def patch_compute_stock(code: str) -> str:
    """对 _compute_stock 函数体应用防御性改动。"""
    lines = code.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("def _compute_stock("):
            # 收集函数体直到下一个列0的非空行
            body = []
            j = i + 1
            while j < len(lines):
                if lines[j].strip() and lines[j][0] != " ":
                    break
                body.append(lines[j])
                j += 1
            while body and not body[-1].strip():
                body.pop()

            # 1) Series 兜底：只在向量化模式中 _val = _series.loc[td] 之后插入
            new_body = []
            for k, bl in enumerate(body):
                new_body.append(bl)
                stripped = bl.strip()
                if stripped == "_val = _series.loc[td]":
                    indent = bl[: len(bl) - len(bl.lstrip())]
                    new_body.append(f"{indent}{VAL_GUARD}")
                    new_body.append(f"{indent}    _val = _val.iloc[-1]")

            # 2) 整个函数体包 try/except
            out.append(line)
            out.append("    try:")
            for bl in new_body:
                if bl.strip():
                    out.append("    " + bl)
                else:
                    out.append("")
            out.append("    except Exception:")
            out.append("        return []  # 单只股票异常不阻塞整批，避免 joblib 线程卡死")
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def main():
    daily_files = [f for f in sorted(RUN_DIR.rglob("*.code.py"))
                   if is_daily(f.read_text(encoding="utf-8"))]
    print(f"日线 .code.py: {len(daily_files)} 个")
    for f in daily_files:
        orig = f.read_text(encoding="utf-8")
        patched = patch_compute_stock(orig)
        if patched != orig:
            bak = f.with_suffix(f.suffix + ".bak_defensive")
            if not bak.exists():
                shutil.copy2(f, bak)
            f.write_text(patched, encoding="utf-8")
            print(f"  ✅ patched: {f.relative_to(PROJ)}")
        else:
            print(f"  ⏭️ no change: {f.relative_to(PROJ)}")

    # 更新模板 factor.py (DAILY_FRAMEWORK_TEMPLATE)
    tp = FACTOR_PY.read_text(encoding="utf-8")
    old_block = """def _compute_stock(stock, _LOAD_COLS=None):
    df = load_stock(stock, _LOAD_COLS)
    if df.empty:
        return []
    results = []
    _td_index = pd.DatetimeIndex(TRADE_DATES)
    _series = calc_factor_series(df, stock)
    if _series is not None and isinstance(_series, pd.Series) and len(_series) > 0:
        _fname = _series.name if _series.name else 'factor'
        for i, td in enumerate(_td_index):
            if i < LOOKBACK_DAYS:
                continue
            if td in _series.index:
                _val = _series.loc[td]
                if not (np.isnan(_val) or np.isinf(_val)):
                    results.append({{"datetime": str(td.date()), "instrument": stock, _fname: float(_val)}})
        return results
    return results"""
    new_block = """def _compute_stock(stock, _LOAD_COLS=None):
    try:
        df = load_stock(stock, _LOAD_COLS)
        if df.empty:
            return []
        results = []
        _td_index = pd.DatetimeIndex(TRADE_DATES)
        _series = calc_factor_series(df, stock)
        if _series is not None and isinstance(_series, pd.Series) and len(_series) > 0:
            _fname = _series.name if _series.name else 'factor'
            for i, td in enumerate(_td_index):
                if i < LOOKBACK_DAYS:
                    continue
                if td in _series.index:
                    _val = _series.loc[td]
                    if isinstance(_val, pd.Series):  # 重复日期兜底：取最后一行
                        _val = _val.iloc[-1]
                    if not (np.isnan(_val) or np.isinf(_val)):
                        results.append({{"datetime": str(td.date()), "instrument": stock, _fname: float(_val)}})
            return results
        return results
    except Exception:
        return []  # 单只股票异常不阻塞整批，避免 joblib 线程卡死"""
    if old_block in tp:
        tp = tp.replace(old_block, new_block)
        FACTOR_PY.write_text(tp, encoding="utf-8")
        print("  ✅ factor.py DAILY_FRAMEWORK_TEMPLATE updated")
    else:
        print("  ⚠️ 模板 pattern 未匹配，跳过 factor.py 更新")


if __name__ == "__main__":
    main()
