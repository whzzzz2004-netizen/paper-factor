#!/usr/bin/env python3
"""测试 block_datascan.py 钩子的拦截/放行行为。"""
import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).parent / "block_datascan.py"
DR = "/mnt/d/" + "paper-factor-data"  # 拼接，避免本文件内容被钩子误判


def run(cmd: str) -> str:
    p = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}}),
        capture_output=True, text=True,
    )
    out = p.stdout.strip()
    if not out:
        return "放行"
    d = json.loads(out)["hookSpecificOutput"]
    return f"拦截:{d['permissionDecisionReason'][:20]}"


CASES = [
    # ── 真凶：实测跑了 64 分钟的整盘扫描 ──
    ('grep -rn "CROSS_SECTION_FRAMEWORK_TEMPLATE" / --include=*.py -l 2>/dev/null | grep -v helper | head -10', "拦"),
    ("grep -rn 'foo' / --include=*.py", "拦"),
    ("grep -rn 'foo' /home --include=*.py", "拦"),
    ("grep -rn 'foo' /mnt --include=*.py", "拦"),
    ("rg 'foo' /", "拦"),
    ("ls -R /", "拦"),
    ("find / -name '*.py'", "拦"),
    ("find $HOME -name '*.py'", "拦"),
    ("grep -rn 'x' ~/ -l", "拦"),
    # ── 数据仓库全仓扫描 ──
    (f"grep -r 'size' {DR}", "拦"),
    (f"grep -R --include='*.py' foo {DR}/数据仓库", "拦"),
    (f"rg 'market_cap' {DR}", "拦"),
    (f"ls -R {DR}/数据仓库", "拦"),
    (f"find {DR} -name '*.parquet'", "拦"),
    (f"tree {DR}", "拦"),
    # ── 合法用法（应放行）──
    ("grep -n 'show-columns' scripts/claude_factor_helper.py", "放"),
    ("grep -rn 'CROSS_SECTION' scripts/", "放"),
    (f"ls {DR}/papers/inbox/*.pdf", "放"),
    (f"find {DR}/papers -maxdepth 2 -name '*.pdf'", "放"),
    ("python3 scripts/claude_factor_helper.py show-columns --type daily_single", "放"),
    (f"python3 -c \"import json; json.load(open('{DR}/schema.json'))\"", "放"),
    ("grep -n 'foo' scripts/run_all.py", "放"),
    (f"du -sh {DR}/数据仓库", "放"),
    ("ls -la .claude/hooks/", "放"),
]

if __name__ == "__main__":
    ok = True
    for cmd, want in CASES:
        got = run(cmd)
        good = got.startswith(want)
        ok = ok and good
        print(f"{'✅' if good else '❌'} [{got:<26}] {cmd[:60]}")
    print()
    print("钩子行为符合预期" if ok else "⚠️ 有用例不符合预期")
