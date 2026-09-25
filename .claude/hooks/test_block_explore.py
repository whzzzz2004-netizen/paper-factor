#!/usr/bin/env python3
"""测试 block_explore.py 钩子的拦截/放行行为。"""
import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).parent / "block_explore.py"
DR = "/mnt/d/" + "paper-factor-data"  # 拼接，避免本文件内容被钩子误判
FO = "因子" + "产出"                   # 同上


def run_bash(cmd: str) -> str:
    p = subprocess.run([sys.executable, str(HOOK)],
                       input=json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}}),
                       capture_output=True, text=True)
    return _verdict(p.stdout)


def run_read(path: str) -> str:
    p = subprocess.run([sys.executable, str(HOOK)],
                       input=json.dumps({"tool_name": "Read", "tool_input": {"file_path": path}}),
                       capture_output=True, text=True)
    return _verdict(p.stdout)


def _verdict(out: str) -> str:
    out = out.strip()
    if not out:
        return "放行"
    d = json.loads(out)["hookSpecificOutput"]
    return f"拦截:{d['permissionDecisionReason'][:18]}"


BASH_CASES = [
    # 应拦截
    ("grep -n 'lookback' scripts/claude_factor_helper.py", "拦"),
    ("sed -n '1,50p' scripts/claude_factor_helper.py", "拦"),
    ("grep -n 'CROSS_SECTION' rdagent/components/coder/factor_coder/factor.py", "拦"),
    (f'python -c "import pyarrow.parquet as pq; print(pq.read_schema(\'{DR}/x.parquet\'))"', "拦"),
    (f'python -c "import pandas as pd; df=pd.read_parquet(\'{DR}/x.parquet\')"', "拦"),
    (f"cat {DR}/数据仓库/{FO}/测试/a/b.code.py", "拦"),
    (f"cat {DR}/数据仓库/{FO}/测试/a/b.meta.json", "拦"),
    # 应放行（ls 列目录是主 agent 需要的，不拦）
    ("python scripts/claude_factor_helper.py show-columns --type daily_single", "放"),
    ("python scripts/claude_factor_helper.py show-columns --type minute", "放"),
    ("python scripts/claude_factor_helper.py test-and-export --code /tmp/factor_X.py --type cross_section", "放"),
    ("python scripts/claude_factor_helper.py deploy-to-full --code /tmp/a.code.py", "放"),
    ("grep -n 'foo' /tmp/factor_X.py", "放"),
    ("cat /tmp/factor_X.py", "放"),
    ("ls /mnt/d/paper-factor-data/papers/inbox/", "放"),
    (f"ls {DR}/数据仓库/{FO}/测试/2026-09-23/", "放"),
    (f"ls {DR}/数据仓库/{FO}/extracted_reports/2026-09-23/", "放"),
    (f"find {DR}/数据仓库/{FO} -maxdepth 2 -type d", "放"),
]

READ_CASES = [
    (f"{DR}/数据仓库/{FO}/测试/2026-09-23/r/F/F.code.py", "拦"),
    (f"{DR}/数据仓库/{FO}/测试/2026-09-23/r/F/F.meta.json", "拦"),
    (f"{DR}/数据仓库/{FO}/测试/2026-09-23/r/F/F.parquet", "拦"),
    ("/home/dministrator/paper-factor/scripts/claude_factor_helper.py", "拦"),
    ("/home/dministrator/paper-factor/rdagent/components/coder/factor_coder/factor.py", "拦"),
    # 应放行
    ("/tmp/factor_X.py", "放"),
    ("/home/dministrator/paper-factor/.claude/skills/factor/phase2_prompt.md", "放"),
    ("/home/dministrator/paper-factor/.claude/skills/factor/knowledge/daily.md", "放"),
    ("/mnt/d/paper-factor-data/papers/inbox/a.pdf", "放"),
]

if __name__ == "__main__":
    ok = True
    print("── Bash 用例 ──")
    for cmd, want in BASH_CASES:
        got = run_bash(cmd)
        good = got.startswith(want)
        ok = ok and good
        print(f"{'✅' if good else '❌'} [{got:<24}] {cmd[:66]}")
    print("\n── Read 用例 ──")
    for path, want in READ_CASES:
        got = run_read(path)
        good = got.startswith(want)
        ok = ok and good
        print(f"{'✅' if good else '❌'} [{got:<24}] {path[:66]}")
    print()
    print("钩子行为符合预期" if ok else "⚠️ 有用例不符合预期")
