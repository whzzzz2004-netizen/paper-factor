#!/usr/bin/env python3
"""测试 block_explore.py 钩子的拦截/放行行为。

钩子只在 `/factor` 期间生效（哨兵 /tmp/factor_hooks_on），
所以本测试先启用哨兵、跑完再关掉。
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import factor_gate

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
    # 用 python 读源码绕过 cat/grep（实测 29 次）—— 应拦截
    ("python3 -c \"src=open('rdagent/components/coder/factor_coder/factor.py').read(); print(src[:100])\"", "拦"),
    ("python3 - <<'EOF'\nsrc=open('/home/dministrator/paper-factor/rdagent/components/coder/factor_coder/factor.py').read()\nprint(src.find('CROSS_SECTION'))\nEOF", "拦"),
    ("python3 -c \"src=open('scripts/claude_factor_helper.py').read(); print(src[9000:14000])\"", "拦"),
    ("python3 - <<'EOF'\np='scripts/claude_factor_helper.py'\nprint(open(p).read_text())\nEOF", "拦"),
    ("python3 -c \"from scripts import claude_factor_helper as h; print(dir(h))\"", "拦"),
    (f"cat {DR}/数据仓库/{FO}/测试/a/b.code.py", "拦"),
    (f"cat {DR}/数据仓库/{FO}/测试/a/b.meta.json", "拦"),
    # sleep / 轮询后台 .output —— 应拦截
    ("sleep 170; tail -35 /tmp/claude-1000/x/tasks/bwktlearv.output", "拦"),
    ("sleep 60", "拦"),
    ("sleep 115; cat /tmp/x/tasks/be2zfy0tj.output", "拦"),
    ("cat /tmp/claude-1000/x/tasks/be2zfy0tj.output", "拦"),
    ("tail -25 /tmp/claude-1000/x/tasks/bcnb4e424.output", "拦"),
    ("grep -c 'Error' /tmp/x/tasks/foo.output", "拦"),
    # 应放行
    ("timeout 300 python scripts/claude_factor_helper.py test-and-export --code /tmp/f.py", "放"),
    # 合法子命令调用（含 helper 路径但只是执行）—— 应放行
    ("python scripts/claude_factor_helper.py test-and-export --code /tmp/f.py --dry-run", "放"),
    ("python scripts/claude_factor_helper.py show-columns --type daily_single", "放"),
    ("python scripts/claude_factor_helper.py deploy-to-full --code /tmp/a.code.py", "放"),
    ("python3 scripts/claude_factor_helper.py extract-pdf a.pdf --outdir /tmp/o", "放"),
    ("python -c \"import time; time.sleep(1)\"", "放"),
    ("python3 -c \"from time import sleep; sleep(2)\"", "放"),
    ("ls /tmp/claude-1000/x/tasks/", "放"),
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
    ("/home/dministrator/paper-factor/.claude/skills/factor/phase2_check.md", "放"),
    ("/home/dministrator/paper-factor/.claude/skills/factor/phase2_code.md", "放"),
    ("/home/dministrator/paper-factor/.claude/skills/factor/knowledge/daily.md", "放"),
    ("/mnt/d/paper-factor-data/papers/inbox/a.pdf", "放"),
]

if __name__ == "__main__":
    factor_gate.enable()
    try:
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

        print("\n── 哨兵关闭后应一律放行 ──")
        factor_gate.disable()
        for cmd, _ in BASH_CASES[:4]:
            got = run_bash(cmd)
            good = got == "放行"
            ok = ok and good
            print(f"{'✅' if good else '❌'} [{got:<24}] {cmd[:66]}")

        print()
        print("钩子行为符合预期" if ok else "⚠️ 有用例不符合预期")
    finally:
        factor_gate.disable()
