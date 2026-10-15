#!/usr/bin/env python3
"""测试 claude_factor_helper 的探测硬闸（_probe_gate）。

背景：不受限的 worker 会把 test-and-export 当网格搜索用 —— 实测单个因子跑 32 次、
烧 4.5M token（正常 worker 的 13 倍）。硬闸保证这个上限不靠 agent 自觉。

跑法：python3 scripts/test_claude_factor_helper.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import claude_factor_helper as H  # noqa: E402

CODE = Path("/tmp/factor_TestFactor.py")
ok = True


def probe_report(name: str, results: list) -> None:
    global ok
    print(f"\n── {name} ──")
    for step, (kind, expect_block, got) in enumerate(results, 1):
        good = bool(got) == expect_block
        ok = ok and good
        print(f"{'✅' if good else '❌'} 第{step}次 {kind:<8} "
              f"{'拦截' if got else '放行'}（预期{'拦截' if expect_block else '放行'}）")


def args(date, report, factor, dry):
    a = type("A", (), {})()
    a.date, a.report, a.factor, a.dry_run = date, report, factor, dry
    return a


def run(factor, dry_flags):
    out = []
    for dry in dry_flags:
        err = H._probe_gate(args("2026-01-01", "测试报告", factor, dry), CODE)
        out.append(("dry-run" if dry else "正式", bool(err)))
    return out


# 计数文件全部清掉，从干净状态开始
for p in Path("/tmp").glob(".factor_probe_*"):
    p.unlink()

# 1) 正式调用：前 6 次放行，第 7 次起拦截
res = run("TotalCap", [False] * 8)
probe_report("正式调用上限 = %d" % H.MAX_TEST_EXPORT,
             [(k, i >= H.MAX_TEST_EXPORT, e) for i, (k, e) in enumerate(res)])

# 2) dry-run：前 3 次放行，第 4 次起拦截
res = run("DryCap", [True] * 5)
probe_report("dry-run 上限 = %d" % H.MAX_DRY_RUN,
             [(k, i >= H.MAX_DRY_RUN, e) for i, (k, e) in enumerate(res)])

# 3) 混合：3 次 dry-run + 3 次正式 = 6 次到顶，第 7 次拦截
res = run("Mixed", [True] * 3 + [False] * 4)
probe_report("混合调用（3 dry + 4 正式）",
             [(k, i >= H.MAX_TEST_EXPORT, e) for i, (k, e) in enumerate(res)])

# 4) 隔离性：另一个因子的计数不受上面影响
err = H._probe_gate(args("2026-01-01", "测试报告", "FreshFactor", False), CODE)
good = err is None
ok = ok and good
print(f"\n{'✅' if good else '❌'} 新因子独立计数（应为放行）")

# 5) dry-run 与正式调用共用一个计数：忘传 --date 也绕不开上限
#    （历史 bug：双方 key 不同 → 3 次 dry-run + 6 次正式 = 9 次）
for _ in range(H.MAX_TEST_EXPORT):
    H._probe_gate(args("2026-01-01", "测试报告", "SharedKey", True), CODE)
    H._probe_gate(args(None, "测试报告", "SharedKey", False), CODE)   # 不带 --date
err = H._probe_gate(args(None, "测试报告", "SharedKey", False), CODE)
good = err is not None
ok = ok and good
print(f"\n{'✅' if good else '❌'} 不带 --date 的调用与带 --date 共用计数（应为拦截）")

# 5b) 若只统计 dry-run 是否也没绕开（dry 上限独立生效）
for _ in range(H.MAX_DRY_RUN):
    H._probe_gate(args(None, "测试报告", "DryShared", True), CODE)
err = H._probe_gate(args("2026-01-01", "测试报告", "DryShared", True), CODE)
good = err is not None
ok = ok and good
print(f"{'✅' if good else '❌'} dry-run 上限跨 --date 也生效（应为拦截）")

# 6) TTL：超龄计数文件自动清零
import os
import time
stale = Path("/tmp/.factor_probe_2026-01-03_测试报告_StaleFactor.json")
stale.write_text('{"dry": 99, "total": 99}', encoding="utf-8")
old = time.time() - H.PROBE_TTL_SECONDS - 60
os.utime(stale, (old, old))
err = H._probe_gate(args("2026-01-03", "测试报告", "StaleFactor", False), CODE)
good = err is None
ok = ok and good
print(f"{'✅' if good else '❌'} 超龄计数文件自动清零（应为放行）")

for p in Path("/tmp").glob(".factor_probe_*"):
    p.unlink()

print()
print("探测硬闸行为符合预期" if ok else "⚠️ 有用例不符合预期")
sys.exit(0 if ok else 1)
