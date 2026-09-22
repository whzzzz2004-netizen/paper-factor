#!/usr/bin/env python3
"""PreToolUse 钩子：拦截对数据仓库的「全仓递归扫描」命令。

背景：/mnt/d/paper-factor-data 有 66GB / 143,797 个文件。
`grep -r` / `ls -R` / 无界 `find` 这类命令会：
  - 逐字节读 66GB parquet 二进制 → 永远读不完
  - 产生海量输出灌爆 agent 上下文
  - 卡住超过 600s → 触发流看门狗，agent 被判定 stalled 杀死

本钩子在命令执行前 deny，彻底杜绝（不依赖 agent 是否遵守 prompt）。

输入：stdin JSON（Claude Code hook 协议）
输出：JSON，permissionDecision=deny 时阻止执行
"""

import json
import re
import sys

DATA_ROOT_MARKERS = ("paper-factor-data",)

# 宽根路径：递归扫这些等于扫全盘（曾实测 grep -rn ... / 跑了 64 分钟）
BROAD_ROOTS = [
    re.compile(r"(?<![\w./])/(?![\w./])"),      # 裸根 /（前后是空白或行尾）
    re.compile(r"/mnt(?![\w./-])"),              # /mnt 本身
    re.compile(r"/home(?![\w./-])"),
    re.compile(r"/usr(?![\w./-])"),
    re.compile(r"/opt(?![\w./-])"),
    re.compile(r"/var(?![\w./-])"),
    re.compile(r"/etc(?![\w./-])"),
    re.compile(r"~/"),                            # 家目录
    re.compile(r"\$HOME"),
]

# 真正致命的模式：递归 + 数据根
DANGEROUS = [
    # grep 递归：兼容 -r / -R / -rn / -rE 等组合短选项，以及 --recursive
    (re.compile(r"\bgrep\b[^\n|;]*\s-[A-Za-z]*[rR][A-Za-z]*(\s|$)"), "grep 递归扫描"),
    (re.compile(r"\bgrep\b[^\n|;]*--recursive"), "grep 递归扫描"),
    # ripgrep（默认递归）
    (re.compile(r"\brg\b"), "ripgrep 递归扫描"),
    # ls -R
    (re.compile(r"\bls\b[^\n|;]*-[A-Za-z]*R"), "ls -R 递归列目录"),
    # find 无 -maxdepth 限制
    (re.compile(r"\bfind\b(?!.*-maxdepth)"), "find 无 -maxdepth 限制"),
    # 递归 du / tree
    (re.compile(r"\bdu\b[^\n|;]*(-a\b|--all)"), "du -a 递归"),
    (re.compile(r"\btree\b"), "tree 递归"),
]

ALLOW_HINT = (
    "禁止对数据仓库/整个文件系统做全仓递归扫描（66GB/14万文件，曾导致 agent 卡死或空转 64 分钟）。"
    "判断数据列请改用：python3 scripts/claude_factor_helper.py show-columns --type daily_single "
    "（或 --type minute）。查代码请限定到具体目录，如 grep -n 'xxx' scripts/*.py。"
)


def is_dangerous(cmd: str) -> str | None:
    """返回命中的危险描述，未命中返回 None。"""
    hit = None
    for pat, desc in DANGEROUS:
        if pat.search(cmd):
            hit = desc
            break
    if hit is None:
        return None
    # 命中递归模式后，判断目标是否是「宽根」或数据仓库
    if any(m in cmd for m in DATA_ROOT_MARKERS):
        return hit
    for pat in BROAD_ROOTS:
        if pat.search(cmd):
            return hit + "（目标是宽根路径）"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # 解析失败不拦截

    if payload.get("tool_name") != "Bash":
        return 0

    cmd = (payload.get("tool_input") or {}).get("command", "") or ""
    if not cmd:
        return 0

    desc = is_dangerous(cmd)
    if desc:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"已拦截：{desc}。{ALLOW_HINT}",
            }
        }
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
