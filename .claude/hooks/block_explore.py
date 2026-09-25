#!/usr/bin/env python3
"""PreToolUse 钩子：拦截 Phase 2 因子 agent 的「无效探索」命令。

背景（实测）：
  每次工具调用 = 一次完整 API 往返 ≈ 20 秒。计算本身只要 20-60 秒。
  曾有一个截面因子 agent 花 990 秒，其中 39 次调用在探索、只有 3 次在干活，
  实际计算仅占 17 秒。探索内容：读别的因子代码、grep helper 源码、
  用 pyarrow 读 parquet schema、反复 ls 因子产出目录。

本钩子在命令执行前 deny，彻底杜绝（不依赖 agent 是否遵守提示词）。
"""

import json
import re
import sys

# 数据仓库/项目内的「不该被探索」的目标
FACTOR_OUT = "因子产出"
HELPER_SRC = "claude_factor_helper.py"
TEMPLATE_SRC = "factor_coder/factor.py"
DATA_ROOT_MARKERS = ("paper-factor-data",)

# ── Read / Grep / Glob 工具：按 file_path / pattern 拦截 ──
READ_BLOCK = [
    (re.compile(r"因子产出/.*\.code\.py$"), "读其他因子的 .code.py（不需要参考别人的实现）"),
    (re.compile(r"因子产出/.*\.meta\.json$"), "读其他因子的 .meta.json"),
    (re.compile(r"因子产出/.*\.parquet$"), "读因子产出 parquet（不需要验证产出）"),
    (re.compile(re.escape(HELPER_SRC) + r"$"), "读 helper 源码（命令用法本文档已给全）"),
    (re.compile(r"factor_coder/factor\.py$"), "读模板源码（函数签名本文档已给全）"),
]

# ── Bash 命令：按命令串拦截 ──
# ⚠️ 只拦「读文件内容」类，不拦 ls/find 列目录——主 agent 需要 ls extracted_reports/
BASH_BLOCK = [
    # 读 helper / 模板源码
    (re.compile(r"\b(grep|rg|cat|sed|head|tail|less)\b[^\n|;]*claude_factor_helper\.py"),
     "grep/读 helper 源码"),
    (re.compile(r"\b(grep|rg|cat|sed|head|tail|less)\b[^\n|;]*factor_coder/factor\.py"),
     "grep/读模板源码"),
    # 直接读 parquet 看 schema / 数据
    (re.compile(r"read_schema\s*\("), "用 pyarrow 读 parquet schema（字段核对只用 show-columns）"),
    (re.compile(r"read_parquet\s*\("), "直接用 pandas/pyarrow 读 parquet（字段核对只用 show-columns）"),
    # 读别的因子代码 / 元数据（只拦读内容，不拦 ls）
    (re.compile(r"\b(cat|head|tail|less|sed|grep)\b[^\n|;]*\.code\.py"), "读因子 .code.py"),
    (re.compile(r"\b(cat|head|tail|less|sed)\b[^\n|;]*" + FACTOR_OUT + r"[^\n|;]*\.json"),
     "读因子产出 json"),
]

ALLOW_HINT = (
    "本次调用被拦截：Phase 2 只做「写核心函数 → test-and-export → deploy-to-full」三件事。"
    "每次工具调用约 20 秒，探索会浪费 5-15 分钟。"
    "字段核对只用 `python scripts/claude_factor_helper.py show-columns --type daily_single`"
    "（或 --type minute）；命令用法与函数签名以 .claude/skills/factor/phase2_prompt.md 为准，"
    "不要去翻源码或别的因子实现。"
)


def check_bash(cmd: str) -> str | None:
    for pat, desc in BASH_BLOCK:
        if pat.search(cmd):
            return desc
    return None


def check_read(path: str) -> str | None:
    for pat, desc in READ_BLOCK:
        if pat.search(path):
            return desc
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool = payload.get("tool_name") or ""
    ti = payload.get("tool_input") or {}
    reason = None

    if tool == "Bash":
        reason = check_bash(ti.get("command", "") or "")
    elif tool in ("Read", "Grep", "Glob"):
        target = ti.get("file_path") or ti.get("path") or ti.get("pattern") or ""
        reason = check_read(str(target))
    else:
        return 0

    if reason:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"已拦截：{reason}。{ALLOW_HINT}",
            }
        }
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
