#!/usr/bin/env python3
"""PreToolUse 钩子：拦截 Phase 2 因子 agent 的「无效探索」命令。

⚠️ 本钩子只在 `/factor` 运行期间生效 —— 见 factor_gate.py（哨兵文件 /tmp/factor_hooks_on）。
平时改代码不受限制。

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

from factor_gate import active as _gate_active

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
    # ── 用 python 读源码绕过上面两条（实测 29 次，全在最贵的 worker 上）──
    # 模板源码在 CLI 里从不作为参数出现，所以「命令里出现该路径」本身就是探索
    (re.compile(r"factor_coder/factor\.py"),
     "用 python/其他方式读模板源码（签名以 knowledge/*.md 为准）"),
    # helper 源码：合法调用形如 `python scripts/claude_factor_helper.py <子命令>`。
    # 「读内容」的用法（open/read_text/.read/import）里，路径可能被赋给变量跨行使用，
    # 所以只要命令里同时出现 helper 路径 + 读操作符，就判为探索。
    (re.compile(r"claude_factor_helper\.py[\s\S]*(open\(|read_text|\.read\(|importlib)"),
     "用 python 读 helper 源码（命令用法文档已给全）"),
    (re.compile(r"(open\(|read_text|importlib)[\s\S]*claude_factor_helper\.py"),
     "用 python 读 helper 源码（命令用法文档已给全）"),
    (re.compile(r"(import|from)\s+[^\n|;]*\bclaude_factor_helper\b"),
     "import helper 源码（命令用法文档已给全）"),
    # 直接读 parquet 看 schema / 数据
    (re.compile(r"read_schema\s*\("), "用 pyarrow 读 parquet schema（字段核对只用 show-columns）"),
    (re.compile(r"read_parquet\s*\("), "直接用 pandas/pyarrow 读 parquet（字段核对只用 show-columns）"),
    # 读别的因子代码 / 元数据（只拦读内容，不拦 ls）
    (re.compile(r"\b(cat|head|tail|less|sed|grep)\b[^\n|;]*\.code\.py"), "读因子 .code.py"),
    (re.compile(r"\b(cat|head|tail|less|sed)\b[^\n|;]*" + FACTOR_OUT + r"[^\n|;]*\.json"),
     "读因子产出 json"),
    # 丢后台 + 轮询（实测最贵的一类浪费：不推进任务，却每轮重发整个上下文）
    # 1) shell sleep —— time.sleep( 不匹配（否定后视 .）
    (re.compile(r"(?<![\w.])sleep\s+[\d$]"),
     "sleep 轮询后台任务（等待不推进任务，却每轮重发整个上下文，实测可致 token 翻 11 倍）"),
    (re.compile(r"(?<![\w.])sleep\s+\$\{"), "sleep 轮询后台任务"),
    # 2) 反复读后台任务输出文件
    (re.compile(r"\b(cat|tail|head|less|grep|rg|sed)\b[^\n|;]*tasks/[^\s'\"]*\.output"),
     "读后台任务 .output 轮询（改用 Bash timeout 参数让命令一次跑完）"),
    (re.compile(r"\b(cat|tail|head|less|grep|rg|sed)\b[^\n|;]*\.output\b"),
     "读后台任务 .output 轮询"),
]

ALLOW_HINT = (
    "本次调用被拦截：Phase 2 只做「写核心函数 → test-and-export → deploy-to-full」三件事。"
    "每次工具调用约 20 秒，探索会浪费 5-15 分钟。"
    "字段核对只用 `python scripts/claude_factor_helper.py show-columns --type daily_single`"
    "（或 --type minute）；命令用法与函数签名以 .claude/skills/factor/phase2_check.md 和 "
    ".claude/skills/factor/phase2_code.md 为准，"
    "不要去翻源码或别的因子实现。"
    "想验证假设（窗口长度 / lookback / 字段可见性）用 "
    "`test-and-export --dry-run`（只跑测试、不落盘、回传诊断），"
    "**不要**用假报告名（pk*/probe*/v*）反复 test-and-export —— 那会在产出目录留下空壳，"
    "也**不要**用 python 的 open/read_text 去读 helper 或模板源码绕过本钩子。"
    "长命令（test-and-export）请把 Bash 工具的 timeout 参数设为 300000 毫秒一次跑完；"
    "命令被转后台或超时后，按 phase2_code.md「超时熔断」改代码重跑，"
    "严禁 sleep / 反复读 .output 等待。"
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
    if not _gate_active():
        return 0
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
