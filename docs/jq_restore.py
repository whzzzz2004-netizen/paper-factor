#!/usr/bin/env python3
"""在「撤下 jq 文档」和「恢复 jq 文档」两个状态之间切换。

用法:
    python docs/jq_restore.py            # 恢复：加回 get_jq_data 说明，删掉「本地不可用」提示
    python docs/jq_restore.py --revert   # 撤回：删掉 get_jq_data 说明，加回「本地不可用」提示
    python docs/jq_restore.py --check    # 只检查当前处于哪个状态

设计要点：
  - 每处改动都是一个**连续文本块**的精确替换，两个方向互为逆操作。
  - 因此 revert(restore(x)) 与原文件**逐字节一致**（已在 /tmp 备份上验证）。
  - 幂等：已是目标状态则跳过。

背景见同目录 jq_restore.md。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# 文本块
# ─────────────────────────────────────────────────────────────────────────────
JQ_HELPER = """\
  get_jq_data(symbol, data_type)  聚宽在线数据，首次调用下载后自动缓存为 parquet，之后读缓存。
                                  data_type='price'            → 指数/个股行情 DataFrame，列 open/close/high/low/volume/money
                                  data_type='index_components' → 指数成分股列表 DataFrame，列 stock
                                  常用指数代码：000300.XSHG(沪深300)、000905.XSHG(中证500)、
                                    000016.XSHG(上证50)、000852.XSHG(中证1000)、000906.XSHG(中证800)、
                                    000001.XSHG(上证指数)、399001.XSHE(深证成指)、399006.XSHE(创业板指)
                                  用途：市场收益、超额收益、Beta、相对强弱、CAPM、指数成分股筛选。
                                  需要有效的 JQ_USER / JQ_PASS 环境变量。"""

JQ_KB = """\
### `get_jq_data(symbol, data_type)` — 聚宽数据（指数行情 / 成分股）

优先读本地缓存 parquet，没有才联网下载（首次调用约数十秒，之后读缓存）。

| data_type | 返回 | 说明 |
|---|---|---|
| `'price'` | DataFrame，列 `open/close/high/low/volume/money` | 指数或个股行情 |
| `'index_components'` | DataFrame，列 `stock` | 指数成分股列表 |

```python
idx = get_jq_data('000300.XSHG', 'price')            # 沪深300 行情
stocks = get_jq_data('000905.XSHG', 'index_components')  # 中证500 成分股
```

常用指数代码：

| 代码 | 指数 | 代码 | 指数 |
|---|---|---|---|
| `000300.XSHG` | 沪深300 | `000001.XSHG` | 上证指数 |
| `000905.XSHG` | 中证500 | `399001.XSHE` | 深证成指 |
| `000016.XSHG` | 上证50 | `399006.XSHE` | 创业板指 |
| `000852.XSHG` | 中证1000 | `000906.XSHG` | 中证800 |
| `000688.XSHG` | 科创50 | `932056.XSHG` | 中证2000 |

- 用途：市场收益、超额收益、Beta、相对强弱、CAPM、指数成分股筛选
- 需要有效的 `JQ_USER` / `JQ_PASS` 环境变量
- ⚠️ **前瞻约束照旧**：按 T 日截断（`idx[idx.index <= trade_date]`），不得使用 T 日之后的数据"""

NOTE_HELPER = "注：需要指数行情 / 指数成分股等其他在线数据时，本地不可用；按缺字段处理即可。"
NOTE_KB = ("**指数行情、指数成分股等其他在线数据本地不可用**，"
           "需要时按缺字段处理（写 `{name}.missing.json`），不要用相近数据代理。")

# ─────────────────────────────────────────────────────────────────────────────
# PAIR_EDITS: (文件, 撤下态文本, 恢复态文本)  —— 精确连续块，互为逆操作
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_HELPER = '                                  用法：industry = INDUSTRY_DICT.get(stock, "未知")'
ANCHOR_KB_USAGE = '- 用法：`industry = INDUSTRY_DICT.get(stock, "未知")`'
ANCHOR_KB_COVER = '- 覆盖：测试集 292/300 只、全量 5207 只；缺失的股票用 `.get(stock, "未知")` 兜底'
ANCHOR_KB_DL = "- 用途：行业中性化、行业 embedding、行业分组特征"
CS_TAIL = "> （如行业动量、行业轮动、行业偏离度）时需要用到 `INDUSTRY_DICT`，而非鼓励做后处理。"

PAIR_EDITS = [
    # ── helper：show-columns 输出 ──
    ("scripts/claude_factor_helper.py",
     ANCHOR_HELPER + "\n\n" + NOTE_HELPER,
     ANCHOR_HELPER + "\n\n" + JQ_HELPER),

    # ── daily.md ──
    (".claude/skills/factor/knowledge/daily.md",
     ANCHOR_KB_COVER + "\n\n" + NOTE_KB,
     ANCHOR_KB_COVER + "\n\n" + JQ_KB),

    # ── minute.md ──
    (".claude/skills/factor/knowledge/minute.md",
     ANCHOR_KB_USAGE + "\n- 用途：行业中性化、行业分组统计、行业内排名、行业动量、行业轮动\n\n" + NOTE_KB,
     ANCHOR_KB_USAGE + "\n- 用途：行业中性化、行业分组统计、行业内排名、行业动量、行业轮动\n\n" + JQ_KB),

    # ── deep_learning.md ──
    (".claude/skills/factor/knowledge/deep_learning.md",
     ANCHOR_KB_DL + "\n\n" + NOTE_KB,
     ANCHOR_KB_DL + "\n\n" + JQ_KB),

    # ── cross_section.md：两处（jq 块插在用法行之后；提示句在节尾）──
    (".claude/skills/factor/knowledge/cross_section.md",
     ANCHOR_KB_COVER + "\n\n",
     ANCHOR_KB_COVER + "\n\n" + JQ_KB + "\n\n"),
    # 注意：两侧必须互不包含（否则撤下态下 revert 会重复插入）。这里把
    # 后续小节标题一起纳入锚点，使 "CS_TAIL\n\n## 特殊约束" 不成为
    # "CS_TAIL\n\nNOTE\n\n## 特殊约束" 的子串。
    (".claude/skills/factor/knowledge/cross_section.md",
     CS_TAIL + "\n\n" + NOTE_KB + "\n\n## 特殊约束",
     CS_TAIL + "\n\n## 特殊约束"),

    # ── phase2_check.md：缺列判定块 ──
    (".claude/skills/factor/phase2_check.md",
     """> ⚠️ **`show-columns` 输出末尾的「额外可用数据」段落也是合法数据源，不算缺字段。**
> 当前它只列出一样东西：`INDUSTRY_DICT[股票代码] = 申万一级行业名`（如 `"银行I"`）——
> **需要「行业分类」时用这个，不要判缺列**。
>
> **除 `INDUSTRY_DICT` 外的其他在线数据（指数行情、指数成分股、市场收益率等）本地不可用**，
> 需要时按缺字段处理，不要用相近数据代理。""",
     """> ⚠️ **`show-columns` 输出末尾的「额外可用数据」段落也是合法数据源，不算缺字段。**
> 它列出的是框架已注入、函数内可直接调用的东西，不属于 parquet 列，所以不在上面的列清单里：
> - `INDUSTRY_DICT[股票代码] = 申万一级行业名`（如 `"银行I"`）—— **需要「行业分类」时用这个，不要判缺列**
> - `get_jq_data(symbol, data_type)` —— 指数行情 / 指数成分股，按需联网并缓存
>   （`data_type='price'` 拿行情；`data_type='index_components'` 拿成分股列表）
>
> 详细用法见你 type 对应的 knowledge md 的「额外可用数据」段。**「行业分类」永远不是缺列理由。**"""),

    # ── phase2_check.md：字段齐全判定 ──
    (".claude/skills/factor/phase2_check.md",
     """「行业分类」用 `INDUSTRY_DICT`（**不要**写进 `--cols`，它不是 parquet 列）。""",
     """「行业分类」用 `INDUSTRY_DICT`（**不要**写进 `--cols`，它不是 parquet 列）；
「指数行情 / 指数成分股」用 `get_jq_data(...)`，同样不写进 `--cols`。"""),

    # ── SKILL.md 规则 19 ──
    (".claude/skills/factor/SKILL.md",
     """    - `INDUSTRY_DICT[股票代码]` → 申万一级行业名（如 `"银行I"`）—— **需要「行业分类」时用它，永远不是缺列理由**；不写进 `--cols`
    - 详细用法见各 type 对应的 `knowledge/*.md` 的「额外可用数据」段
    - **其他在线数据（指数行情、指数成分股、市场收益率等）本地不可用**：按缺字段处理，不要用相近数据代理""",
     """    - `INDUSTRY_DICT[股票代码]` → 申万一级行业名（如 `"银行I"`）—— **需要「行业分类」时用它，永远不是缺列理由**；不写进 `--cols`
    - `get_jq_data(symbol, data_type)` → 指数行情（`'price'`）/ 指数成分股（`'index_components'`），按需联网并缓存；同样不写进 `--cols`
    - 详细用法见各 type 对应的 `knowledge/*.md` 的「额外可用数据」段"""),
]

FILES = sorted({rel for rel, _, _ in PAIR_EDITS})


def apply(reverse: bool, check: bool) -> int:
    n_ok = n_skip = n_bad = 0
    for rel, removed, restored in PAIR_EDITS:
        p = ROOT / rel
        s = p.read_text(encoding="utf-8")
        src, dst = (restored, removed) if reverse else (removed, restored)

        if check:
            if dst in s and src not in s:
                n_ok += 1
            elif src in s and dst not in s:
                n_bad += 1
            else:
                n_skip += 1
            continue

        # 顺序很重要：先试源文本。某些条目的目标文本是源文本的子串
        # （如 cross_section 的 jq 块是「锚点+\n\n」的扩展），倒过来判会误跳过。
        if src in s:
            p.write_text(s.replace(src, dst, 1), encoding="utf-8")
            print(f"✓ [{rel}] 已更新")
            n_ok += 1
        elif dst in s:
            n_skip += 1                      # 已是目标状态 → 幂等跳过
        else:
            print(f"❌ [{rel}] 找不到待替换文本（文件已改动？）")
            n_bad += 1

    if check:
        for rel in FILES:
            s = (ROOT / rel).read_text(encoding="utf-8")
            mode = "恢复态" if "get_jq_data" in s else "撤下态"
            note = "有提示句" if "本地不可用" in s else "无提示句"
            print(f"  {mode} / {note} | {rel}")
        print(f"\n匹配 {n_ok} 处，待切换 {n_bad} 处，未知 {n_skip} 处")
    return 0 if n_bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="撤回：删 jq 说明、加回「本地不可用」提示")
    ap.add_argument("--check", action="store_true", help="只检查状态，不修改")
    args = ap.parse_args()

    if args.check:
        return apply(reverse=False, check=True)

    if args.revert:
        print("== 撤回模式：回到「不给 agent 提供 jq」 ==")
        rc = apply(reverse=True, check=False)
        print("\n完成。撤下后 agent 只会看到 INDUSTRY_DICT。")
        return rc

    print("== 恢复模式：把 get_jq_data 说明加回 ==")
    rc = apply(reverse=False, check=False)
    print("\n完成。**恢复前请先确认聚宽凭据可用**（见 jq_restore.md 第三节）。然后跑：")
    print("  python scripts/claude_factor_helper.py show-columns --type daily_single")
    print("  python scripts/claude_factor_helper.py show-columns --type minute")
    return rc


if __name__ == "__main__":
    sys.exit(main())
