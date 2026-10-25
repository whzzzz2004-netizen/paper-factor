# 批次调度 agent 作业指导

> **数据根目录**：先运行 `python3 scripts/claude_factor_helper.py data-root` 获取实际路径（本机 D 盘 `/mnt/d/paper-factor-data`；另一台电脑 G 盘自动识别）。下文所有 `/mnt/d/paper-factor-data` 均以该输出为准。

处理本批 ≤10 篇研报。**自己不读原文、不写代码、不派因子 worker。**

---

## ⛔ 回合纪律（优先于下面每一步）

**派发论文 agent 之后，本回合不得再调用任何工具 —— 连一次 `ls` 都不要。**
直接结束回合，等系统唤醒；磁盘核对一律留到**被唤醒后的下一回合**做。

> **为什么**：每次工具调用都要把「到目前为止的整段对话」重发一遍。派发后空转等结果，
> 每个回合白烧一份完整上下文（实测批次调度 agent 在同一回合内连打占其 token 的 81%）。
> 同一个回合内连打工具 = 纯浪费，一点也没让结果来得更快。

---

## 输入

主 agent 的 prompt 给了你：`批号` / `本批研报列表（报告名 → 原文 txt 路径）` / `DATE`。

---

## 派发论文 agent（2 篇在飞，滚动补位）

**先派 2 篇**，之后**谁先出复现报告就立刻补派下一篇**，始终保持 2 篇在飞。

**用 Agent 工具，`subagent_type: "factor-paper"`**，prompt 只有这几行：

```
你是单篇研报处理 agent，按 /factor 流程处理这一篇。
Read `.claude/skills/factor/paper_worker.md` 并严格照做，不做文件之外的事。

### 本篇参数
- 报告名: {报告名}
- 原文 txt 路径: {txt_path}
- DATE: {DATE}

完成后**只返回这一行 JSON，前后不加任何文字**：
{"report": "{报告名}", "total": 0, "ok": 0, "missing": 0, "error": null}
```

**派发循环**：

```
in_flight = 先派 2 篇（记下报告名）；未派队列 = 其余篇目
while 还有未派的篇 or in_flight 非空:
    **立即结束回合**，等系统唤醒（不要 sleep、不要轮询、不要 TaskOutput）
    被唤醒 → **先跑「唤醒后先磁盘核对」**，用磁盘结果更新 in_flight / 未派队列
    若还有未派的篇 → 补派到维持 2 篇在飞
    若「磁盘核对」显示全部完成 → 按「返回」格式返回 results，结束
```

> ⚠️ **in_flight 取自磁盘，不取自通知**：每次唤醒后用磁盘核对重算哪些篇还没完成，
> 而不是「收到一篇通知就 in_flight -= 1」。通知会丢，磁盘不会。
>
> ⚠️ **“谁先出报告就补谁”，不要“等 2 篇都完再派下一对”**。
>
> ⚠️ **派发后立刻结束回合，本回合不得再调用任何工具**（包括"再核对一次"）。
> 系统会在**每个**后台子 agent 完成时唤醒你一次，机制是通的，不需要你自己盯着。
> **不要 `sleep N` 轮询**；**不要用 `TaskOutput`**（子 agent 层没有这个工具）。
>
> ⚠️ 论文 agent 返回值格式不对（解析不出 `report`）时，**不要追问、不要重派**，
> 以磁盘为准记账。

---

## 唤醒后先磁盘核对（防通知丢失，**必须做**）

**子 agent 的完成通知会丢**：两个子 agent 相隔几秒先后结束时，前一个触发的回合
会把后一个的完成事件吞掉，之后不再补投。**靠通知记账 → 永远等一个不会来的通知**。

所以每次被唤醒，**先跑磁盘核对**（`REPORTS` 填本批全部篇目，未派的也填上）。
**每次唤醒只跑一次这条命令**，跑完就按下面两条分支行动，**不要连续重跑等它变**：

```bash
python3 - <<'EOF'
import os
from pathlib import Path
D = "{DATE}"
START = float(Path("/tmp/.factor_run_start").read_text().strip())
REPORTS = ["报告名1", "报告名2", ...]   # 本批全部篇目
ts = Path("/mnt/d/paper-factor-data/数据仓库/因子产出/测试") / D

def fresh(p):
    return p.exists() and p.stat().st_mtime >= START

done, undone = [], []
for r in REPORTS:
    d = ts / r
    if fresh(d / "复现报告.md"):
        sub = [x for x in os.listdir(d) if os.path.isdir(d / x)
               and (fresh(d / x / f"{x}.code.py") or fresh(d / x / f"{x}.missing.json"))]
        ok = sum(1 for s in sub if fresh(d / s / f"{s}.code.py"))
        ms = sum(1 for s in sub if fresh(d / s / f"{s}.missing.json"))
        done.append((r, len(sub), ok, ms))
    else:
        undone.append(r)
print("DONE", len(done), "/", len(REPORTS), "  undone:", undone)
for r, t, o, m in done: print(f"{r}\t{t}\t{o}\t{m}")
EOF
```

> ⚠️ **只认本次运行的产物**（`mtime ≥ /tmp/.factor_run_start`）。同日重跑时，
> 上一次留下的 `复现报告.md` 不算完成，旧的 `code.py` / `missing.json` 也不计入统计。

- **`undone` 为空 → 立即按「返回」格式返回 results**，不要再等通知、不要再补派。
  results 的 `total/ok/missing` 直接取上面输出的三列，`error` 置 null。
- **`undone` 非空 → 只对 `undone` 里的篇目补位**（它们在磁盘上没完成，才是真在飞）。
  已完成的篇目**立刻从 in_flight 剔除**——不管是否收到过它的通知。

> ⚠️ 「完成」只认磁盘：**收到通知但磁盘无产物 → 未完成**；**磁盘有产物但没收到通知 → 已完成**。
> 判据是 `测试/{DATE}/{报告名}/复现报告.md` 存在（论文 agent 最后一步才写）。

### 无进展熔断

派完最后一批后若连续 2 次唤醒都**没有任何未完成篇目转为完成**，或从本批开始
已超过 20 分钟：**不要再等**，直接按磁盘现状返回 results，仍未完成的篇目
`error` 填 `"timeout"`。**绝不重派**，收尾交给主 agent 的磁盘兜底。

---

## 返回（只有一行）

```
{"batch": "1", "results": [{"report": "...", "total": 12, "ok": 3, "missing": 9, "error": null}, ...]}
```

**必须有 `results` 数组，每篇一条**，即使某篇失败也要有一条（带 `error`）。

**只返回这一行，前后不加任何文字。**
禁止附加：完成说明、因子清单、formulation、代码片段、判定理由、统计汇总。

---

## 硬约束

1. **你不读研报、不写因子代码、不跑 test-and-export**
2. **不要读 `.claude/skills/factor/` 下的其他文件**（paper_worker / phase2_* / knowledge）
3. **不要 `ls` / `grep` 探索目录**
4. **维持 2 篇在飞 + 滚动补位**
5. **绝不重派**失败或超时的论文
6. **每次唤醒先磁盘核对**，`in_flight` 以磁盘为准，不靠通知记账
