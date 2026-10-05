# 论文 agent 作业指导（一篇研报：定义 → 编码 → 复现报告）

你按 `/factor` 的流程编排**这一篇**研报，自己**不读原文、不写因子代码**。

---

## 输入

主 agent 的 prompt 给了你：`报告名` / `原文 txt 路径` / `DATE`。

---

## Step 1：派定义 agent

```
你是定义 agent，只做「读原文 → 挖定义 → 写 extracted.json → 返回」。
Read `.claude/skills/factor/define_worker.md` 并严格照做，不做文件之外的事。

### 本篇参数
- 报告名: {报告名}
- 原文 txt 路径: {txt_path}
- DATE: {DATE}

完成后**只返回这一行 JSON，前后不加任何文字**：
{"report": "{报告名}", "total": 0, "error": null}
```

派发后**立刻结束回合**，等系统唤醒。被唤醒后核对定义是否落盘：

```bash
ls /mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/{报告名}.extracted.json
```

文件不存在 → 直接跳到 Step 4 并带 error。存在 → 继续 Step 2。

---

## Step 2：取因子清单

```bash
python -c "
import json
d=json.load(open('/mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/{报告名}.extracted.json'))
print(len(d['factors']))
for f in d['factors']: print(f['name'], f['type'], f['lookback'])
"
```

第一行是**因子总数**，后面每行是 `名称 类型 lookback`。

> 🚫 **不要用 Read 工具读 `extracted.json`** —— 只用上面这条命令取紧凑清单。

---

## Step 3：派发因子 worker（6 路并行，滚动补位）

对清单里**每一个**因子派一个独立 agent（**1 因子/agent**）。

**并发：维持 6 个在飞。** 先派 6 个，然后**立刻结束你的回合**；
系统每唤醒你一次（= 一个因子 worker 完成），就**从队列里补派一个**。

**收尾判据（每次唤醒后跑，只认这一条）：**

```bash
N=$(ls /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{报告名}/*/*.code.py \
        /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{报告名}/*/*.missing.json 2>/dev/null | wc -l)
echo "$N"; [ "$N" -ge {因子总数} ] && echo DONE || echo NOT-DONE
```

- `NOT-DONE` → 从队列补派，结束回合
- `DONE` → **立即执行 Step 4**，不要再多等一轮

> ⚠️ **通知会丢**：若某个因子 worker 完成的瞬间正好有另一个 worker 的通知在触发回合，
> 它的完成事件会被吞掉、不再补投。所以**每次唤醒都重跑上面的计数命令**，
> 用 `N` 判断还剩几个没完成，**不要靠「收到几条通知」记账**。
>
> ⚠️ **无进展熔断**：连续 2 次唤醒 `N` 都没有增加，或从 Step 3 开始已超过 15 分钟
> → 不再等，**直接执行 Step 4**（收尾只看磁盘产物，未完成的因子由报告如实记 missing）。
> **绝不重派**。

> ⚠️ **只数 `*.code.py` / `*.missing.json` 产物文件，不要用 `ls {报告名}/` 数目录** ——
> 目录是因子 worker 一启动就创建的，目录存在 ≠ 因子已完成。
>
> ⚠️ **派发后立刻结束回合**。不要 `sleep`、不要轮询、不要 `TaskOutput`（你没有这个工具）。

factor worker 的 prompt：

```
你是因子编码 agent，做两步：先判字段，字段齐全才写码。

### 第一步（必读）
Read `.claude/skills/factor/phase2_check.md`，按其说明判断本因子所需字段是否齐全。
- 若判缺列 → 按该文件「分支 B」写 `{name}.missing.json` 后直接返回，不要读 phase2_code.md
- 若字段齐全 → 继续第二步

### 第二步（仅字段齐全时）
Read `.claude/skills/factor/phase2_code.md`，按其说明写核心函数并跑 test-and-export + deploy-to-full。

### 本因子参数
- 因子名: {name}
- 类型: {type}（type_key: daily→daily_single, minute→minute, cross_section→cross_section, deep_learning→deep_learning）
- 函数名: daily→calc_factor_series, minute→calc_factors_one_day, cross_section→calc_factor_cross_section, deep_learning→train_model
- lookback: {lookback}
- 报告名: {报告名}
- 因子定义来源: /mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/{报告名}.extracted.json
  （取 `factors` 里 name=="{name}" 的条目。**不要读该文件里的其他因子，也不要读别的文件**）

完成后**只返回这一行 JSON，前后不加任何文字**：
{"name": "{name}", "success": true/false, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null 或 "不超过20字的失败原因"}
```

单个因子失败/超时**不阻塞其他因子**，继续跑剩下的。
**绝不重派**失败的因子 —— 重派等于同一个因子跑两遍，翻倍消耗。
返回值格式不对时，按失败记一笔即可，**不要追问、不要修复**。

---

## Step 4：生成本篇复现报告

```bash
python scripts/claude_factor_helper.py write-repro-report --date {DATE} --report "{报告名}"
```

输出：`因子产出/测试/{DATE}/{报告名}/复现报告.md`。

---

## Step 5：返回（只有一行）

```
{"report": "{报告名}", "total": 12, "ok": 3, "missing": 9, "error": null}
```

失败时：`{"report": "...", "total": 0, "ok": 0, "missing": 0, "error": "一句话原因"}`

**只返回这一行，前后不加任何文字。**
禁止附加：完成说明、因子清单、formulation、代码片段、判定理由。

---

## 硬约束

1. **绝不读原文**，绝不读 `knowledge/` 下的知识文件 —— 那些是定义 agent / 因子 worker 读的
2. **不要写因子代码、不要跑 test-and-export、不要 deploy-to-full**
3. **不要 `ls`/`grep` 探索目录**，不要读别的报告
4. **绝不重派**失败的因子
5. **维持 6 个在飞 + 滚动补位**（谁先完成立刻补谁，**不要按波次等齐**）
