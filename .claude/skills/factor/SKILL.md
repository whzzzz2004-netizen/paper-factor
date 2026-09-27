# /factor — 研报/文章因子端到端处理（两阶段架构）

## 用法

- `/factor` — 扫描 `/mnt/d/paper-factor-data/papers/inbox/`，处理所有未处理项
- `/factor /mnt/d/paper-factor-data/papers/inbox/某篇.pdf` — 处理单个 PDF

### 研报目录：inbox（处理区） / done（归档区）

| 目录 | 用途 |
|---|---|
| `papers/inbox/` | **投放 + 处理区**。放进去就会被处理 |
| `papers/done/{DATE}/` | **归档区**。处理完由 `/factor` 自动移过去 |

**inbox 里的文件一律重新处理**——不看历史、不做去重。投放前先确认 inbox 已清空。

> **单次建议 ≤5 篇**：每个 sub-agent 的启动回执 + 完成通知约 1,350 token 固定成本，
> 永久驻留主上下文；每因子约 1,900 token（含 Phase 1 摊薄）。
> 因子数比篇数更决定开销——一篇提 15 个因子就派 15 个 agent，即使多数会判缺字段。

**`/factor` 结束时自动归档**：
```bash
python scripts/claude_factor_helper.py archive-inbox --date {DATE}
```

**只做 `/factor`**（定义+测试+部署代码）。全量数值（parquet/IC/图表）需另跑 `python3 scripts/run_all.py {DATE}`。

## 核心规则

0. **必须用 `claude_factor_helper.py` 的命令**，不得自己写爬虫、装库、手动处理数据
1. **全自动决策**，不问用户
2. **强制提取**：每篇最多15个因子，无论是否明确写"因子"二字。择时策略的阈值→截面排序因子；行业轮动→行业偏离度；选股逻辑→多单维度因子
3. **子因子独立提取**，formulation 必须完整（从原始数据字段出发），禁止 `f(·)` 占位符
4. **禁止合成因子**
5. **唯一跳过场景**：所需数据完全不可用（如专有数据库API）。择时/选基/宏观/债券不跳过。**因子缺字段（某列不存在）不算"跳过"**：Phase 1 照常定义所有因子（不看列、不填列名）；Phase 2 看 `show-columns` 后判断，缺列则写 `{factor}.missing.json`，不生成代码、不部署全量
6. `source_excerpt` 从原文直接复制
7. **DATE 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`，不用研报的原始日期
8. **每次全量处理**：每次运行直接扫 inbox 里所有文件并全部处理
9. **所有分钟因子用 `minute` 模板，禁用 `minute_cs`**（太慢，且截面标准化对单因子无意义）。type 选项只有 `daily/minute/cross_section/deep_learning`。因子若需全市场数据（市场收益率等），先 `ls` 检查 `minute_by_date/` 有无预计算文件，没有就在代码里自己算（模块级预加载）
10. **minute 模板内 `df.index` 是 DatetimeIndex**（非 MultiIndex），直接用，不要 `get_level_values("datetime")`
11. **lookback 只取决于核心计算需要多少天**：论文末尾的"截面标准化 + 取std20/取波动率"是后处理，不作为依据；核心计算只用当天数据 → lookback=1

## 两阶段工作流

```
扫描 inbox → [Phase 1] 提取+定义因子 (每个paper一个sub-agent)
         → [Phase 2] 两阶段 (每个factor一个sub-agent)
                      ① 字段核对 phase2_check.md  → 缺列? 写 missing.json 即止
                      ② 字段齐全才读 phase2_code.md → 写码 + test-and-export + deploy-to-full
```

**核心原则：每个 sub-agent 的任务极其简单，没有犯错空间。**

---

### Step 0: 扫描 inbox + 数据预检

```bash
ls /mnt/d/paper-factor-data/papers/inbox/*.pdf /mnt/d/paper-factor-data/papers/inbox/*.md 2>/dev/null
```

> ⚠️ **必须同时包含 `.md`**：只扫 `*.pdf` 会静默漏掉 Markdown 研报。

**队列排序：** 含 "深度学习/GRU/TCN/LSTM/deep_learning" 的排末尾，其他优先。

**数据完整性预检（跳过会导致后续跑全量而非测试数据）：**
```bash
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/trade_dates.json')); print(f'日线测试: {len(sl)}只×{len(td)}天')"
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/trade_dates.json')); print(f'分钟测试: {len(sl)}只×{len(td)}天')"
```
日线/分钟都必须是 300只×300天，否则**先修复数据再继续**。

**全市场数据（如全市场分钟收益率）需预计算市场代理文件**，检查两个目录：
```bash
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/minute_by_date/market_minute_return.parquet
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/market_minute_return.parquet
```
缺失就跑（测试 + 全量都要）：
```bash
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试"
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量"
```

---

### Step 1: Phase 1 — 提取 + 定义因子

每个待处理项（paper/website）启动一个 sub-agent，**只做两件事：提取原文 → 定义因子**。同时最多 5 个（`run_in_background=true`）。

#### Phase 1 sub-agent prompt

```
subagent_type=general-purpose
prompt = """
你只做一件事：读原文 → 定义因子。不做编码，不做测试。

### 输入
- 类型: {type}
- 文本路径: {txt_path}  (仅paper，已预提取为 .txt)
- 网站索引: {index} (仅website)

### Step 1: 获取原文
- paper：用 Read 工具直接读取 {txt_path}（主进程已用 extract-pdf --outdir 预提取，无需自己跑命令）
- website：运行 `python scripts/claude_factor_helper.py extract-website --index {index}`，解析输出 JSON 的 `content` 字段作为正文

如果文本为空，跳过（返回 skipped）。

### Step 2: 定义因子
**只看原文定义因子，不看任何数据列、不填列名（列名交给 Phase 2 统一判断）。**

分析原文，定义所有因子。每条：
- name: 英文驼峰
- description: 中文
- formulation: 完整数学表达式
- type: daily/minute/cross_section/deep_learning
  **类型选择规则**：
  - 分钟频因子 → **一律 `minute`**（禁用 `minute_cs`）
  - 需要多个股票数据作为输入（如行业平均、市值分组）→ `cross_section`
  - 其他 per-stock 计算 → `daily` 或 `minute` 根据数据频率
- lookback: 天数 (1月≈20, 1季≈60, 6月≈120, 1年≈250)。
  **⚠️ 只考虑核心计算需要多少天**。论文末尾的"截面标准化+std20/取波动率"是后处理，不作为lookback依据。
  如果核心计算只需要当天数据 → lookback=1。
- source_excerpt: 原文复制

最多15个因子。formulation 必须完整。**不填 cols（不要臆造列名）**。**不要静默丢弃字段可能缺失的因子**：所有因子照常定义并保留（含字段可能缺失的），字段是否存在、用什么列名，全部交给 Phase 2 的 agent 看 show-columns 输出后自行判断（确实缺失会在测试因子目录记录 `{name}.missing.json`）。

### Step 3: 保存（⚠️ 严格照抄下面的 schema，不要去看 helper 源码）

用 Write 工具写 JSON 到 `/tmp/factor_{DATE}_{序号}.json`，**顶层和每条因子的字段必须齐全且同名**：

```json
{
  "report_name": "标题",
  "date": "{DATE}",
  "factors": [
    {
      "name": "VolumeRatioConfirm",
      "description": "中文描述：这个因子算什么、怎么用",
      "formulation": "完整数学表达式，从原始数据字段出发，禁止 f(·) 占位符",
      "type": "daily",
      "lookback": 20,
      "source_excerpt": "从原文直接复制的原句"
    }
  ]
}
```

**字段硬要求**（写错会导致 Phase 2 取不到定义）：
- `factors` 是数组，每条**必须**含这 6 个键：`name` / `description` / `formulation` / `type` / `lookback` / `source_excerpt`
- `name` 用英文驼峰（如 `VolumeRatioConfirm`），全篇唯一
- `type` 只能是 `daily` / `minute` / `cross_section` / `deep_learning` 四者之一
- `lookback` 是整数（天数）
- **不要**加 `cols` 字段（Phase 1 不提供列名）

然后**用 Bash 把这个文件喂给 helper**（不要用 heredoc 手写 JSON，避免引号转义出错）：

```bash
python scripts/claude_factor_helper.py save-extracted --name "标题" --date {DATE} < /tmp/factor_{DATE}_{序号}.json
# 期望输出：OK: /mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/标题.extracted.json
```

### Step 4: 返回
{{{{
  "report_name": "标题",
  "date": "{DATE}",
  "factors": [{{"name": "F1", "type": "daily", "lookback": 20, "cols": []}}, ...],
  "skipped": false
}}}}

### ⚠️ 返回值纪律（省 token 关键）
**只返回上面那段 JSON，前后不加任何文字。**
禁止附加：完成说明、实现要点、判定理由、注意事项、原文摘录、验证过程。
需要留档的写进因子 JSON 的 formulation/description 字段，不要放进返回值。
（同理：不写代码、不跑测试、不调 FactorFBWorkspace、不加载 parquet。）
"""
```

#### 派发逻辑

```
# 1. 先 resolve 所有文件路径（避免文件名含特殊字符导致 shell 解析错误）
for each paper:
    run: python3 -c "import glob; print(glob.glob(paper_path)[0] if glob.glob(paper_path) else '')"
    得到真实路径 → 存入 task.path

# 2. 主进程一次性预提取所有 PDF → /tmp/factor_pdf/
run: rm -rf /tmp/factor_pdf
run: python scripts/claude_factor_helper.py extract-pdf {全部真实paper路径，空格分隔} --outdir /tmp/factor_pdf
     → 输出 {"<源路径>": "/tmp/factor_pdf/00_xxx.txt", ...} 映射
     → 每个 paper 的 txt 路径存入 task.txt_path；无映射的 paper 跳过

tasks = flatten(papers + websites, DL排最后)
```

**派发循环（Phase 1/Phase 2 通用，只换队列和 worker 函数）：**

```
next_idx = 0; active = {}; results = []

for _ in range(min(5, len(queue))):
    agent_id = dispatch(queue[next_idx]); active[agent_id] = queue[next_idx]; next_idx += 1

# ⚠️ 必须用 block=true 阻塞等待。不要 block=false + sleep(5) 轮询空转——
#    那会让每个因子空转几十轮主 agent 调用，纯烧 token，总耗时不变。
while active:
    TaskOutput(task_id=next(iter(active)), block=true, timeout=600000)  # 等任意一个完成
    for agent_id in list(active.keys()):
        out = TaskOutput(task_id=agent_id, block=false, timeout=0)
        if out.status == "completed":
            results.append({agent_id: out.result}); del active[agent_id]
            if next_idx < len(queue):
                new_id = dispatch(queue[next_idx]); active[new_id] = queue[next_idx]; next_idx += 1
    # 仍有人跑 → 回 while 顶部继续阻塞，不 sleep
```

- Phase 1：`queue = tasks`，`dispatch = dispatch_phase1_worker`
- Phase 2：`queue = all_factors`，`dispatch = dispatch_phase2_worker`

---

### Step 2: Phase 2 — 先判字段，再编码 + 测试

**⚠️ 收集策略：不要只依赖 Phase 1 agent 的返回值**（agent 可能超时/失败）。从
`/mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/` 读取所有 `.extracted.json`，合并去重。

每个因子启动一个 sub-agent，用上面的派发循环，最多同时 5 个。

> **type→type_key 映射**（填 `--type {type_key}` 用）：daily→daily_single, minute→minute,
> cross_section→cross_section, deep_learning→deep_learning
> **`--cols` 格式**：空格或逗号分隔均可（helper 自动 split）

#### Phase 2 sub-agent prompt

子 agent 先读 `phase2_check.md`（字段核对），只在不缺列时才读 `phase2_code.md`（写码规范）。
缺字段的因子省掉第二步。**主 agent 的 prompt 只有下面这几行**：

```
subagent_type=general-purpose
prompt = """
你是因子编码 agent，做两步：先判字段，字段齐全才写码。

### 第一步（必读）
Read `.claude/skills/factor/phase2_check.md`，按其说明判断本因子所需字段是否齐全。
- 若**判缺列** → 按该文件「分支 B」写 `{name}.missing.json` 后直接返回，**不要**读 phase2_code.md
- 若**字段齐全** → 继续第二步

### 第二步（仅字段齐全时）
Read `.claude/skills/factor/phase2_code.md`，按其说明写核心函数并跑 test-and-export + deploy-to-full。

### 本因子参数
- 因子名: {name}
- 类型: {type}（type_key={type_key}）
- 函数名: {func_name}
- lookback: {lookback}
- 报告名: {report_name}
- 因子定义来源: /mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/{report_name}.extracted.json
  （取 `factors` 里 name=="{name}" 的条目，**不要读该文件里的其他因子，也不要读别的文件**）

完成后**只返回这一行 JSON，前后不加任何文字**（不写完成说明/实现要点/判定理由/注意事项）：
{{"name": "{name}", "success": true/false, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null 或 "不超过20字的失败原因"}}
"""
```

> ⚠️ **不要内联 `formulation`/`description`/`source_excerpt`**（本批 75 因子全文 ≈18.4k token，
> 会永久驻留主上下文）。子 agent 自己从 extracted.json 取，零边际成本。
> **不要贴 `phase2_check.md` / `phase2_code.md` 正文**，子 agent 自己 Read 一次即可。

**收尾（Phase 2 全部完成后）：**

```bash
# 按报告生成复现报告（遍历当天 extracted_reports/{DATE}/ 所有报告名）
python scripts/claude_factor_helper.py write-repro-report --date {DATE} --report "{report_name}"
# → 因子产出/测试/{DATE}/{report_name}/复现报告.md

# 归档 inbox
python scripts/claude_factor_helper.py archive-inbox --date {DATE}
```

---

### 并行全量计算（自动重扫）

想在 `/factor` 生成因子的同时让另一个 dialog 自动跑全量：`/all 2026-08-16`。
`/all` 处理完所有待处理因子后会自动重扫目录，队列清空后自动退出。两边互不打断。

---

## 编码约束

**函数签名、向量化要求、字段纪律、各类型入参与额外可用数据，全部由 `phase2_code.md` +
对应 `knowledge/{type}.md` 提供，此处不重复。** 子 agent 按 type 自行读取：

| type | type_key | 知识文件 |
|---|---|---|
| daily | daily_single | `knowledge/daily.md` |
| minute | minute | `knowledge/minute.md` |
| cross_section | cross_section | `knowledge/cross_section.md` |
| deep_learning | deep_learning | `knowledge/deep_learning.md` |

**`show-columns --type` 只有两个合法取值**：`minute` 和 `daily_single`（daily / cross_section /
deep_learning 三者都用 `daily_single`）。**没有** `--type cross_section` / `--type deep_learning`。
