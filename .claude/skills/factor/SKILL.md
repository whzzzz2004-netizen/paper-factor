# /factor — 研报/文章因子端到端处理（两阶段架构）

## 用法

- `/factor` — 扫描 `/mnt/d/paper-factor-data/papers/inbox/`，处理所有未处理项
- `/factor /mnt/d/paper-factor-data/papers/inbox/某篇.pdf` — 处理单个 PDF

### 研报目录：inbox（处理区） / done（归档区）

| 目录 | 用途 |
|---|---|
| `papers/inbox/` | **投放 + 处理区**。放进去就会被处理 |
| `papers/done/{DATE}/` | **归档区**。处理完由 `/factor` 自动移过去，避免 inbox 越堆越多 |

**规则：inbox 里的文件一律重新处理**——不看历史、不做去重。
所以**每次投放前请先确认 inbox 已清空**（处理完会自动清），避免把已完成的重跑一遍。

> 想控制单次规模时：**只往 inbox 放你这次想跑的几篇**（建议 ≤15 篇）。
> 理由：`/factor` 的 Phase 1/Phase 2 派发都在同一个主会话里，每个因子在主上下文留下约 180 token 且永久驻留。
> 实测外推：10 篇 ≈ 27k（安全）、30 篇 ≈ 81k（勉强）、50 篇 ≈ 135k（会爆）。

**`/factor` 结束时自动归档**（无需手动）：
```bash
python scripts/claude_factor_helper.py archive-inbox --date {DATE}
```
把 inbox 里本次处理的研报移到 `papers/done/{DATE}/`。也可单独跑（不带 `--names` 就是归档 inbox 全部）。

**只做 `/factor`**（定义+测试+部署代码）。全量数值（parquet/IC/图表）需另跑 `python3 scripts/run_all.py {DATE}`。

## 核心规则

0. **必须用 `claude_factor_helper.py` 的命令**，不得自己写爬虫、装库、手动处理数据
1. **全自动决策**，不问用户
2. **强制提取**：每篇最多15个因子，无论是否明确写"因子"二字。择时策略的阈值→截面排序因子；行业轮动→行业偏离度；选股逻辑→多单维度因子
3. **子因子独立提取**，formulation 必须完整（从原始数据字段出发），禁止 `f(·)` 占位符
4. **禁止合成因子**
5. **唯一跳过场景**：所需数据完全不可用（如专有数据库API）。择时/选基/宏观/债券不跳过。**因子缺字段（某列不存在）不算"跳过"**：Phase 1 照常定义所有因子（不看列、不填列名）；Phase 2 才看 `show-columns` 全部列名、结合因子定义自行判断。若判断因子因缺列无法实现 → 自己在测试因子目录写 `{factor}.missing.json` 记录缺的列，不生成代码、不部署全量。字段齐全的因子 → Phase 2 从 show-columns 挑出真实列名写代码。
6. `source_excerpt` 从原文直接复制
7. **DATE 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`。所有 save-extracted / test-and-export / deploy-to-full 的 `--date` 都传当天日期，不用研报的原始日期。
8. **每次全量处理**：每次运行直接扫 `/mnt/d/paper-factor-data/papers/inbox/` 里的所有文件并全部处理。
9. **所有分钟因子必须用 `minute` 类型模板，严禁用 `minute_cs`**：
   - `minute_cs` 太慢（测试 6~8 分钟/个，全量可能数小时），且截面标准化对单因子无意义
   - 如果因子需要全市场数据（市场收益率、总成交量等），先用 `ls` 检查 `minute_by_date/` 下是否有预计算文件，没有就让 LLM **在代码中自己计算**（模块级预加载，一次性计算）
   - 如果因子逻辑本质依赖全市场截面（如排名、市值分组等），**用简单近似代替**（如用个股自身过去 N 天分位数代替截面排名）
10. **minute 模板内 `df.index` 是 DatetimeIndex（非 MultiIndex）**：`calc_factors_one_day(df, stock)` 收到的 `df.index` 是模板转换后的 `DatetimeIndex`，不要调用 `get_level_values("datetime")`。直接用 `df.index` 即可。
11. **lookback 只取决于核心计算需要多少天**：
   - 论文末尾的"截面标准化 + 取std20/取波动率"是高频因子低频化的**后处理步骤**，不作为日频因子的 lookback 依据
   - 如果某步需要跨日平滑/差分/滚动（如过去20天弹性系数移动平均）→ 设对应 lookback
   - 如果核心计算只用到当天数据 → lookback=1
12. **禁用 `minute_cs` 类型**：Phase 1 定义因子时 type 选项只有 `daily/minute/cross_section/deep_learning`，不再有 `minute_cs`。Phase 2 编码统一走 minute 模板。

## 两阶段工作流

```
扫描 inbox → [Phase 1] 提取+定义因子 (每个paper一个sub-agent, 只做extract+define)
         → [Phase 2] 编码+测试+部署 (每个factor一个sub-agent, 写核心函数+跑test-and-export+deploy-to-full)
```

**核心原则：每个 sub-agent 的任务极其简单，没有犯错空间。**

---

### Step 0: 扫描 inbox + 数据预检

直接扫 `/mnt/d/paper-factor-data/papers/inbox/` 目录列出所有**研报文件（PDF 和 Markdown）**。

```bash
ls /mnt/d/paper-factor-data/papers/inbox/*.pdf /mnt/d/paper-factor-data/papers/inbox/*.md 2>/dev/null
```

> ⚠️ **必须同时包含 `.md`**：`extract-pdf` 支持 PDF 和 .md；只扫 `*.pdf` 会静默漏掉 Markdown 研报（曾经漏过 `EpsRevision因子研究.md`）。

**队列排序规则：** 含 "深度学习/GRU/TCN/LSTM/deep_learning" 的排末尾，其他优先。

**`{DATE}` 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`。所有子命令的 `--date` 参数都传这个值，不传研报原始日期。

**数据完整性预检（跳过会导致后续跑全量而非测试数据）：**
```bash
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/trade_dates.json')); print(f'日线测试: {len(sl)}只×{len(td)}天')"
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/trade_dates.json')); print(f'分钟测试: {len(sl)}只×{len(td)}天')"
```
如果日线不是 300只×300天，或分钟不是 300只×300天，说明测试数据有问题，**先修复数据再继续**。

**如果需要全市场数据（如全市场分钟收益率），预计算市场代理文件：**
检查两个目录：
```bash
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/minute_by_date/market_minute_return.parquet
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/market_minute_return.parquet
```
如果有缺失，跑预计算脚本（测试 + 全量都要跑）：
```bash
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试"
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量"
```

---

### Step 1: Phase 1 — 提取 + 定义因子

> **`{DATE}` 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`。所有 `--date` 参数都传这个值，不传研报原始日期。

对每个待处理项（paper/website），启动一个 sub-agent。**每个 sub-agent 只做两件事：提取原文 → 定义因子。不做编码测试。**

sub-agent 同时启动最多 **5 个**（`run_in_background=true`），完成后主 Claude 立即派发下一个。

#### Phase 1 sub-agent prompt（极简 ~30 行）

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

### Step 3: 保存
python scripts/claude_factor_helper.py save-extracted --name "标题" --date {DATE} < 因子JSON

### Step 4: 返回
{{{{
  "report_name": "标题",
  "date": "{DATE}",
  "factors": [{{"name": "F1", "type": "daily", "lookback": 20, "cols": []}}, ...],
  "skipped": false
}}}}

### 禁止
- ❌ 不写代码，不跑测试
- ❌ 不调 FactorFBWorkspace
- ❌ 不加载 parquet

### ⚠️ 返回值纪律（省 token 关键）
**只返回上面那段 JSON，前后不加任何文字。**
禁止附加：完成说明、实现要点、判定理由、注意事项、原文摘录、验证过程。
需要留档的写进因子 JSON 的 formulation/description 字段，不要放进返回值。
"""
```

#### 派发逻辑
```
# 1. 先 resolve 所有文件路径（避免文件名含特殊字符导致 shell 解析错误）
for each paper:
    run: python3 -c "import glob; print(glob.glob(paper_path)[0] if glob.glob(paper_path) else '')"
    得到真实路径 → 存入 task.path

# 2. 主进程一次性预提取所有 PDF → /tmp/factor_pdf/（一次性，无子代理内联 subprocess）
run: rm -rf /tmp/factor_pdf
run: python scripts/claude_factor_helper.py extract-pdf {全部真实paper路径，空格分隔} --outdir /tmp/factor_pdf
     → 输出 {"<源路径>": "/tmp/factor_pdf/00_xxx.txt", ...} 映射
     → 把每个 paper 的 txt 路径存入 task.txt_path；无映射的 paper 跳过
     （paper 路径含特殊字符时，直接传目录 /mnt/d/paper-factor-data/papers/inbox 代替，映射仍按文件输出）

tasks = flatten(papers + websites, DL排最后)
next_idx = 0
active = {}  # {agent_id: task_info}
results = []

# 启动前 min(5, len(tasks)) 个
for _ in range(min(5, len(tasks))):
    agent_id = dispatch_phase1_worker(tasks[next_idx])
    active[agent_id] = tasks[next_idx]
    next_idx += 1

# 每当一个 worker 返回 → 立即派发下一个
# ⚠️ 用 block=true 阻塞等待，不要用 block=false + sleep(5) 轮询空转
#    （后者每个因子要空转几十轮主 agent 调用，纯烧 token；总耗时不变）
while active:
    # 阻塞等待「任意一个」agent 完成；单次上限 10 分钟，超时后本轮重扫再继续等
    output = TaskOutput(task_id=next(iter(active)), block=true, timeout=600000)
    # 该 agent 完成（或超时）后，用 block=false 一次性收走所有已完成的
    for agent_id in list(active.keys()):
        out = TaskOutput(task_id=agent_id, block=false, timeout=0)
        if out.status == "completed":
            results.append({agent_id: out.result})
            del active[agent_id]
            if next_idx < len(tasks):
                new_id = dispatch_phase1_worker(tasks[next_idx])
                active[new_id] = tasks[next_idx]
                next_idx += 1
    # 仍有人在跑 → 回到 while 顶部继续阻塞等待，不 sleep 空转

# 全部完成 → 进入 Phase 2
```

---

### Step 2: Phase 2 — 编码 + 测试

收集 Phase 1 所有成功定义的因子，**为每个因子启动一个 sub-agent**。每个 sub-agent 只做：**写核心函数 → 跑 test-and-export**。

**⚠️ 收集策略：不要只依赖 Phase 1 agent 的返回值。agent 可能超时/失败。此外从 `/mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/` 目录读取所有已保存的 `.extracted.json` 文件，合并去重，确保不漏因子。**

最多同时启动 **5 个** sub-agent。主 Claude 控制派发。

> **type→type_key 映射**（Phase 2 prompt 里填 `--type {type_key}` 用）：daily→daily_single, minute→minute, cross_section→cross_section, deep_learning→deep_learning。
> **`--cols` 格式**：空格或逗号分隔均可（helper 自动 split），如 `--cols "close factor"` 或 `--cols "close,factor"`。

#### Phase 2 sub-agent prompt（主 agent 只传参数，不重复输出全文）

**主 agent 的 prompt 只有下面这几行**（把 190 行作业指导交给子 agent 自己读）：

```
subagent_type=general-purpose
prompt = """
你是因子编码 agent。第一步：Read `.claude/skills/factor/phase2_prompt.md`（完整作业指导，含 Step 0 列核对、写码规范、test-and-export 命令、失败重试规则）。然后严格按该文件执行，不要做文件之外的事。

### 本因子参数
- 因子名: {name}
- 类型: {type}（type_key={type_key}）
- 函数名: {func_name}
- lookback: {lookback}
- 报告名: {report_name}
- formulation: {formulation}
- description: {description}
- source_excerpt: {source_excerpt}

完成后**只返回这一行 JSON，前后不加任何文字**（不写完成说明/实现要点/判定理由/注意事项）：
{{"name": "{name}", "success": true/false, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null 或 "不超过20字的失败原因"}}
"""
```

> ⚠️ **不要再把 `phase2_prompt.md` 的正文贴进 prompt**。以前主 agent 每个因子重复输出 ~2.5k tokens 的作业指导（13 个因子≈32k tokens 纯浪费），现在由子 agent 自己 Read 一次即可。
>
> ⚠️ **返回值纪律**：agent 的返回值会**永久留在主 agent 上下文**（每轮重发）。附加说明会累积成几十 k token。
> 因此主 agent 派发时必须带上"只返回一行 JSON，不加任何文字"的约束（见上方 prompt）。

#### 派发逻辑
```
all_factors = flatten(所有Phase1结果的factors)
next_idx = 0
active = {}  # {agent_id: factor_info}
results = []

for _ in range(min(5, len(all_factors))):
    agent_id = dispatch_phase2_worker(all_factors[next_idx]); next_idx += 1
    active[agent_id] = all_factors[next_idx-1]

# ⚠️ 同 Phase 1：用 block=true 阻塞等待，不要 block=false + sleep(5) 轮询空转
while active:
    output = TaskOutput(task_id=next(iter(active)), block=true, timeout=600000)
    for agent_id in list(active.keys()):
        out = TaskOutput(task_id=agent_id, block=false, timeout=0)
        if out.status == "completed":
            results.append({agent_id: out.result})
            del active[agent_id]
            if next_idx < len(all_factors):
                new_id = dispatch_phase2_worker(all_factors[next_idx])
                active[new_id] = all_factors[next_idx]; next_idx += 1
    # 仍有人跑 → 回 while 顶部继续阻塞，不 sleep

# 全部完成

# ── 收尾：按报告生成复现报告（哪些因子成功复现 / 哪些未复现及原因）──
# 遍历当天所有 extracted_reports/{DATE}/*.extracted.json 的报告名，逐个生成
for each report_name in 当天所有报告的集合:
    run: python scripts/claude_factor_helper.py write-repro-report --date {DATE} --report "{report_name}"
# 每份报告输出 因子产出/测试/{DATE}/{report_name}/复现报告.md

# ── 最后：归档 inbox，避免越堆越多 ──
run: python scripts/claude_factor_helper.py archive-inbox --date {DATE}
# 把 papers/inbox/ 本次处理的研报移到 papers/done/{DATE}/
```

---

### 并行全量计算（自动重扫）

如果想在 `/factor` 生成因子的同时让另一个 dialog 自动跑全量：

```bash
/all 2026-08-16
```

`/all` 处理完所有待处理因子后会自动重扫目录，如果 `/factor` 在此期间 deploy 了新因子，继续处理；队列清空后自动退出。两边互不打断。

---

## 模板类型 → 函数名对照
- daily → `def calc_factor_series(df, stock) -> pd.Series`（向量化，优先）。可选 `def calc_factor_single_stock(df, trade_date, stock)`（逐日 fallback）
- minute → `def calc_factors_one_day(df, stock):` 或 `def calc_factor_series(df, stock):`（向量化版，每只股票只调1次）
- cross_section → `def calc_factor_cross_section(all_data, trade_date):`
- deep_learning → `def train_model(all_data, trade_date):` + `def predict_batch(model, data_dict, trade_date):`（LOOKBACK_DAYS 只决定预测窗口大小；训练用截止日全部历史（walk-forward），模型内部自行决定用多少历史）

## type→type_key 映射
- daily → daily_single
- minute → minute
- cross_section → cross_section
- deep_learning → deep_learning

## 编码硬约束
**主进程按 type 拼 prompt 时：仅把适用该类型的条目拼进去，其余丢弃**（通用条目所有类型都适用）：
- 通用（所有类型）：2, 3, 6, 7, 8, 9, 11, 12, 13, 15, 17
- daily 独有：1（T日=df.iloc[-1]）, 4（日线收益率用 close.pct_change()）, 5（df.index.date 不放循环内）
- minute 独有：10（禁止分钟级 for 循环）
- cross_section/deep_learning：无独有条目（用通用即可）

1. T日 = df.iloc[-1]
2. 返回 `{"因子名": np.nan}`，不返回 None
3. 禁止月末判断
4. 日线收益率用 `close.pct_change()`
5. `df.index.date` 不放循环内
6. 布尔 shift() 后 fillna(False)
7. 禁止 `len(df) < X` 做上市天数筛选
8. 禁止未来数据
9. np.inf/-np.inf → np.nan
10. 禁止分钟级 for 循环（用向量化操作）
11. 禁止 `transform('count')` → 用 `transform('size')`
12. 禁止 `rolling.apply(lambda)`
13. 禁止合成因子
14. **所有分钟因子用 minute 模板**（禁用 minute_cs：太慢，且截面标准化无意义）
15. **lookback 只含核心计算天数**，不含论文末尾的截面标准化/std20/取波动率等后处理
16. **禁用截面操作（排名/标准化/行业中性化）**：因子只输出个股原始值。如果截面是核心逻辑，额外写后处理函数对产出 .parquet 做截面变换
17. **`df.index.date` 返回 ndarray**，没有 `.isin()` 方法。用 `np.isin(date_arr, list)` 替代
18. **字段只认 `show-columns` 当次输出**：输出里有则用、无则判缺列。数据会持续补字段，不要凭记忆假定有哪些列，也不要去数据仓库/源码/memory 翻找