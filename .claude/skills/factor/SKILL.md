# /factor — 研报/文章因子端到端处理（两阶段架构）

## 用法

- `/factor` — 扫描 `papers/inbox/` 和 `papers/ideas/ideas.json`，处理所有未处理项
- `/factor papers/inbox/某篇.pdf` — 处理单个 PDF
- `/factor 一段因子描述` — 处理纯文本

## 核心规则

0. **必须用 `claude_factor_helper.py` 的命令**，不得自己写爬虫、装库、手动处理数据
1. **全自动决策**，不问用户
2. **强制提取**：每篇最多15个因子，无论是否明确写"因子"二字。择时策略的阈值→截面排序因子；行业轮动→行业偏离度；选股逻辑→多单维度因子
3. **子因子独立提取**，formulation 必须完整（从原始数据字段出发），禁止 `f(·)` 占位符
4. **禁止合成因子**
5. **唯一跳过场景**：所需数据完全不可用（如专有数据库API）。择时/选基/宏观/债券不跳过。**因子缺字段（某列不存在）不算"跳过"**：Phase 1 照常定义所有因子（不看列、不填列名）；Phase 2 才看 `show-columns` 全部列名、结合因子定义自行判断。若判断因子因缺列无法实现 → 自己在测试因子目录写 `{factor}.missing.json` 记录缺的列，不生成代码、不部署全量。字段齐全的因子 → Phase 2 从 show-columns 挑出真实列名写代码。
6. `source_excerpt` 从原文直接复制
7. **DATE 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`。所有 save-extracted / test-and-export / deploy-to-full 的 `--date` 都传当天日期，不用研报的原始日期。
8. **不标记完成**：不跑 `mark-done`，不跟踪处理状态。每次运行直接扫 `papers/inbox/` 里的所有文件，全量处理。
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

直接扫 `papers/inbox/` 目录列出所有 PDF 文件，以及 `papers/ideas/ideas.json` 中的所有 ideas。不跑 `scan-pending`，不检查完成状态。

```bash
ls papers/inbox/*.pdf 2>/dev/null
python3 -c "import json; d=json.load(open('papers/ideas/ideas.json')); print([x.get('text','') or x.get('description','') for x in d])"
```

**队列排序规则：** 含 "深度学习/GRU/TCN/LSTM/deep_learning" 的排末尾，其他优先。

**`{DATE}` 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`。所有子命令的 `--date` 参数都传这个值，不传研报原始日期。

**数据完整性预检（跳过会导致后续跑全量而非测试数据）：**
```bash
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/trade_dates.json')); print(f'日线测试: {len(sl)}只×{len(td)}天')"
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/trade_dates.json')); print(f'分钟测试: {len(sl)}只×{len(td)}天')"
```
如果日线不是 300只×300天，或分钟不是 300只×300天，说明测试数据有问题，**先修复数据再继续**。

**主进程预跑列清单（一次性，所有 Phase 2 agent 共享，省去每个 agent 各跑一次 show-columns）：**
```bash
python scripts/claude_factor_helper.py show-columns --type daily_single   # → 保存为 {DAILY_COLS_TEXT}
python scripts/claude_factor_helper.py show-columns --type minute         # → 保存为 {MINUTE_COLS_TEXT}
```
把两份输出直接内联进每个 Phase 2 sub-agent 的 prompt（见下），agent **不再自己跑 show-columns**。

**如果需要全市场数据（如全市场分钟收益率），预计算市场代理文件：**
检查两个目录：
```bash
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/minute_by_date/market_minute_return.parquet
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/market_minute_return.parquet
```
如果有缺失，跑预计算脚本（测试 + 全量都要跑）：
```bash
python scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试"
python scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量"
```

---

### Step 1: Phase 1 — 提取 + 定义因子

> **`{DATE}` 永远是当天日期**：`datetime.now().strftime("%Y-%m-%d")`。所有 `--date` 参数都传这个值，不传研报原始日期。

对每个待处理项（paper/website/idea），启动一个 sub-agent。**每个 sub-agent 只做两件事：提取原文 → 定义因子。不做编码测试。**

sub-agent 同时启动最多 **5 个**（`run_in_background=true`），完成后主 Claude 立即派发下一个。

#### Phase 1 sub-agent prompt（极简 ~30 行）

```
subagent_type=general-purpose
prompt = """
你只做一件事：读原文 → 定义因子。不做编码，不做测试。

### 输入
- 类型: {type}
- 文本路径: {txt_path}  (仅paper，已预提取为 .txt)
- 文本: {text}  (仅idea)
- 网站索引: {index} (仅website)

### Step 1: 获取原文
- paper：用 Read 工具直接读取 {txt_path}（主进程已用 extract-pdf --outdir 预提取，无需自己跑命令）
- website：运行 `python scripts/claude_factor_helper.py extract-website --index {index}`，解析输出 JSON 的 `content` 字段作为正文
- idea：直接用 {text}

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
     （paper 路径含特殊字符时，直接传目录 papers/inbox 代替，映射仍按文件输出）

tasks = flatten(papers + websites + ideas, DL排最后)
next_idx = 0
active = {}  # {agent_id: task_info}
results = []

# 启动前 min(5, len(tasks)) 个
for _ in range(min(5, len(tasks))):
    agent_id = dispatch_phase1_worker(tasks[next_idx])
    active[agent_id] = tasks[next_idx]
    next_idx += 1

# 每当一个 worker 返回 → 立即派发下一个
while active:
    # 对每个 active agent 用 TaskOutput block=false 检查完成状态
    for agent_id in list(active.keys()):
        output = TaskOutput(task_id=agent_id, block=false, timeout=0)
        if output.status == "completed":
            results.append({agent_id: output.result})
            del active[agent_id]
            if next_idx < len(tasks):
                new_id = dispatch_phase1_worker(tasks[next_idx])
                active[new_id] = tasks[next_idx]
                next_idx += 1
    if active:
        import time; time.sleep(5)  # 等 5 秒再检查

# 全部完成 → 进入 Phase 2
```

---

### Step 2: Phase 2 — 编码 + 测试

收集 Phase 1 所有成功定义的因子，**为每个因子启动一个 sub-agent**。每个 sub-agent 只做：**写核心函数 → 跑 test-and-export**。

**⚠️ 收集策略：不要只依赖 Phase 1 agent 的返回值。agent 可能超时/失败。此外从 `/mnt/d/paper-factor-data/数据仓库/因子产出/extracted_reports/{DATE}/` 目录读取所有已保存的 `.extracted.json` 文件，合并去重，确保不漏因子。**

最多同时启动 **5 个** sub-agent。主 Claude 控制派发。

> **type→type_key 映射**（Phase 2 prompt 里填 `--type {type_key}` 用）：daily→daily_single, minute→minute, cross_section→cross_section, deep_learning→deep_learning。
> **`--cols` 格式**：空格或逗号分隔均可（helper 自动 split），如 `--cols "close factor"` 或 `--cols "close,factor"`。

#### Phase 2 sub-agent prompt（极简 ~30 行）

```
subagent_type=general-purpose
prompt = """
你只做一件事：写一个核心函数并跑 test-and-export。不做其他任何事。

### 因子定义
- 因子名: {name}
- 类型: {type}（minute/daily/cross_section/deep_learning）
- 函数名: {func_name}  (见下方对照表)
- lookback: {lookback}  **（⚠️ 只含核心计算天数，不含论文末尾的截面标准化/std20后处理）**
- 报告名: {report_name}
- formulation: {formulation}
- description: {description}
- source_excerpt: {source_excerpt}
- 需要的字段: 由你从 formulation/description 推导（Phase 1 不提供列名，见下方 Step 0）

### Step 0：看全部列 → 推导字段 → 判断缺列 / 挑出真实列名（最重要的一步）
**列清单已在下方直接给出（主进程预跑 show-columns 的内联结果），无需自己跑命令：**
- **minute** 类型 → 用下方 `{MINUTE_COLS_TEXT}`
- **daily / cross_section / deep_learning** 类型 → 用下方 `{DAILY_COLS_TEXT}`

**从因子 definition/formulation/description 推导它需要的字段（语义，如"当日分钟收益率"→return、"市盈率"→pe_ttm），再对照上面给出的完整列名逐一核对（这是唯一的字段核对方式）：**

- **因子需要的字段都能在输出里找到 → 字段齐全**：
  - **从 show-columns 输出里挑出每个字段对应的真实列名**（如"市盈率"对应 `pe_ttm` 而非臆造 `pe`），这些真实列名就是本因子的 `{cols}`
  - 继续 Step 1 写代码，`--cols` 用这些真实列名
- **因子需要某字段但输出里没有对应列 → 先尝试"用现有列组合推导"，实在没有才算缺列**：
  - **字段可推导知识点**（常见派生字段）：`换手率 ≈ volume / (float_shares × 10000)`（volume 单位=股，float_shares 单位=万股）；涨跌幅可用 `close.pct_change()`（或用 `pct_chg`）；"昨日收盘"可用 `close.shift(1)`；市值相关已直接有 market_cap/circulating_market_cap 列。
  - 如果所需字段能用清单里现有列组合算出来 → **不判缺列**，用这些列实现（如 CrossSectionTurnover 用 `volume` + `float_shares` 算换手率），正常写代码。
  - 只有组合也实现不了（如"分析师一致预期营收"这类清单里完全没有、也无法用现有列推导的专有数据）→ **才判断为缺列**：
  - **不要写代码，不要跑 test-and-export，不要 deploy-to-full**
  - 在测试因子目录用 Write 工具写 `{name}.missing.json` 记录缺的字段（路径见下文"缺列 JSON 格式"——**用完整路径 `/mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.missing.json`**，不要写在项目根目录的 因子产出/ 下）
  - 然后直接返回 success=true，missing_fields=true
  - 不需要修改字段或换因子

**缺列 JSON 格式**（用 Write 工具写到 `因子产出/测试/{DATE}/{report_name}/{name}/{name}.missing.json`，即 **`/mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.missing.json`**，与其他测试因子产出同一目录；目录不存在时先创建）：
```json
{
  "factor_name": "{name}",
  "report_name": "{report_name}",
  "date": "{DATE}",
  "factor_type": "{type_key}",
  "status": "missing_fields",
  "missing_fields": ["缺的字段1", "缺的字段2"],
  "checked_cols": ["因子定义推导需要的所有字段（语义）"],
  "detected_by": "show-columns",
  "message": "因子 {name} 所需字段 缺的字段1, 缺的字段2 在测试数据中不存在，因此未生成代码、未部署到全量。"
}
```

### 你的任务（只有两步）

#### 1. 写核心函数到 /tmp/factor_{name}.py
只有 Step 0 显示字段齐全时才写代码。列名以 Step 0 的 `show-columns --type` 输出为准。
**写代码前，用 Read 工具读**恰好一个**知识文件——只读与你自己 type 对应的那一个（按类型映射选，不要读其他类型，省 token）：**
```bash
# 只读这一个（根据类型选择）：
#   daily          → .claude/skills/factor/knowledge/daily.md
#   minute         → .claude/skills/factor/knowledge/minute.md
#   cross_section  → .claude/skills/factor/knowledge/cross_section.md
#   deep_learning  → .claude/skills/factor/knowledge/deep_learning.md
```
根据类型写核心函数：
{daily: **`def calc_factor_series(df, stock) -> pd.Series`**（向量化，1次调用算完全部日期）。可选写 `calc_factor_single_stock(df, trade_date, stock)` 作为 fallback，模板默认提供包装。
 minute: `def calc_factors_one_day(df, stock):`,
 cross_section: `def calc_factor_cross_section(all_data, trade_date):`,
 deep_learning: `def train_model(all_data, trade_date):` + `def predict_batch(model, data_dict, trade_date):`}

**⚠️ 各模板入参的硬事实（照此写，勿猜）**：
- **minute**：`calc_factors_one_day(df, stock)` 的 `df.index` 是 **DatetimeIndex**（模板已转换），直接用 `df.index`；`df` 是 LOOKBACK 天的滑动窗口切片（最后一天是 T 日）。不要调用 `get_level_values("datetime")`。
- **cross_section**：`calc_factor_cross_section(all_data, trade_date)` 的 `all_data` 是 **`dict {股票代码: DataFrame}`**，不是单个 DataFrame！`all_data[stock]` 是该股票截至 `trade_date` 的 **LOOKBACK 窗口切片**（含所需列）。返回 `dict {股票代码: 值}` 或 `{股票代码: {"因子名": 值}}`，需要遍历 `all_data` 的 key。行业分类用模板已加载的 `INDUSTRY_DICT`。
- **deep_learning**：`train_model(all_data, trade_date)` 的 `all_data` 同理是 dict。
- **daily**：`calc_factor_series(df, stock)` 的 `df` 是单股票全历史 DataFrame（DatetimeIndex），一次调用返回全部日期的 pd.Series。

**⚠️ 所有分钟因子用 minute 模板，不要用 minute_cs。**

**日线因子必须优先写 `calc_factor_series`（向量化版本）**：
```python
def calc_factor_series(df, stock):
    \"\"\"一次算完全部日期的因子值。返回 pd.Series(index=原日期, name=因子名)\"\"\"
    if df is None or len(df) < LOOKBACK_DAYS:
        return pd.Series(dtype=float, name=因子名)
    # 用 pandas rolling 向量化计算，避免逐日循环
    s1 = df["col1"].rolling(20, min_periods=20).sum()
    s2 = df["col2"].rolling(20, min_periods=20).mean()
    ...
    result = ...  # 组合逻辑
    result.name = "因子名"
    return result
```
**性能要求**：`calc_factor_series` 内禁止 for 循环逐行/逐日计算。必须用 pandas/numpy 向量化操作（rolling/expanding/shift/diff/groupby transform）。
`calc_factor_single_stock` 可省略（模板自动 fallback 到逐日模式，但速度慢 10~100x）。

只实现核心计算逻辑。不要写模板框架代码（数据加载、并行、涨停剔除等模板会自动处理）。

**分钟模板有两条路径，LLM 自行判断用哪个：**
- `calc_factors_one_day(df, stock)` → **非向量化**，模板按 LOOKBACK 滑动窗口逐天调用。
  适合：单日截面因子、LOOKBACK≤21 的因子。每只股票调 N 次（N=天数），每次处理 LOOKBACK 天数据。
- `calc_factor_series(df, stock)` → **向量化**，一次接收全部数据，返回 `pd.Series(index=日期, name="因子名")`。
  适合：滚动累积/平均类因子（过去 N 天累加、累乘、均值等），LOOKBACK 大的因子。每只股票只调 1 次，快 10~30 倍。
  **不需要删 `calc_factors_one_day`，模板会自动优先走 `calc_factor_series`。**

**判断原则：** 如果因子逻辑可以拆成"先算每日值，再跨日 rolling" → 用 `calc_factor_series`。如果因子逻辑依赖滑动窗口内的全量数据计算 → 用 `calc_factors_one_day`。不确定时两种都写，模板自动优先走向量化。

<!-- BEGIN_MINUTE_ONLY 主进程按 type 拼 prompt：仅 minute 因子包含以下"市场数据模式"段落 -->
**市场数据模式（仅 minute 因子需要全市场数据时才相关；daily/cross_section/deep_learning 因子跳过本段）：**

**Step 0：检查预计算文件是否存在（测试 + 全量都要检查）**
```bash
# 测试目录
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/minute_by_date/market_minute_return.parquet
# 全量目录
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/market_minute_return.parquet
```
预计算文件路径规则：`{分钟线数据目录}/stock_data/minute_by_date/market_minute_return.parquet`

**Step 1：如果不存在，立即预计算（先跑再写函数！）**
```bash
python scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试"
python scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量"
```
**必须两个目录都跑**，因为因子要在测试和全量两个环境运行。

**Step 2：在因子代码模块级加载预计算文件**
```python
# 模块级：在导入时一次性加载，不在逐日函数内重复读取
MARKET_RETURN = pd.read_parquet(
    MINUTE_BY_DATE_DIR / "market_minute_return.parquet"
)
MARKET_RETURN = MARKET_RETURN.groupby(level=0).first()  # 去重

def calc_factors_one_day(df, stock):
    # df.index 是 DatetimeIndex（模板已转换），直接用
    dt_idx = df.index
    market_ret = MARKET_RETURN.reindex(dt_idx)["market_return"].values
    # ... 后续计算
```
注意：`calc_factors_one_day` 收到的 `df.index` 是 `DatetimeIndex`。**不要调用 `get_level_values("datetime")`**，直接用 `df.index` 即可。
<!-- END_MINUTE_ONLY -->

**如果因子逻辑依赖截面数据（全市场排名/标准化/分组等）：**
- **直接跳过**，输出原始值即可。不需要做截面处理
- 或者如果截面处理是因子核心逻辑（非后处理），可以额外写一个**后处理函数**，在 test-and-export 生成 `.parquet` 后读取并做截面变换：

#### 2. 立即跑 test-and-export + deploy-to-full（写完后立刻执行，不停顿）
类型在 Phase 1 已定义，**显式传 `--type {type_key}`**（确定，不依赖自动检测）。
```bash
python scripts/claude_factor_helper.py test-and-export \
  --code /tmp/factor_{name}.py \
  --report "{report_name}" --factor "{name}" \
  --cols "{cols}" --lookback {lookback} \
  --type {type_key} \
  --description "{description}" --formulation "{formulation}" \
  --source-excerpt "{source_excerpt}" \
  --source-report-title "{report_name}" \
  --date {DATE}
```
> `{cols}` = Step 0 从 show-columns 挑出的**真实列名**（空格或逗号分隔均可，helper 自动 split）。字段齐全时才需要；缺列已返回，不会走到这一步。

**test-and-export 成功后，立即部署到全量：**
> 注意路径结构：test-and-export 输出为 `因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py`（因子目录 = `{report_name}/{name}/`，内部文件 = `{name}.code.py`）。deploy-to-full 的 `--code` 用同一路径，**不要**多套一层 `{name}` 目录。
```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py \
  --date {DATE}
```

#### ⚠️ 绝对禁止（违反将导致流程失败）
1. ❌ 不要编译代码（`py_compile`）
2. ❌ 不要 import FactorFBWorkspace
3. ❌ 不要自己加载 parquet
4. ❌ 不要自己加载 parquet 手动检查 schema —— 字段核对只用 Step 0 的 `show-columns --type` 输出
5. ❌ 不要手动 debug
6. **写代码 → 跑 test-and-export，中间不做任何事**
7. **跑通后不得再改代码**：`test-and-export` + `deploy-to-full` 一旦成功（含"缺列已返回"），**绝不再回头修改代码**——注释措辞、变量名、格式优化等一律禁止（改了=改代码=要重跑，纯浪费 token 和时间）。只有 test-and-export **失败**时，才允许按下面"如果 test-and-export 失败"的规则修改重试。

#### 如果 test-and-export 失败（含错误和超时）
- **缺字段已在 Step 0 判断处理**（判断缺列就在 Step 0 写 `{name}.missing.json` 并返回，不会走到 test-and-export）
- **普通错误**：看错误信息，修改函数代码后重新跑，最多重试 2 次
- **超时**（超过 300s 无结果）：修改代码优化性能（减天数、向量化等）后重试，最多 **2 次修改机会**
- **累计 3 次都失败** → 在结果中报告 failure，不阻塞后续因子

### 返回格式
{{"name": "{name}", "success": true/false, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null 或 "失败原因"}}

**缺字段（Step 0 判断缺列）**：返回 {"name": "{name}", "success": true, "missing_fields": true, "error": null}。不部署全量。
"""
```

#### 派发逻辑
```
all_factors = flatten(所有Phase1结果的factors)
next_idx = 0
active = {}  # {agent_id: factor_info}
results = []

for _ in range(min(5, len(all_factors))):
    agent_id = dispatch_phase2_worker(all_factors[next_idx]); next_idx += 1
    active[agent_id] = all_factors[next_idx-1]

while active:
    for agent_id in list(active.keys()):
        output = TaskOutput(task_id=agent_id, block=false, timeout=0)
        if output.status == "completed":
            results.append({agent_id: output.result})
            del active[agent_id]
            if next_idx < len(all_factors):
                new_id = dispatch_phase2_worker(all_factors[next_idx])
                active[new_id] = all_factors[next_idx]; next_idx += 1
    if active:
        import time; time.sleep(5)

# 全部完成
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
- daily 独有：1（T日=df.iloc[-1]）, 4（日线用 pct_chg）, 5（df.index.date 不放循环内）
- minute 独有：10（禁止分钟级 for 循环）, 14（禁用 minute_cs）
- cross_section/deep_learning：无独有条目（用通用即可）

1. T日 = df.iloc[-1]
2. 返回 `{"因子名": np.nan}`，不返回 None
3. 禁止月末判断
4. 日线用 `pct_chg` 或 `close.pct_change()`
5. `df.index.date` 不放循环内
6. 布尔 shift() 后 fillna(False)
7. 禁止 `len(df) < X` 做上市天数筛选
8. 禁止未来数据
9. np.inf/-np.inf → np.nan
10. 禁止分钟级 for 循环（用向量化操作）
11. 禁止 `transform('count')` → 用 `transform('size')`
12. 禁止 `rolling.apply(lambda)`
13. 禁止合成因子
14. **所有分钟因子用 minute 模板，不用 minute_cs**（太慢，且截面标准化无意义）
15. **lookback 只含核心计算天数**，不含论文末尾的截面标准化/std20/取波动率等后处理
16. **禁用截面操作（排名/标准化/行业中性化）**：因子只输出个股原始值。如果截面是核心逻辑，额外写后处理函数对产出 .parquet 做截面变换
17. **`df.index.date` 返回 ndarray**，没有 `.isin()` 方法。用 `np.isin(date_arr, list)` 替代