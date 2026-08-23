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
5. **唯一跳过场景**：所需数据完全不可用（如专有数据库API）。择时/选基/宏观/债券不跳过
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
运行 python scripts/claude_factor_helper.py show-columns 查看可用列。

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
- cols: 列名列表
- source_excerpt: 原文复制

最多15个因子。formulation 必须完整。只有可用列存在的因子才保留。

### Step 3: 保存
python scripts/claude_factor_helper.py save-extracted --name "标题" --date {DATE} < 因子JSON

### Step 4: 返回
{{{{
  "report_name": "标题",
  "date": "{DATE}",
  "factors": [{{"name": "F1", "type": "daily", "lookback": 20, "cols": ["close"]}}, ...],
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

### 参考同类型因子（节省 token）
查看已生成的成功因子代码，参考其核心函数结构：
ls /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py
只看核心函数部分（calc_factor_xxx），不要复制模板代码。

### 因子定义
- 因子名: {name}
- 类型: {type}（minute/daily/cross_section/deep_learning）
- 函数名: {func_name}  (见下方对照表)
- lookback: {lookback}  **（⚠️ 只含核心计算天数，不含论文末尾的截面标准化/std20后处理）**
- 列: {cols}
- 报告名: {report_name}
- formulation: {formulation}
- description: {description}
- source_excerpt: {source_excerpt}

### 你的任务（只有两步）

#### 1. 写核心函数到 /tmp/factor_{name}.py
根据类型写核心函数：
{daily: **`def calc_factor_series(df, stock) -> pd.Series`**（向量化，1次调用算完全部日期）。可选写 `calc_factor_single_stock(df, trade_date, stock)` 作为 fallback，模板默认提供包装。
 minute: `def calc_factors_one_day(df, stock):`,
 cross_section: `def calc_factor_cross_section(all_data, trade_date):`,
 deep_learning: `def train_model(all_data, trade_date):` + `def predict_batch(model, data_dict, trade_date):`}

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

**市场数据模式（因子需要全市场分钟收益率/总成交量等作为输入）：**

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

**test-and-export 成功后，立即部署到全量：**
```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py \
  --date {DATE}
```

#### ⚠️ 绝对禁止（违反将导致流程失败）
1. ❌ 不要编译代码（`py_compile`）
2. ❌ 不要 import FactorFBWorkspace
3. ❌ 不要自己加载 parquet
4. ❌ 不要检查 schema
5. ❌ 不要手动 debug
6. **写代码 → 跑 test-and-export，中间不做任何事**

#### 如果 test-and-export 失败（含错误和超时）
- **普通错误**：看错误信息，修改函数代码后重新跑，最多重试 2 次
- **超时**（超过 300s 无结果）：修改代码优化性能（减天数、向量化等）后重试，最多 **2 次修改机会**
- **累计 3 次都失败** → 在结果中报告 failure，不阻塞后续因子

### 返回格式
{{"name": "{name}", "success": true/false, "code_path": "/tmp/factor_{name}.py", "error": null 或 "失败原因"}}
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