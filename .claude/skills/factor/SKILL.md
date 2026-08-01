# /factor — 研报/文章因子端到端处理（两阶段架构）

## 用法

- `/factor` — 扫描所有未处理内容，处理所有待处理项
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

## 两阶段工作流

```
scan-pending → [Phase 1] 提取+定义因子 (每个paper一个sub-agent, 只做extract+define)
            → [Phase 2] 编码+测试 (每个factor一个sub-agent, 只写核心函数+跑test-and-export)
            → deploy-to-full + mark-done
```

**核心原则：每个 sub-agent 的任务极其简单，没有犯错空间。**

---

### Step 0: 扫描
```bash
python scripts/claude_factor_helper.py scan-pending
```
输出 JSON，含 `papers[]`、`websites[]`、`ideas[]` 三个列表。

**队列排序规则：** 含 "深度学习/GRU/TCN/LSTM/deep_learning" 的排末尾，其他优先。

---

### Step 1: Phase 1 — 提取 + 定义因子

对每个待处理项（paper/website/idea），启动一个 sub-agent。**每个 sub-agent 只做两件事：提取原文 → 定义因子。不做编码测试。**

sub-agent 同时启动最多 **5 个**（`run_in_background=true`），完成后主 Claude 立即派发下一个。

#### Phase 1 sub-agent prompt（极简 ~30 行）

```
subagent_type=general-purpose
prompt = """
你只做一件事：读原文 → 定义因子。不做编码，不做测试。

### 输入
- 类型: {type}
- PDF路径: {path}  (仅paper)
- 文本: {text}  (仅idea)
- 网站索引: {index} (仅website)

### Step 1: 获取原文
{{
  'paper': f'python scripts/claude_factor_helper.py extract-pdf "{path}"',
  'website': f'python scripts/claude_factor_helper.py extract-website --index {index}',
  'idea': '直接使用 text 字段'
}}
命令输出是 JSON。如果提取失败/空内容，直接跳过：{{"skipped": true}}

### Step 2: 定义因子
运行 python scripts/claude_factor_helper.py show-columns 查看可用列。

分析原文，定义所有因子。每条：
- name: 英文驼峰
- description: 中文
- formulation: 完整数学表达式
- type: daily/minute/cross_section/minute_cs/deep_learning
- lookback: 天数 (1月≈20, 1季≈60, 6月≈120, 1年≈250)。注意：minute/minute_cs 类型最大 120（约 6 个月），即使论文用 1 年也要截断
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
tasks = flatten(papers + websites + ideas, DL排最后)
for _ in range(min(5, len(tasks))):
    dispatch_phase1_worker(task)

每当一个 worker 返回:
    results.append(worker.result)
    if 还有剩余任务: dispatch_phase1_worker(下一个任务)
    elif len(results) == len(tasks): 进入 Phase 2
```

---

### Step 2: Phase 2 — 编码 + 测试

收集 Phase 1 所有成功定义的因子，**为每个因子启动一个 sub-agent**。每个 sub-agent 只做：**写核心函数 → 跑 test-and-export**。

最多同时启动 **5 个** sub-agent。主 Claude 控制派发。

#### Phase 2 sub-agent prompt（极简 ~30 行）

```
subagent_type=general-purpose
prompt = """
你只做一件事：写一个核心函数并跑 test-and-export。不做其他任何事。

### 参考同类型因子（节省 token）
查看已生成的成功因子代码，参考其核心函数结构：
ls literature_reports/{DATE}/{report_name}/{name}/{name}.code.py
只看核心函数部分（calc_factor_xxx），不要复制模板代码。
注意参考同类型因子（daily 参考 daily，minute_cs 参考 minute_cs）。

### 因子定义
- 因子名: {name}
- 类型: {type}
- 函数名: {func_name}  (见下方对照表)
- lookback: {lookback}
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
 minute_cs: `def calc_factor_minute_raw(df, stock):` + `def cross_section_transform(all_values):`,
 deep_learning: `def train_model(all_data, trade_date):` + `def predict_batch(model, data_dict, trade_date):`}

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

**性能注意（分钟截面）：** 全量 5435 只股票 × 120 天分钟数据，避免 Python 逐元素循环（`for i in range` + `np.argmin`/`np.sum` 等）。优先用 numpy 向量化、O(n) 单调队列或前缀和。

#### 2. 立即跑 test-and-export（写完后立刻执行，不停顿）
```bash
python scripts/claude_factor_helper.py test-and-export \
  --code /tmp/factor_{name}.py \
  --report "{report_name}" --factor "{name}" \
  --cols "{cols}" --lookback {lookback} \
  --description "{description}" --formulation "{formulation}" \
  --source-excerpt "{source_excerpt}" \
  --source-report-title "{report_name}" \
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

for _ in range(min(5, len(all_factors))):
    dispatch_phase2_worker(all_factors[next_idx]); next_idx += 1

每当一个 worker 返回:
    results.append(worker.result)
    if next_idx < len(all_factors):
        dispatch_phase2_worker(all_factors[next_idx]); next_idx += 1
    elif len(results) == len(all_factors):
        进入 Step 3
```

---

### Step 3: 部署 + 同步 + 标记完成

**⚠️ 本步骤只做下面三件事，绝不跑全量计算（`run_all.py`/`run_factor_full.py` 等一律不碰）。全量 parquet 由用户自行启动的增量运算负责。** 不要因为全量目录里还没有 .parquet 就"好心"去补算。

对每个成功的因子：
```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code literature_reports/{DATE}/{report}/{factor}/{factor}.code.py \
  --date {DATE}
```

同步到远程：
```bash
python scripts/claude_factor_helper.py sync-full --all --date {DATE}
```

标记完成：
```bash
python scripts/claude_factor_helper.py mark-done --name "文件名.pdf"
# 或 website/idea:
python scripts/claude_factor_helper.py mark-done --name <slug>
```

**Step 3 完成后即结束，不执行任何额外计算步骤。**

---

## 模板类型 → 函数名对照
- daily → `def calc_factor_series(df, stock) -> pd.Series`（向量化，优先）。可选 `def calc_factor_single_stock(df, trade_date, stock)`（逐日 fallback）
- minute → `def calc_factors_one_day(df, stock):`
- cross_section → `def calc_factor_cross_section(all_data, trade_date):`
- minute_cs → `def calc_factor_minute_raw(df, stock):` + `def cross_section_transform(all_values):`
- deep_learning → `def train_model(all_data, trade_date):` + `def predict_batch(model, data_dict, trade_date):`

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