# /factor — 研报/文章因子端到端处理

完全替代 `start` 命令。全自动决策，不问用户任何问题。

## 用法

- `/factor` — 扫描所有未处理内容，**全部并行提取 → 编码 → 测试 → 导出到 literature_reports**
- `/factor papers/inbox/某篇.pdf` — 处理单个 PDF
- `/factor 一段因子描述` — 处理纯文本

## 核心规则

0. **高效使用工具，禁止重复造轮子**：所有数据获取（PDF提取、网站抓取、因子测试、部署等）都必须用 `claude_factor_helper.py` 提供的命令，不要自己安装库、写爬虫、或手动处理数据。给的命令不够用再问，不要擅自尝试其他方法。

1. **全自动**：不问用户。数据不可用→跳过；3次修复仍失败→跳过并记录；网站反爬/无法读取→跳过并 mark-done
2. **强制提取，宁可多提不可漏提**：每篇最多15个因子。**无论文章是否明确写了"因子"二字，都必须从中提炼可量化信号**。择时策略的阈值→转为截面排序因子；行业轮动→提炼行业偏离度；选股逻辑→拆解为多个单维度因子。**不允许"纯方法论跳过"——任何文章都有可因子化的东西。**
3. **子因子独立提取**，不合并。方法论也要提取（阈值未给用median）
4. **formulation 必须完整**：从原始数据字段出发的完整数学表达式，禁止 `f(·)` 等占位符
5. **禁止合成因子**（等权合成、IC加权、Z-score标准化后相加等）
6. **唯一跳过场景**：所需数据完全不可用（如需要专有数据库API、另类数据）。择时、选基、宏观、债券等研报**不跳过**，从中提取可量化的截面/时间序列因子
7. `source_excerpt` 从研报原文直接复制，不做概括

## 工作流

```
scan-pending → 收集所有待处理项（papers + websites + ideas 合并为一个队列）
  → 启动 5 个 sub-agent worker（run_in_background=true）
  → 每个 worker 领一个任务，处理完立即领下一个
  → 队列为空后，worker 返回结果
→ 主 Claude 收集全部结果 → mark-done
```

**`/factor` 只负责测试线（提取 → 编码 → 测试 → 导出到 `literature_reports/`）。全量线（`文献因子_全量/`）是独立命令 `run-full`，不在 `/factor` 流程内。**

核心原则：工作队列模式。把所有待处理项（不论类型）放进一个队列，启动固定数量的 worker，谁空闲谁领任务。不允许任何 worker 空闲等待。

错误做法（禁止）：
- ❌ 分批启动（先 paper 再 website）
- ❌ 等全部 paper 结束再启动 website
- ❌ 主 Claude 串行派发

正确做法（强制）：
- ✅ scan-pending 后，所有 papers + websites + ideas 合并为一个任务列表
- ✅ 同时启动 5 个 background sub-agent
- ✅ 每个 sub-agent 处理完当前任务后，主 Claude 立即从队列取下一个任务派给它
- ✅ 队列为空时，让 sub-agent 返回结果

### Step 1: 扫描 + 建队列
```
python scripts/claude_factor_helper.py scan-pending
```
返回 JSON，含 `papers[]`、`websites[]`、`ideas[]` 三个列表。

**将所有待处理项合并为一个队列**（不论类型），并按"浅层优先"排序：deep_learning 类型的研报放到末尾，确保普通因子不会被 DL 因子阻塞。每个队列元素含：
- `type`: "paper" / "website" / "idea"
- `path`: PDF 路径（仅 paper）
- `filename`: PDF 文件名（仅 paper）
- `index`: 在 scan-pending 返回列表中的序号（用于 website extract）
- `slug`: 唯一标识（用于网站 extracted.json 读取和 mark-done）
- `text`: idea 原文（仅 idea）
- `title`: 标题（可能为 null）
- `source_excerpt`: 网站摘要（仅 website）

### Step 2: 启动 worker 池

同时启动 **5 个** `general-purpose` sub-agent（`run_in_background=true`）。

**队列排序规则（浅层优先）：** 把所有任务按文件名/路径中是否含 "深度学习"、"GRU"、"TCN"、"LSTM"、"deep_learning" 等关键词排序。含这些词的排到队列末尾，不含的优先执行。这样普通因子先跑完释放 worker，DL 因子最后只剩少量 worker 处理，避免 GPU 争抢。

派发逻辑：
```
tasks = flatten(papers + websites + ideas)  # 全部任务
next_task_idx = 0

for _ in range(5):  # 启动 5 个 worker
    task = tasks[next_task_idx]; next_task_idx += 1
    start_worker(task)  # run_in_background=true

每当一个 worker 返回结果:
    if next_task_idx < len(tasks):
        task = tasks[next_task_idx]; next_task_idx += 1
        start_worker(task)  # 立即派新任务
    else:
        worker_pool_empty_slots += 1

当 worker_pool_empty_slots == 5:  # 全部完成
    进入 Step 3
```

**关键：worker 返回结果后，必须在同一轮立即派发下一个任务，不能等所有 worker 都结束再派。**

#### 构造 sub-agent 的 prompt

```
subagent_type=general-purpose
prompt = f"""
你是一个研报因子提取 agent。请全权处理以下这篇报告：读原文 → 提取因子 → 逐个编码测试。

### 报告信息
- 类型: {{type}}  (paper / website / idea)
- PDF 路径: {{path}}  (仅 paper)
- 文件名: {{filename}}  (仅 paper)
- 网站索引: {{index}}  (仅 website)
- slug: {{slug}}

### 你的任务

#### Step A: 获取原文
{'  python scripts/claude_factor_helper.py extract-pdf ' + path if type == 'paper'
 else '  python scripts/claude_factor_helper.py extract-website --index ' + str(index) if type == 'website'
 else '  直接使用以下 text 字段作为原文'}
**直接使用以上命令的输出结果**，不要自己安装 pdfminer/pymupdf 等库去读 PDF，不要用其他方式获取内容。命令的输出就是 JSON 格式的全文。
输出是 JSON，含 `content`（全文）和 `metadata`（标题、作者等）。
仔细阅读全文，理解其投资逻辑和核心方法论。**即使全文没提"因子"二字，也必须从中提炼可量化的信号。**

**⚠️ 重要：如果 extract 失败（如网站反爬/无法读取/返回空内容），直接跳过该任务，返回：**
```json
{"report_name": "{{报告标题}}", "factors": [], "skipped": true, "skip_reason": "网站无法读取"}
```
**不要尝试其他方式获取内容，不要重试。**

#### Step B: 定义因子
分析原文，定义该篇报告的所有因子。每条因子包含：
- name: 因子名（英文驼峰）
- description: 中文描述
- formulation: 从原始数据字段出发的完整数学表达式
- type: daily / minute / cross_section / minute_cs / deep_learning
- lookback: 回溯天数（1月≈20, 1季≈60, 6月≈120, 1年≈250）
- cols: 需要的列名列表
- source_excerpt: 从原文直接复制

**规则：** 最多15个因子；formulation 必须完整，禁止 `f(·)`；子因子独立提取不合并；宁可多提不可漏提。

**数据可用性检查：** 定义因子前先运行 `python scripts/claude_factor_helper.py show-columns` 查看可用列。如果因子需要的数据在可用列中不存在（如期权数据、期货数据、Level2 订单簿、另类数据等），直接跳过该因子，不浪费时间去编码测试。**只有所需列都在可用列表中的因子才进入 Step C。**

全部定义完成后，保存：
  python scripts/claude_factor_helper.py save-extracted --name "{{报告标题}}" < 你的因子列表 JSON

#### Step C: 逐个编码 + 测试（顺序执行）
对 Step B 定义的每个因子，逐个执行：

1. 确定模板类型 → 预读知识文件（在 prompt 的最后提供了）
2. 对每个因子，先调用 find_similar_factors() 获取同类因子参考
3. 调用 retrieve_domain_knowledge() 获取相关领域知识（涨停规则、A 股惯例等）：
   ```bash
   # 用因子描述作为检索 query，获取相关市场规则参考
   python scripts/claude_factor_helper.py retrieve-knowledge <因子描述/关键词>
   ```
4. **只写核心函数**到 `/tmp/factor_{{name}}.py`：
   - 根据上表确定函数签名，**只实现核心计算逻辑**
   - **不要写模板框架代码**（数据加载、日期遍历、并行调度、输出格式转换、涨停剔除等都由模板自动处理）
   - **不要调试模板行为**（如日期索引格式、数据加载方式）— 模板已处理好，看 test-and-export 的错误信息修改你的核心函数即可
   - **辅助函数必须定义在核心函数内部**（`def inner_func(): ...` 嵌在核心函数内），不得定义为外部函数（否则模板列检测扫不到 → KeyError）
5. 运行 test-and-export（**必须传入所有元数据参数，不要留空**）:
   python scripts/claude_factor_helper.py test-and-export \
     --code /tmp/factor_{{name}}.py \
     --report "{{报告标题}}" --factor "{{name}}" \
     --cols "{{cols}}" --lookback {{lookback}} \
     --description "{{description}}" --formulation "{{formulation}}" \
     --source-excerpt "{{source_excerpt}}" \
     --source-report-title "{{报告标题}}" \
     --source-report-path "{{path if type == 'paper' else ''}}"
6. 如果失败 → 修改核心函数 → 重试，最多3次。**不要修改模板，不要自己写测试去调试数据格式。**

**重要：** 因子之间可以共享上下文。因子2如果与因子1计算相似，可以复用因子1的代码模式。顺序执行完所有因子。

#### Step D: 返回结果

返回 JSON:
{{
  "report_name": "{{报告标题}}",
  "factors": [
    {{"name": "因子1", "success": true, "code_path": "...", "error": null}},
    {{"name": "因子2", "success": false, "code_path": "...", "error": "失败原因", "retries": 3}}
  ]
}}

### 模板类型 → 用户函数名对照
- daily → def calc_factor_single_stock(df, trade_date, stock):
- minute → def calc_factors_one_day(df, stock):
- cross_section → def calc_factor_cross_section(all_data, trade_date):
- minute_cs → def calc_factor_minute_raw(df, stock): + def cross_section_transform(all_values):
- deep_learning → def train_model(all_data, trade_date): + def predict_batch(model, data_dict, trade_date) -> (factor_name, {stock: value})
- deep_learning (fallback) → def train_model(all_data, trade_date): + def predict(model, df, trade_date, stock):
  > `predict_batch` 优先（GPU batch推理），`predict` 作为无GPU时的回退

### 编码硬约束
{ENCODING_RULES}
"""
```

### Step 3: 部署到全量目录 + 同步远程 + 标记完成

等所有 sub-agent 完成后，对**每个成功的因子**执行一条命令搞定：

```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code literature_reports/<报告>/<因子>/<因子>.code.py
```

这个命令自动完成：
1. 复制 `.code.py` 到 `文献因子_全量/<报告>/<因子>/`，同时把 DATA_DIR 路径中的 `_1000` 去掉（测试数据→全量数据）
2. 从测试目录继承 `meta.json`，标注 `pipeline_status: "deployed"`（未运行，仅部署）
3. 输出部署结果 JSON

然后同步到远程：
```bash
# 同步单个因子
python scripts/claude_factor_helper.py sync-full --report "报告标题" --factor "因子名"

# 或同步整份报告的所有因子
python scripts/claude_factor_helper.py sync-full --report "报告标题"

# 或同步所有已部署因子
python scripts/claude_factor_helper.py sync-full --all
```

最后标记完成：
```bash
python scripts/claude_factor_helper.py mark-done --name "文件名.pdf"
# 或 website/idea 用 slug:
python scripts/claude_factor_helper.py mark-done --name <slug>
```

---

**`/factor` 到此结束。产出：**

| 目录 | 内容 | 状态 |
|------|------|------|
| `literature_reports/<report>/<factor>/` | .code.py, .parquet (测试300只), .meta.json | 测试通过 |
| `文献因子_全量/<report>/<factor>/` | .code.py (全量数据路径), meta.json (pipeline_status=deployed) | 已部署，未运行 |

全量计算是独立命令 `run-full`，需手动触发：
```bash
python scripts/claude_factor_helper.py run-full \
  --code 文献因子_全量/<report>/<factor>/<factor>.code.py \
  --factor-name MyFactor \
  --report-name "报告名"
```

## 编码硬约束

1. **T日 = df.iloc[-1]**
2. 条件不满足返回 `{"因子名": np.nan}`，不返回 None
3. **禁止月末判断**：每交易日都算，用滚动窗口
4. 日线用 `pct_chg`（%）或 `close.pct_change()`，无 return 列
5. 复权价：`close * factor`；单日涨跌用 `pct_chg`
6. `df.index.date` 不放循环内
7. 布尔序列 shift() 后 fillna(False)
8. 字段不可用时用语义最接近的替代，注明"近似"
9. GPU 仅 4GB，分 batch
10. **禁止 `len(df) < X` 做上市天数筛选**
11. **禁止未来数据**
12. 日频窗口是整数交易日数
13. np.inf/-np.inf → np.nan
14. **禁止 Python for 循环遍历分钟级数据**：分钟因子（minute/minute_cs）`calc_factors_one_day` 内禁止逐分钟 for 循环。必须用 `groupby+shift`、`numpy` 向量化操作。每只股票每次调用的数据跨度固定（LOOKBACK_DAYS 天 × 240 分钟），必须 vectorized
15. **分钟因子返回的 `pd.Series`，index 用 `datetime.date`**：`r.index = pd.Index(r.index.date)`，不要返回 `DatetimeIndex`。日线/daily 因子无此问题（按天返回 dict）。
16. **禁止 `transform('count')`，统一用 `transform('size')`**：pandas 2.3 + `copy_on_write=False` 下 `transform('count')` 触发 `UnboundLocalError`，`transform('size')` 语义完全等价且无此 bug。
17. **分钟因子需要按天滑动窗口时，用 dict 预分组代替 `np.isin`**：`np.isin(d_dates, window)` 对2000天×5000分钟构造全量掩码 O(N²)，改为 dict 按日期预收集值再窗口内 extend，O(N) 且 ~4x 加速。注意 `d_dates` 转成 `list(df.index.date)` 避免线程共享问题。模式：

```python
# 预分组（快4倍）
d_dates = list(df.index.date)  # list防线程共享
j_bool = condition.values
day_dict = {}
for j in range(len(d_dates)):
    if j_bool[j]:
        dt = d_dates[j]
        if isinstance(dt, pd.DataFrame): dt = dt.iloc[0, 0]  # 线程安全保护
        day_dict.setdefault(dt, []).append(value[j])

unique_dates = sorted(set(d_dates))
result = pd.Series(index=pd.Index(unique_dates), dtype=float)
for i in range(20, len(unique_dates)):
    d = unique_dates[i]
    vals = []
    for dd in unique_dates[i-20:i+1]:
        arr = day_dict.get(dd)
        if arr is not None: vals.extend(arr)
    # ... 计算 ...
```

## 禁止规则

1. **禁止 `rolling.apply(lambda)`** → loky segfault
2. **禁止分钟因子读日线数据**
3. **禁止合成因子**
4. **禁止 `transform('count')`** → pandas 2.3 bug，统一用 `transform('size')`

## 收益率区分

- 日收益率（含隔夜跳空）：`pct_chg` 列（%）或 `close.pct_change()`
- 日内收益率：`close / open - 1`

## 领域知识 RAG

编码前可通过 `retrieve_domain_knowledge()` 检索市场规则参考，知识库位于 `.claude/skills/factor/knowledge/`：
- `limit_up.md` — 涨跌停规则、代码区间判断
- `daily.md` / `minute.md` / `cross_section.md` — 各类型模板的数据列定义和约束
- `precomputed.md` — 预计算因子列（121个）

```bash
python scripts/claude_factor_helper.py retrieve-knowledge "涨停阈值 科创板"
python scripts/claude_factor_helper.py retrieve-knowledge "日线计算收益率"
python scripts/claude_factor_helper.py retrieve-knowledge "分钟数据列名"
```

## lookback 转换

1月≈20天, 1季≈60, 6月≈120, 1年≈250

## 输出目录

```
git_ignore_folder/factor_outputs/literature_reports/<report>/<factor>/
  .code.py, .parquet, .meta.json
```
