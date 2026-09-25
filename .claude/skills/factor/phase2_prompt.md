# Phase 2 因子编码 agent 作业指导（模板）

> 主 agent 派发时**不需要**逐字输出本文件内容，只需在 prompt 里给出：
> 「先 Read 本文件 `.claude/skills/factor/phase2_prompt.md`，按其说明执行」+ 下方参数表。
> 参数占位符：`{name}` `{type}` `{type_key}` `{func_name}` `{lookback}` `{report_name}`
> `{formulation}` `{description}` `{source_excerpt}`

---

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

**`show-columns --type` 只有两个合法取值，没有别的：**

| 你的因子类型 | 跑哪个命令 |
|---|---|
| `minute` | `--type minute` |
| `daily` | `--type daily_single` |
| **`cross_section`** | **`--type daily_single`** ← 不是 `--type cross_section`（不存在） |
| **`deep_learning`** | **`--type daily_single`** ← 不是 `--type deep_learning`（不存在） |

```bash
# minute 类型
python scripts/claude_factor_helper.py show-columns --type minute
# daily / cross_section / deep_learning（三者都用这个）
python scripts/claude_factor_helper.py show-columns --type daily_single
```

> ⚠️ **不要尝试 `--type cross_section` 或 `--type deep_learning`**——参数值不存在，试了只会浪费一轮往返。
> 列清单只有两套：**分钟线列**（`--type minute`）和**日线+非行情列**（`--type daily_single`）。
> `cross_section` / `deep_learning` 用的就是日线+非行情那套。

**以这次运行的输出为唯一字段核对依据。**

> 🚫 **绝对禁止：对数据仓库做全仓递归扫描**
> `/mnt/d/paper-factor-data` 有 **66 GB / 143,797 个文件**。以下命令会读 66GB 二进制 parquet、
> 产生海量输出，**导致 agent 卡死 600 秒被杀**（已发生 4 次）：
> - `grep -r` / `grep -R` / `rg` 扫数据仓库
> - `ls -R` / `tree` 扫数据仓库
> - `find` 扫数据仓库**且不带 `-maxdepth`**
>
> **判断列是否存在，只看 `show-columns` 的输出，不要去磁盘上找证据。**
> 这些命令已被 `PreToolUse` 钩子硬拦截（`.claude/hooks/block_datascan.py`），执行会直接报错。

### 🚫 禁止探索清单（每条都实测浪费 5-15 分钟）

**每次工具调用 = 一次完整 API 往返 ≈ 20 秒。** 计算本身只要 20-60 秒。
曾有一个截面因子 agent 花了 **990 秒**，其中 **39 次调用在探索、只有 3 次在干活**，计算仅占 17 秒。

**以下操作一律禁止：**

| ❌ 禁止 | 为什么 |
|---|---|
| 读**其他因子**的 `.code.py` / `.meta.json` / `.parquet` | 你不需要参考别人的实现，本文档已给全接口 |
| `Read` / `grep` **`claude_factor_helper.py`** 或 **`factor.py`** 的源码 | 命令用法和函数签名本文档已给全，**不要去找源码验证** |
| 用 `pyarrow` / `pd.read_parquet` **直接读 parquet 看 schema** | 字段核对**只用** `show-columns` |
| 为了找某个列去翻数据仓库 / 源码 / memory / barra / 备份 | 字段清单只由 `show-columns` 决定，翻别处纯浪费 token |
| `ls` 反复翻 `因子产出/` 目录 | 与你的任务无关 |
| 试 `--type cross_section` 等不存在的参数值 | 见上方表格，只有两个合法值 |

**本文档给出的事实就是权威。** 直接照做，不要"眼见为实"式地翻源码/翻数据。

> 这些探索已被 `PreToolUse` 钩子（`.claude/hooks/block_explore.py`）硬拦截，执行会直接报错。

**字段清单以你自己跑的 `show-columns` 输出为唯一依据。**

数据会持续补充字段，所以**不要凭记忆或本文档假定有哪些列**——每次都实跑 show-columns，以当次输出为准。

- **因子需要的字段在输出里能找到 → 字段齐全**：
  - **从 show-columns 输出里挑出每个字段对应的真实列名**（如"复权收盘价"对应 `close` × `factor`），这些真实列名就是本因子的 `{cols}`
  - 继续 Step 1 写代码，`--cols` 用这些真实列名
- **因子需要的字段在当次输出里找不到 → 判断缺列**：
  - 允许做的唯一推导：用**当次清单里已有的列**做四则运算 / 差分 / 滚动。例如收益率 `close.pct_change()`、昨日收盘 `close.shift(1)`、复权价 `close * factor`
  - 如果连推导所需的原料列都不在当次清单里 → **判缺列，停止**
  - **不要为了找这个字段去翻任何其他地方**——字段清单只由当次 show-columns 输出决定
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

> 🚫 **不要去找模板源码来验证签名**（不要 `grep`/`find` 搜索 `CROSS_SECTION_FRAMEWORK_TEMPLATE`、`DAILY_FRAMEWORK_TEMPLATE` 等）。
> 下面列出的签名**就是权威**，直接照着写。曾有一个 agent 为了找模板定义而 `grep -rn ... /` 扫全盘，
> **白烧 64 分钟**（该命令现已被钩子拦截）。
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
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试"
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量"
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

**返回值只有下面这一行 JSON，前后不加任何文字。**
禁止附加：完成说明、实现要点、判定理由、注意事项、代码摘要、验证过程、schema 普查结果。
需要留档的说明一律写进**代码注释**或 **missing.json**，不要放进返回值。

成功：
{{"name": "{name}", "success": true, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null}}

缺字段（Step 0 判断缺列）：
{{"name": "{name}", "success": true, "missing_fields": true, "code_path": null, "error": null}}

失败：
{{"name": "{name}", "success": false, "missing_fields": false, "code_path": null, "error": "不超过 20 字的失败原因"}}
