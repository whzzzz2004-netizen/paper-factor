# Phase 2 · 第二步：写核心函数 + 跑测试（字段已确认齐全）

> **前提**：你已按 `phase2_check.md` 跑过 `show-columns`，确认字段齐全，并挑出了本因子的 `{cols}`。
> **不要重复跑 show-columns。**

### 你的因子

- 因子名: `{name}`
- 类型: `{type}`（minute/daily/cross_section/deep_learning）
- 函数名: 见下方对照表
- lookback: `{lookback}`  **（只含核心计算天数，不含论文末尾的截面标准化/std20 后处理）**
- 报告名: `{report_name}`
- 因子定义: 见主 agent prompt 给的提取命令（`formulation` / `description` / `source_excerpt`）

> ⚠️ **类型不符时自己改，不要翻源码。**
> 若上面给的 `{type}` 是 `daily` / `minute`，但 `formulation` 里出现了需要**同行股票**才能算的量
> （行业均值 / 行业内排名 / 行业分位 / 行业偏离 / 全市场分组…），**直接把模板换成
> `cross_section`**（函数签名 `calc_factor_cross_section(all_data, trade_date)`，
> 见 `knowledge/cross_section.md`），并在 `test-and-export` 上用 `--type cross_section`。
> 单股模板的入参只有这一只股票，**定义表达不了就是类型标错了，不是模板缺功能**。
> 不要去 grep/读 `rdagent/.../factor.py` 或 helper 源码找答案 —— 那是被硬拦截的无效探索。

---

## 1. 写核心函数到 /tmp/factor_{name}.py

**写代码前，用 Read 工具读**恰好一个**知识文件——只读与你自己 type 对应的那一个（不要读其他类型）：**

```
#   daily          → .claude/skills/factor/knowledge/daily.md
#   minute         → .claude/skills/factor/knowledge/minute.md
#   cross_section  → .claude/skills/factor/knowledge/cross_section.md
#   deep_learning  → .claude/skills/factor/knowledge/deep_learning.md
```

函数签名与各类型入参（`df` 的含义、`df.index` 类型、T 日取值、`all_data` 的结构、
分钟两条路径与市场数据预计算等）**全部在该知识文件里**，照它写。

> 🚫 **不要去找模板源码验证签名**（不要 grep 搜 `CROSS_SECTION_FRAMEWORK_TEMPLATE` 等）。
> 知识文件里列出的签名**就是权威**。

### 日线因子必须写 `calc_factor_series`（向量化）

```python
def calc_factor_series(df, stock):
    """一次算完全部日期的因子值。返回 pd.Series(index=原日期, name=因子名)"""
    if df is None or len(df) < LOOKBACK_DAYS:
        return pd.Series(dtype=float, name=因子名)
    # 用 pandas rolling 向量化计算，避免逐日循环
    s1 = df["col1"].rolling(20, min_periods=20).sum()
    result = ...  # 组合逻辑
    result.name = "因子名"
    return result
```

**禁止 for 循环逐行/逐日计算**，必须用 pandas/numpy 向量化（rolling/expanding/shift/diff/groupby transform）。
`calc_factor_single_stock` 可省略（模板自动 fallback，但慢 10~100x）。

只实现核心计算逻辑，**不要写模板框架代码**（数据加载、并行、涨停剔除等模板自动处理）。

### 通用硬约束

- 条件不满足时返回 `{"因子名": np.nan}`，**不返回 `None`**
- **禁止未来数据**：任何用到的值在 T 日及之前必须已知
- **禁止合成因子**（一个因子只算一件事）
- **禁用截面操作**（排名/标准化/行业中性化）：只输出个股原始值。若截面是核心逻辑，额外写后处理函数对产出 `.parquet` 做截面变换
- 禁止月末判断、禁止用 `len(df) < X` 做上市天数筛选
- 禁止 `transform('count')` → 用 `transform('size')`；禁止 `rolling.apply(lambda)`
- 布尔序列 `shift()` 后必须 `fillna(False)`；`np.inf` / `-np.inf` → `np.nan`
- **禁止分钟级 for 循环**（用向量化操作）
- **`INDUSTRY_DICT[股票代码]` = 申万一级行业名**，需要「行业分类」时用它（不写进 `--cols`）

---

## 2. 立即跑 test-and-export + deploy-to-full（写完后立刻执行，不停顿）

**显式传 `--type {type_key}`**（确定，不依赖自动检测）。

```bash
timeout 300 python scripts/claude_factor_helper.py test-and-export \
  --code /tmp/factor_{name}.py \
  --report "{report_name}" --factor "{name}" \
  --cols "{cols}" --lookback {lookback} \
  --type {type_key} \
  --description "{description}" --formulation "{formulation}" \
  --source-excerpt "{source_excerpt}" \
  --source-report-title "{report_name}" \
  --date {DATE}
echo "exit=$?"      # 124 = 超时被杀
```

> `timeout 300` 上限 5 分钟，**必须加**（见下方「超时熔断」）。
> `{cols}` = 你在 phase2_check.md 里从 show-columns 挑出的**真实列名**（空格或逗号分隔均可）。

> ⚠️ `--description` / `--formulation` / `--source-excerpt` 从提取命令的输出里取；
> 含特殊字符（引号、换行、`$`）时用单引号包裹或写入临时文件传参。

**test-and-export 成功后，立即部署到全量：**

```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py \
  --date {DATE}
```
> deploy-to-full 的 `--code` 用上面这个路径，**不要**多套一层 `{name}` 目录。

### 想验证假设？用 `--dry-run`，不要造假报告名

怀疑「窗口给几行 / lookback 该填几 / 某列在不在 / 为什么全是 NaN」时，用 **`--dry-run`**：
它照常跑测试并回传诊断（`result_shape` / `non_null_ratio` / 你的 print 输出），
但**不写任何产物**。

```bash
timeout 300 python scripts/claude_factor_helper.py test-and-export \
  --code /tmp/factor_{name}.py --report "{report_name}" --factor "{name}" \
  --cols "{cols}" --lookback {lookback} --type {type_key} --dry-run
```

> 🚫 **禁止用假报告名做试错**（`--report pk1`、`--report probe`、`--report v0`、`--report dbg`…）。
> 每跑一次就会在产出目录留下一个空壳报告，污染后续 `ls` 与统计，且这些目录**不会被清理**。
> 需要几次诊断就 `--dry-run` 几次，最后用真报告名跑一次正式的。

### ⚠️ 绝对禁止（违反将导致流程失败）

1. ❌ 不要编译代码（`py_compile`）
2. ❌ 不要 import FactorFBWorkspace
3. ❌ 不要自己加载 parquet（含手动检查 schema）
4. ❌ 不要手动 debug

### 输出全空时怎么改（`result_exists:false` 或 `all_nan_warning`）

`test-and-export` 返回全空 = **代码有明确的结构性错误**，不是参数不合适。
按顺序核对，**只改命中的那一项**，改完重跑一次：

1. **返回结构**：必须是 `{股票代码: {"因子名": 值}}`，值是标量。返回 `pd.Series` → 全空。
2. **行数过滤**：窗口天然是 `lookback+1` 行（最后一行 = T 日）。**不要**写
   `if len(df) < lookback+2: continue` 这类过滤——会把所有股票滤空。要算 `X.shift(k)` 只需 `lookback >= k`。
3. **列名**：只能用在 `show-columns` 输出里见过的名字。写错列名 → `KeyError` 或全 NaN。
4. **全 NaN 传播**：除零、`pct_change` 首行、对全 NaN 序列取均值。

**🚫 禁止用 `for lb in 1 5 20 21 ...` 扫 lookback。** `lookback` 是提取阶段给定的定义参数，
不是网格搜索出来的；全空几乎从不因 lookback 差 1 引起（未修模板时是唯一例外，现已修）。
**🚫 `--dry-run` 累计最多 3 次**，超出说明没找到结构性根因，直接按失败报告。
5. **写代码 → 跑 test-and-export，中间不做任何事**
6. **跑通后不得再改代码**：注释措辞、变量名、格式优化等一律禁止。
   只有 **test-and-export 失败**时才允许按下方规则修改重试。

### 如果 test-and-export 失败（含错误和超时）

- **普通错误**：看错误信息，修改函数代码后重新跑，最多重试 2 次
- **超时（>5 分钟没产出）**：见下方「超时熔断」
- **累计 3 次都失败** → 在结果中报告 failure，不阻塞后续因子

### 超时熔断

**第 2 步的命令已用 `timeout 300` 包住。** `exit=124` = 5 分钟没跑完，判定为**性能问题**。

> ⚠️ **Bash 工具自身默认 120 秒**就会把命令转后台。所以 `test-and-export` 这类长命令，
> 调用时**必须把 Bash 工具的 `timeout` 参数设为 300000（毫秒）**，让它在一次调用内跑完，
> 不要被工具提前转后台。

按下面改代码后**重跑一次**（仍用 `timeout 300` + Bash `timeout: 300000`）：

- **`lookback` 大的分钟因子（≥60）必须走向量化**：写 `calc_factor_series(df, stock)`，
  一次算完全部日期；**不要**让模板按日循环调用 `calc_factors_one_day`。
- 常用手段：`rolling` / `ewm` / `groupby(dates).transform` / 先按日聚合再跨日 rolling。
- **禁止 `rolling.apply(lambda)`、禁止 for 循环逐日/逐股**。

**重跑仍超时（exit=124）→ 直接报告 failure**，不要继续试。

### 🚫 超时后绝对禁止：丢后台 + 轮询

**无论命令是自己 `timeout` 超时，还是被 Bash 工具提前转后台（返回 "moved to the background"），
都绝对不允许用 `sleep` / 反复 `cat`·`tail` 输出文件来等它跑完。**

```bash
# ❌ 绝对禁止 —— 每一个这样的回合都要把整个上下文重发一遍
sleep 170; tail -35 /tmp/.../tasks/xxx.output
sleep 60;  cat  /tmp/.../tasks/xxx.output
```

**为什么**：Agent 每次工具调用，都要把「到目前为止的整段对话」重新发一遍。
`sleep` 和 `cat` **不推进任何任务**，却各产生一个完整回合——等 10 分钟就是白烧 5~10 份
完整上下文（实测有 worker 因此从 12 轮涨到 56 轮、token 翻 11 倍）。

**正确做法**：把「被转后台」或「超时」**一律当作超时熔断**处理 ——

1. **立刻**按上方熔断规则改代码（向量化），不要等、不要看后台输出；
2. 重跑一次，Bash 调用带 `timeout: 300000`，让它在单次调用内阻塞返回；
3. 仍失败 → **直接报告 failure**。

**最多看一眼**：若确实需要判断上次是否仍在跑，只允许**一次** `cat` 输出文件（不是 `sleep`+多次），
随后无论结果如何都按熔断处理。**禁止第二次轮询。**

---

## 返回格式

**返回值只有下面这一行 JSON，前后不加任何文字**（同主 agent prompt 给的那一行）。
禁止附加：完成说明、实现要点、判定理由、注意事项、代码摘要、验证过程。

成功：`{"name": "{name}", "success": true, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null}`

失败：`{"name": "{name}", "success": false, "missing_fields": false, "code_path": null, "error": "不超过 20 字的失败原因"}`
