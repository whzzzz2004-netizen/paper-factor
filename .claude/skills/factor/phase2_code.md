# Phase 2 · 第二步：写核心函数 + 跑测试（字段已确认齐全）

> **数据根目录**：先运行 `python3 scripts/claude_factor_helper.py data-root` 获取实际路径（本机 D 盘 `/mnt/d/paper-factor-data`；另一台电脑 G 盘自动识别）。下文所有 `/mnt/d/paper-factor-data` 均以该输出为准。

> **前提**：你已按 `phase2_check.md` 跑过 `show-columns`，确认字段齐全，并挑出了本因子的 `{cols}`。
> **不要重复跑 show-columns。**

### 你的因子

- 因子名: `{name}`
- 类型: `{type}`（minute/daily/cross_section/deep_learning）
- 函数名: 见下方对照表
- lookback: `{lookback}`  **（只含核心计算天数，不含论文末尾的截面标准化/std20 后处理）**
- 报告名: `{report_name}`
- 因子定义: 见主 agent prompt 给的提取命令（`formulation` / `description` / `source_excerpt`）

> ⚠️ **严格按分配的类型实现，不要私自改类型、不要改模板。**
> `{type}` 是定义阶段判定的权威类型，也是全量阶段实际使用的类型；你改不动它，
> 改了只会让测试与全量不一致。遇到"看起来装不下"的情况，按下面判断：
>
> **① 分钟因子要行业分位 / 排名 / 全市场分组？** 类型仍是 `minute`，
> 用 `minute` 模板的**可选截面钩子**——除 per-stock 函数外，额外定义
> `cross_section_transform(all_values)`（签名与示例见 `knowledge/minute.md`）。
> **不要**换成 `cross_section` 或任何"分钟截面"模板，那种模板已删除。
>
> **② 日线因子要行业分位 / 排名？** 类型应是 `cross_section`，用
> `calc_factor_cross_section(all_data, trade_date)`。若分配的是 `daily` 却确实需要同行数据，
> **不要自己改模板**——在返回值里报 failure，说明"类型与定义不符"，由上游纠正。
>
> 拿不准就往上报，不要靠翻源码猜。**去 grep/读 `rdagent/.../factor.py` 或 helper 源码
> 找答案是被硬拦截的无效探索。**

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
- **禁用截面操作**（排名/标准化/行业中性化）：只输出个股原始值。
  截面是核心逻辑时**按类型走对应入口**——`cross_section` 用 `calc_factor_cross_section`；
  `minute` 用可选的 `cross_section_transform` 钩子（见 `knowledge/minute.md`）。
  **不存在**「对产出 `.parquet` 做后处理」这条路，写了不会被调用。
- 禁止月末判断、禁止用 `len(df) < X` 做上市天数筛选
- 禁止 `transform('count')` → 用 `transform('size')`；禁止 `rolling.apply(lambda)`
- 布尔序列 `shift()` 后必须 `fillna(False)`；`np.inf` / `-np.inf` → `np.nan`
- **禁止分钟级 for 循环**（用向量化操作）
- **分钟线没有 `amount`**：成交额用 `df['volume'] * df['close']`（volume 股 × close 元）；`--cols` 只写 `volume`，不要写 `amount`
- **`INDUSTRY_DICT[股票代码]` = 申万一级行业名**，需要「行业分类」时用它（不写进 `--cols`）
- **`get_ts_index_data(symbol, data_type)` = Tushare 指数数据**：`'price'`(行情)、`'components'`(成分股)、
  `'valuation'`(PE/PB 等)、`'global'`(全球指数)，如 `get_ts_index_data('000300.SH', 'price')`；
  常用代码 `000300.SH`(沪深300)、`000905.SH`(中证500)、`000016.SH`(上证50)、`000852.SH`(中证1000)、
  `399006.SZ`(创业板指)、`932000.CSI`(中证2000)。需要指数数据时用它（不写进 `--cols`）；
  **历史回测必须传 `end_date=trade_date` 或自行截断** `idx[idx.index <= trade_date]`（成分股同理）

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

> `timeout 300` 上限 5 分钟，**必须加**（超时怎么办见文末「失败 / 超时」与 `phase2_troubleshoot.md`）。
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
  --cols "{cols}" --lookback {lookback} --type {type_key} \
  --date {DATE} --dry-run
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

> ⚠️ 这条**由 helper 强制**：同一个因子 `--dry-run` 超 3 次、或 test-and-export 累计超 6 次，
> helper 直接返回 `{"success": false, ...}` 并且**连计算都不跑**。看到这个失败就是在告诉你
> 「别再试了」——直接按 failure 报告，不要换参数重来。
5. **写代码 → 跑 test-and-export，中间不做任何事**
6. **跑通后不得再改代码**：注释措辞、变量名、格式优化等一律禁止。
   只有 **test-and-export 失败**时才允许按下方规则修改重试。

### 失败 / 超时怎么办

- **普通错误**：看错误信息改代码，最多重试 2 次（helper 侧另有总量 6 次的硬上限）。
- **累计 3 次都失败** → 在结果里报 failure，不阻塞后续因子。
- **超时（`exit=124`）或被转后台** → 这是**性能问题**，只做一件事：改成向量化
  （`calc_factor_series` 一次算完、rolling/ewm/groupby transform、禁逐日循环），
  带 Bash `timeout: 300000` 重跑**一次**；仍失败直接报 failure。

> 🚫 **超时/转后台后严禁 `sleep` / 反复 `cat`·`tail` 等它跑完** —— 等待不推进任务，
> 每个回合却要重发整段上下文（实测有 worker 因此 token 翻 11 倍）。
>
> 完整规则（熔断步骤、向量化手段、为什么）在 **`phase2_troubleshoot.md`**，
> **只在真遇到超时/转后台时才 Read，正常路径不要读。**

---

## 返回格式

**返回值只有下面这一行 JSON，前后不加任何文字**（同主 agent prompt 给的那一行）。
禁止附加：完成说明、实现要点、判定理由、注意事项、代码摘要、验证过程。

成功：`{"name": "{name}", "success": true, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null}`

失败：`{"name": "{name}", "success": false, "missing_fields": false, "code_path": null, "error": "不超过 20 字的失败原因"}`
