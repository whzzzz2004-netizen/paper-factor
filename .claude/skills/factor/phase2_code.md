# Phase 2 · 第二步：写核心函数 + 跑测试（字段已确认齐全）

> **进入本文件的前提**：你已按 `phase2_check.md` 跑过 `show-columns`，确认字段齐全，
> 并已挑出本因子的 `{cols}` 真实列名。**不要重复跑 show-columns。**

### 你的因子

- 因子名: `{name}`
- 类型: `{type}`（minute/daily/cross_section/deep_learning）
- 函数名: 见下方对照表
- lookback: `{lookback}`  **（⚠️ 只含核心计算天数，不含论文末尾的截面标准化/std20后处理）**
- 报告名: `{report_name}`
- 因子定义: 见主 agent prompt 给的提取命令（`formulation` / `description` / `source_excerpt`）

---

## 1. 写核心函数到 /tmp/factor_{name}.py

**写代码前，用 Read 工具读**恰好一个**知识文件——只读与你自己 type 对应的那一个（不要读其他类型，省 token）：**

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
- **字段只认 `show-columns` 当次输出**：有 → 用；无但能精确推导 → 推导着用；其余判缺列
- **`INDUSTRY_DICT[股票代码]` = 申万一级行业名**，需要「行业分类」时用它、不算缺字段（不写进 `--cols`）。**其他在线数据（指数行情/成分股/市场收益率）本地不可用**，按缺字段处理
- **lookback 只含核心计算天数**，不含论文末尾的截面标准化/std20/取波动率等后处理

---

## 2. 立即跑 test-and-export + deploy-to-full（写完后立刻执行，不停顿）

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
> `{cols}` = 你在 phase2_check.md 里从 show-columns 挑出的**真实列名**（空格或逗号分隔均可）。

> ⚠️ `--description` / `--formulation` / `--source-excerpt` 从提取命令的输出里取；
> 含特殊字符（引号、换行、`$`）时用单引号包裹或写入临时文件传参。

**test-and-export 成功后，立即部署到全量：**
> 路径结构：test-and-export 输出为 `因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py`。
> deploy-to-full 的 `--code` 用同一路径，**不要**多套一层 `{name}` 目录。
```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.code.py \
  --date {DATE}
```

### ⚠️ 绝对禁止（违反将导致流程失败）

1. ❌ 不要编译代码（`py_compile`）
2. ❌ 不要 import FactorFBWorkspace
3. ❌ 不要自己加载 parquet（含手动检查 schema）
4. ❌ 不要手动 debug
5. **写代码 → 跑 test-and-export，中间不做任何事**
6. **跑通后不得再改代码**：注释措辞、变量名、格式优化等一律禁止（改了=要重跑，纯浪费）。
   只有 **test-and-export 失败**时才允许按下方规则修改重试。

### 如果 test-and-export 失败（含错误和超时）

- **普通错误**：看错误信息，修改函数代码后重新跑，最多重试 2 次
- **超时**（超过 300s 无结果）：修改代码优化性能（减天数、向量化等）后重试，最多 **2 次修改机会**
- **累计 3 次都失败** → 在结果中报告 failure，不阻塞后续因子

---

## 返回格式

**返回值只有下面这一行 JSON，前后不加任何文字。**
禁止附加：完成说明、实现要点、判定理由、注意事项、代码摘要、验证过程。

成功：
```
{"name": "{name}", "success": true, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null}
```

失败：
```
{"name": "{name}", "success": false, "missing_fields": false, "code_path": null, "error": "不超过 20 字的失败原因"}
```
