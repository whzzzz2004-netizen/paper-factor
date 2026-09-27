# Phase 2 · 第一步：字段核对（所有因子必读）

> 本文件只做一件事：**判断因子需要的字段在本地数据里是否齐全**。
> 判完为止 —— 不缺字段才去读 `phase2_code.md` 写代码。

### 你的因子

主 agent 的 prompt 给了你：`因子名` / `类型` / `lookback` / `报告名` / `extracted.json 路径`。

**先取因子定义**（不要读该文件里的其他因子）：

```bash
python -c "
import json; d=json.load(open('<extracted_json 路径>'))
f=[x for x in d['factors'] if x['name']=='<因子名>'][0]
print('description:', f['description']); print('formulation:', f['formulation'])
print('source_excerpt:', f['source_excerpt'])
"
```

`formulation` 里出现的每一个原始数据输入，都是你需要核对的字段。

---

## Step 0：看全部列 → 判断缺列 / 挑出真实列名

**`show-columns --type` 只有两个合法取值：**

| 你的因子类型 | 跑哪个命令 |
|---|---|
| `minute` | `--type minute` |
| `daily` / `cross_section` / `deep_learning` | `--type daily_single` |

```bash
python scripts/claude_factor_helper.py show-columns --type minute
python scripts/claude_factor_helper.py show-columns --type daily_single
```

> ⚠️ **没有** `--type cross_section` / `--type deep_learning` —— 试了只会浪费一轮往返。
> 列清单只有两套：**分钟线列** 和 **日线+非行情列**。

**以这次运行的输出为唯一字段核对依据**（含输出末尾的「额外可用数据」段落）。
数据会持续补充字段，**不要凭记忆或本文档假定有哪些列**。

> 🚫 **不要探索**——`show-columns` 的输出就是权威，翻别处（数据仓库 / helper 源码 / 其他因子实现 /
> parquet schema / memory）纯属浪费：每次工具调用 ≈20 秒完整 API 往返，而计算本身只要 20-60 秒。
> 这些操作已被 PreToolUse 钩子（`.claude/hooks/block_explore.py`、`block_datascan.py`）**硬拦截**，
> 执行会直接报错（全仓递归扫描还会读 66GB 二进制卡死 600 秒）。

### 判断缺列的标准（两条都满足才算缺列）

1. 你已经看过当次 `show-columns` 的完整输出
2. 你明确知道：想要的列**既不在清单里，也不能由清单里的列精确推导得到**

**只有三种结果：**

| 情况 | 动作 |
|---|---|
| 字段在清单里 | 直接用（列名照抄） |
| 能由清单内的列**精确**推导 | 用推导式实现 |
| 其余一切情况 | **判缺列** |

**禁止的近似（一律判缺列，不得写代码）：**
- ❌ 把缺失的列**假设成常数**（如"流通股本在窗口内不变，所以用 volume 代替换手率"）
- ❌ 用语义相近的列**代理**目标列（如用 volume 代理 turnover、用成交量代理成交额）
- ❌ 丢掉定义里的某个真实分量（如"总换手率 − 竞价换手率"只取总换手率）
- ❌ 任何让因子数值与原文定义产生偏差的处理

**允许的推导只有一种**：用清单内的列做**精确的**数学运算。判据是算出来的值与原文定义**逐点相等**，不是"大致相当"。例如 `close.pct_change()`、`close * factor`、`close / open - 1`。

### ⚠️ 三种「看着像缺字段、其实不是」的情况

**① 时间点 / 滞后 / 滚动统计 —— 不是字段。**
30 日前的价格就是 `close.shift(30)`，过去 20 日均量就是 `volume.rolling(20).mean()`。
**判缺列只看「基础列」存不存在，不看定义里出现了多少个时间点或窗口。**

**② 分组 / 排名 / 分位数 —— 不是字段。**
「按市值分大小组」「截面市值排名」都由清单内的列算出（截面运算不算缺字段）。

**③ 全市场数据面板 —— 天然具备。**
`cross_section` / `deep_learning` 模板给 `calc_factor_cross_section(all_data, trade_date)` 的
`all_data` 就是**全市场 `{股票代码: DataFrame}`**（每只股票截至 T 日的窗口切片）。

> 因此判缺列的依据**只能是「列」本身**（如 `成交额`、`自由流通市值`、指数行情），不能是
> 「历史价格」/「市值分组」/「全市场面板」这类由列或由模板推出的东西。
> 写 `missing_fields` 时逐条自检：**这一项真的是一「列」吗？** 不是就删掉。

### 额外可用数据（不算缺字段）

`show-columns` 输出末尾的「额外可用数据」段落也是合法数据源。当前只有一样：

- `INDUSTRY_DICT[股票代码]` → 申万一级行业名（如 `"银行I"`）。
  **需要「行业分类」时用它，永远不是缺列理由**；不要写进 `--cols`（它不是 parquet 列）。
  用法见各 type 的 `knowledge/*.md`。

**除 `INDUSTRY_DICT` 外的其他在线数据（指数行情、指数成分股、市场收益率等）本地不可用**，
按缺字段处理，不要用相近数据代理。

> 注意：**截面后处理**（排名/标准化/中性化）不属于缺字段，照常跳过、只输出个股原始值。但"缺某个字段"永远不能跳过或近似。

---

## 分支 A：字段齐全 → 继续读 phase2_code.md

挑出对应的真实列名（如"复权收盘价"对应 `close` × `factor`），这些就是本因子的 `{cols}`。
「行业分类」用 `INDUSTRY_DICT`（**不要**写进 `--cols`，它不是 parquet 列）。

**然后 Read `.claude/skills/factor/phase2_code.md`，按它写代码并跑测试。本文件到此结束。**

---

## 分支 B：判缺列 → 写 missing.json 后直接返回

**不要再去找**——字段清单只由当次 show-columns 输出决定，翻别处不会有。
**不要写代码，不要跑 test-and-export，不要 deploy-to-full。**

用 Write 工具写 `{name}.missing.json`（**完整路径**，目录不存在时先创建）：
`/mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report_name}/{name}/{name}.missing.json`

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

> ⚠️ `missing_fields` **只写真正不存在的基础列**（如 `成交额`、`自由流通市值`、`指数行情`）。
> 「历史价格」「30 日前收盘价」「市值分组」「全市场面板」这类一律删掉——见上「三种看着像缺字段、其实不是」。

然后直接返回。

---

## 返回格式（两种分支通用）

**返回值只有下面这一行 JSON，前后不加任何文字。**
禁止附加：完成说明、实现要点、判定理由、注意事项、代码摘要、验证过程、schema 普查结果。
需要留档的说明一律写进**代码注释**或 **missing.json**，不要放进返回值。

缺字段（分支 B）：
```
{"name": "{name}", "success": true, "missing_fields": true, "code_path": null, "error": null}
```

成功（分支 A 跑完 test-and-export 后）：
```
{"name": "{name}", "success": true, "missing_fields": false, "code_path": "/tmp/factor_{name}.py", "error": null}
```

失败：
```
{"name": "{name}", "success": false, "missing_fields": false, "code_path": null, "error": "不超过 20 字的失败原因"}
```
