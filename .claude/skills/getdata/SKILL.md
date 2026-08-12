---
name: getdata
description: 从本地「新建文件」目录导入新增行情/非行情数据，自动检测新列 → 更新 schema → 更新 prompt
---

# /getdata — 本地增量数据导入

用户手动把新增数据放到 `新建文件/` 目录，用本 skill 导入。
（`数据仓库/行情数据/日线/` + `数据仓库/非行情数据/` + `数据仓库/行情数据/分钟线/`）

## 三种数据格式

| 类型 | 子目录 | 格式 |
|------|--------|------|
| **日线** | `日线/dailyData.parquet` | 单文件全量日线，含 `symbol` + `date` 列 |
| **分钟** | `分钟线/YYYYMMDD.parquet` | per-date，MultiIndex[instrument, datetime] |
| **截面因子** | `非行情/因子名.parquet` | index=日期(str, yyyy-mm-dd), columns=股票代码(int), value=float64 |
| **描述** | `新建文件/基本面因子说明.csv` | CSV 无表头，每行 = 非行情/ 下一个 pqt 文件的描述 |

## 流程

**每次导入按此顺序执行：**

```
1. python3 scripts/import_new_data.py --check
   → 看 新建文件/ 下有什么新文件，预览每个文件的类型

2. python3 scripts/import_new_data.py
   → 自动检测格式并导入（日线补齐 / 分钟补齐 / 截面因子入库）
   → 脚本自动完成全部数据操作

3. 判断输出：
   ├─ 有 NEW_COLUMNS_DETECTED → 读 `新建文件/基本面因子说明.csv` 找新字段对应行的描述
   │  → 用描述更新 data/schema.json + 两个 factor_field_schema.json 的 short_name/note
   │  → python3 scripts/import_new_data.py --update-prompts-only
   │
   └─ 无 NEW_COLUMNS_DETECTED → 只是数据补齐，无新字段
      → 直接汇报结果即可（无需 agent 干预）

4. 验证：
   → 测试数据仍是固定 300 天（trade_dates.json 未变、股票数未变）
   → 抽查一个因子能跑
```

### ⚠️ 新增字段 → CSV 加一行 + 非行情/ 加一个 pqt（核心约定）

**CSV 与 `非行情/` 目录一一对应**：用户每新增一个字段/因子，会同时做两件事——
1. 在 `非行情/` 目录放一个 pqt 数据文件
2. 在 `新建文件/基本面因子说明.csv` 加一行描述

CSV 是字段含义清单，**每行对应 `非行情/` 下一个 pqt 文件**。agent 必须以此为准更新数据描述。

CSV 无表头，3 列逗号分隔：
```
文件名.parquet,因子名,描述文本
```
例如：`momentum.parquet,momentum,动量因子，过去20日累计收益`

- 第一列是数据文件名（= `非行情/` 里的 pqt 文件名，含 `.parquet` 后缀）
- 第二列是因子名（数据里的列名）
- 第三列是中文含义描述
- 也兼容 2 列格式：`因子名,描述文本`
- 编码兼容 GBK / UTF-8（Windows 下保存的 CSV 多为 GBK，脚本自动识别）

**Agent 职责**：导入出现 `NEW_COLUMNS_DETECTED` 时，在 CSV 中找该字段对应行的描述，用它更新 `factor_field_schema.json` 的 `short_name`（中文含义）和 `note`（完整说明）。

## 使用方式

```bash
# 查看 新建文件/ 状态（推荐先执行）
python3 scripts/import_new_data.py --check

# 自动检测格式并导入
python3 scripts/import_new_data.py

# 只看将要导入的内容，不执行
python3 scripts/import_new_data.py --dry-run

# 仅根据 schema.json 更新 prompt 标记块
python3 scripts/import_new_data.py --update-prompts-only
```

## 新建文件/ 目录结构

```
新建文件/
  日线/
    dailyData.parquet  # 全量日线单文件
  分钟线/
    20260806.parquet   # per-date 分钟数据
  非行情/              # 截面因子（CSV 每行对应这里一个 pqt）
    momentum.parquet   # 例如：CSV 里有 momentum.parquet,momentum,动量因子...
  基本面因子说明.csv    # 字段含义清单：每行 = 非行情/ 下一个 pqt 文件
```

- 子目录是推荐约定；放根目录也能导入（按列名自动分类行情/非行情，自动检测格式）
- **新增字段时**：`非行情/` 放一个 pqt + CSV 加一行 `文件名.parquet,因子名,描述文本`（无表头，GBK/UTF-8 均可）

## 导入规则

- **行情列** → 全量 `数据仓库/行情数据/日线/全量/stock_data/daily/{code}.parquet`
- **非行情列** → 全量 `数据仓库/非行情数据/全量/stock_data/daily/{code}.parquet`
- **分钟数据** → 复制到 `minute_by_date/` + 更新 per-stock `minute/{code}.parquet`
- **截面因子** → 转长格式 → 合并进全量非行情 per-stock parquet（新列）
- 逐股票合并（concat + 按日期去重，新数据优先，sort）
- **新列** → 自动注册 `data/schema.json` + 两个 `factor_field_schema.json`
  → 输出 `NEW_COLUMNS_DETECTED`
- 更新全量 `trade_dates.json` / `stock_list.json`（并集）；`industry.json` 不变
- 分钟数据更新分钟 `trade_dates.json` / `stock_list.json`

### ⚠️ 测试数据固定窗口（重要规则）

测试数据固定 **300 天、固定不动**（窗口不滑动）：
- 导入只把**本次新列**从全量拷进测试 per-stock parquet（行情→测试行情、非行情→测试非行情）
- 对齐到固定 300 个测试交易日，仅限现有 300 只测试股票
- **绝不**加新日期、不加新股票、不改 `trade_dates.json` / `stock_list.json`
- 分钟测试数据同样固定不动

## 关键文件

| 文件 | 说明 |
|------|------|
| `scripts/import_new_data.py` | 导入脚本（全部逻辑，含分钟/截面因子支持） |
| `新建文件/` | 用户放置新增数据的目录 |
| `新建文件/基本面因子说明.csv` | 用户维护的字段含义清单（每行 = 非行情/ 下一个 pqt 文件） |
| `data/schema.json` | 字段注册表，定义所有可用列及其来源 |
| `数据仓库/行情数据/日线/{全量,测试}/` | 行情数据（价量 20 列） |
| `数据仓库/非行情数据/{全量,测试}/` | 非行情数据（非行情列 + 新增截面因子） |
| `数据仓库/行情数据/分钟线/{全量,测试}/` | 分钟数据（per-date + per-stock） |
| `*/factor_field_schema.json` | LLM 数据可用性检查用的字段含义表，新列自动同步 |

## 新列与新数据源

`import_new_data.py` 自动完成：
1. 扫描 `新建文件/` 下的 parquet/csv，自动检测文件类型
2. 日线标准数据：按列名分类行情/非行情，按股票合并
3. 分钟数据：复制到 minute_by_date + 更新 per-stock
4. 截面因子：转长格式，按股票合并到非行情
5. 新列注册到 schema.json 和 factor_field_schema.json
6. 输出 `⚠️ NEW_COLUMNS_DETECTED: [...]`

### Agent 必做步骤

**导入完成后**，如果出现 `NEW_COLUMNS_DETECTED`：

1. 读 `新建文件/基本面因子说明.csv`，找出每个新列对应行的描述（用户维护的含义清单）
2. 更新 `data/schema.json`：将新列的 `description` 改为 CSV 里的实际含义
3. 更新两个 `factor_field_schema.json`：`short_name` 改为 CSV 里的中文含义，`note` 更新为完整说明
4. 运行 `python3 scripts/import_new_data.py --update-prompts-only` 刷新 prompt 文件

**数据说明提到了新文件夹/新文件格式时**（import_new_data.py 暂不支持的）：
- Agent 需要修改 `scripts/import_new_data.py`，新增对应的解析分支
- 参考现有 `_detect_file_type` 的格式检测逻辑

### 被更新的 prompt 文件

| 文件 | 日线 | 分钟 |
|------|------|------|
| `rdagent/components/coder/factor_coder/prompts.yaml` | ✅ DAILY_COLUMNS | ✅ MINUTE_COLUMNS |
| `.claude/skills/factor/knowledge/daily.md` | ✅ DAILY_COLUMNS | |
| `.claude/skills/factor/knowledge/cross_section.md` | ✅ DAILY_COLUMNS | |
| `.claude/skills/factor/knowledge/deep_learning.md` | ✅ DAILY_COLUMNS | |
| `.claude/skills/factor/knowledge/minute.md` | | ✅ MINUTE_COLUMNS |
| `.claude/skills/factor/knowledge/minute_cs.md` | | ✅ MINUTE_COLUMNS |

### 描述生成规则

列描述优先使用 `factor_field_schema.json` 的 `short_name` + `note`（含中文含义与备注），
fallback 到 `data/schema.json` 的 `description`。