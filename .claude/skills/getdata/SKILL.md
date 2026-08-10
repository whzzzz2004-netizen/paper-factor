---
name: getdata
description: 从本地「新建文件」目录导入新增行情/基本面数据，自动检测新列 → 更新 schema → 更新 prompt
---

# /getdata — 本地增量数据导入

用户手动把新增数据放到 `新建文件/` 目录，用本 skill 导入。数据仓库已重构为本地分层
（`数据仓库/行情数据/日线/` + `数据仓库/基本面数据/` + `数据仓库/行情数据/分钟线/`），
远程 SMB 同步已删除。

## 三种数据格式

| 类型 | 子目录 | 格式 |
|------|--------|------|
| **日线** | `日线/dailyData.parquet` | 单文件全量日线，含 `symbol` + `date` 列 |
| **分钟** | `分钟线/YYYYMMDD.parquet` | per-date，MultiIndex[instrument, datetime] |
| **截面因子** | `基本面/因子名.parquet` | index=日期(str, yyyy-mm-dd), columns=股票代码(int), value=float64 |
| **描述** | `描述.txt` | `因子名: 描述文本`，每行一个 |

## 流程

**每次导入按此顺序执行：**

```
1. python3 scripts/import_new_data.py --check
   → 看 新建文件/ 下有什么新文件，预览每个文件的类型

2. python3 scripts/import_new_data.py
   → 自动检测格式并导入（日线补齐 / 分钟补齐 / 截面因子入库）
   → 脚本自动完成全部数据操作

3. 判断输出：
   ├─ 有 NEW_COLUMNS_DETECTED → 读 描述.txt 理解新列含义
   │  → 更新 data/schema.json + 两个 factor_field_schema.json
   │  → python3 scripts/import_new_data.py --update-prompts-only
   │
   └─ 无 NEW_COLUMNS_DETECTED → 只是数据补齐，无新字段
      → 直接汇报结果即可（无需 agent 干预）

4. 验证：
   → 测试数据仍是固定 300 天（trade_dates.json 未变、股票数未变）
   → 抽查一个因子能跑
```

### 描述.txt 步骤说明

导入截面因子（`基本面/因子名.parquet`）时，**必须先**在 `描述.txt` 中写一行：
```
因子名: 描述文本
```
例如：`momentum: 动量因子，过去20日累计收益`

脚本自动读取 `描述.txt`，将描述文本填入 `factor_field_schema` 的 `short_name` 和 `note`。
找不到描述时 fallback 到「用户描述.txt 未提供说明」。

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
  README.md        # 用途 + 格式约定
  描述.txt          # 新因子描述：因子名: 描述文本
  日线/
    dailyData.parquet  # 全量日线单文件
  分钟线/
    20260806.parquet   # per-date 分钟数据
  基本面/
    momentum.parquet   # 截面因子
```

- 子目录是推荐约定；放根目录也能导入（按列名自动分类行情/基本面，自动检测格式）
- 描述.txt 以 `#` 开头的行为注释，空行跳过

## 导入规则

- **行情列** → 全量 `数据仓库/行情数据/日线/全量/stock_data/daily/{code}.parquet`
- **基本面列** → 全量 `数据仓库/基本面数据/全量/stock_data/daily/{code}.parquet`
- **分钟数据** → 复制到 `minute_by_date/` + 更新 per-stock `minute/{code}.parquet`
- **截面因子** → 转长格式 → 合并进全量基本面 per-stock parquet（新列）
- 逐股票合并（concat + 按日期去重，新数据优先，sort）
- **新列** → 自动注册 `data/schema.json` + 两个 `factor_field_schema.json`
  → 输出 `NEW_COLUMNS_DETECTED`
- 更新全量 `trade_dates.json` / `stock_list.json`（并集）；`industry.json` 不变
- 分钟数据更新分钟 `trade_dates.json` / `stock_list.json`

### ⚠️ 测试数据固定窗口（重要规则）

测试数据固定 **300 天、固定不动**（窗口不滑动）：
- 导入只把**本次新列**从全量拷进测试 per-stock parquet（行情→测试行情、基本面→测试基本面）
- 对齐到固定 300 个测试交易日，仅限现有 300 只测试股票
- **绝不**加新日期、不加新股票、不改 `trade_dates.json` / `stock_list.json`
- 分钟测试数据同样固定不动

## 关键文件

| 文件 | 说明 |
|------|------|
| `scripts/import_new_data.py` | 导入脚本（全部逻辑，含分钟/截面因子支持） |
| `新建文件/` | 用户放置新增数据的目录 |
| `新建文件/描述.txt` | 新因子描述（截面因子导入时必填） |
| `data/schema.json` | 字段注册表，定义所有可用列及其来源 |
| `数据仓库/行情数据/日线/{全量,测试}/` | 行情数据（价量 20 列） |
| `数据仓库/基本面数据/{全量,测试}/` | 基本面数据（18 列 + 新增截面因子） |
| `数据仓库/行情数据/分钟线/{全量,测试}/` | 分钟数据（per-date + per-stock） |
| `*/factor_field_schema.json` | LLM 数据可用性检查用的字段含义表，新列自动同步 |

## 新列与新数据源

`import_new_data.py` 自动完成：
1. 扫描 `新建文件/` 下的 parquet/csv，自动检测文件类型
2. 日线标准数据：按列名分类行情/基本面，按股票合并
3. 分钟数据：复制到 minute_by_date + 更新 per-stock
4. 截面因子：转长格式，按股票合并到基本面
5. 新列注册到 schema.json 和 factor_field_schema.json
6. 输出 `⚠️ NEW_COLUMNS_DETECTED: [...]`

### Agent 必做步骤

**导入完成后**，如果出现 `NEW_COLUMNS_DETECTED`：

1. 读 `新建文件/描述.txt`，理解每个新列的实际含义
2. 更新 `data/schema.json`：将新列的 `description` 改为实际含义
3. 更新两个 `factor_field_schema.json`：`short_name` 改为中文含义，`note` 更新为完整说明
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