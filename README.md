# paper-factor

LLM 驱动的量化因子挖掘系统。从研报 PDF / 网站文章中提取因子定义 → 自动编码测试 → 全量计算 → 每日增量更新 → 远程同步。

> **所有产出强制使用日期子目录**。新因子写入 `literature_reports/{YYYYMMDD}/` 和 `文献因子_全量/{YYYYMMDD}/`，`deploy-to-full` 和 `run_all` 不传 `--date` 时默认用当天日期。根目录下的旧因子仍被支持（`daily_update.py` 可扫描），但新因子不会写入根目录。

---

## 整体流程

```
研报PDF / 网站 / 想法文本
        │
        ▼  [Phase 1: 提取+定义]
  提取因子定义（5个agent并行）
  输出: extracted_reports/{DATE}/{report}/factor_definitions.json
        │
        ▼  [Phase 2: 编码+测试]
  每个因子一个agent，写核心函数 → test-and-export
  输出: literature_reports/{DATE}/{report}/{factor}/
          ├── {factor}.code.py       # 自包含代码（模板+用户函数）
          ├── {factor}.parquet       # 测试结果（300只×600天）
          └── {factor}.meta.json     # 元数据
        │
        ▼  [deploy-to-full]
  原样复制 .code.py，接入全量数据目录
  输出: 文献因子_全量/{DATE}/{report}/{factor}/
          ├── {factor}.code.py
          ├── {factor}.parquet       # 全量结果（5435只×全历史）
          ├── {factor}.decile.png    # 十分组收益图
          └── {factor}.meta.json     # 含IC/IR/Barra等评估指标
        │
        ▼  [sync-full]
  同步到远程 E 盘: \\192.168.1.13\E\paper_factors\文献因子_全量\
        │
        ▼  [每日增量更新]
  daily_update.py 自动检测新交易日 → 增量计算 → 合并
```

---

## 1. 数据目录：git_ignore_folder/

所有大文件、产出数据都在 `git_ignore_folder/` 下，该目录被 `.gitignore` 排除。

```
git_ignore_folder/
│
├── factor_implementation_source_data/          # 全量数据源（5435只股票）
│   ├── stock_data/
│   │   ├── daily/{code}.parquet                # 日线 per-stock（5435个文件）
│   │   │                                        # 列: open, close, high, low, volume, pct_chg,
│   │   │                                        #      turnover_rate, market_cap, pe_ttm, pb,
│   │   │                                        #      roe, roa, revenue_yoy, profit_yoy... 120+列
│   │   ├── minute/{code}.parquet               # 分钟 per-stock（5435个文件）
│   │   │                                        # 列: open, close, high, low, volume, vwap, return
│   │   ├── minute_by_date/{YYYY-MM-DD}.parquet # 分钟 by date（2029个文件）
│   │   │                                        # 列: datetime, instrument, open, close, high, low,
│   │   │                                        #      volume, vwap, factor, return
│   │   ├── stock_list.json                     # 全量股票列表（5435只）
│   │   ├── trade_dates.json                    # 交易日列表（~2027天）
│   │   └── industry.json                       # 申万一级行业分类
│   ├── daily_pv.h5                             # 日线 H5 格式（原始源数据）
│   ├── minute_pv.h5                            # 分钟 H5 格式（原始源数据）
│   ├── limit_up_daily.parquet                  # 涨停列表（每日涨停股票）
│   ├── factor_field_schema.json                # 字段注册表（所有可用列的定义）
│   └── data_field_dictionary.md                # 字段说明文档
│
├── factor_implementation_source_data_1000/     # 测试数据子集（300只×600天）
│   └── stock_data/                             # 同全量结构，只有300只
│       ├── daily/{code}.parquet
│       ├── minute/{code}.parquet
│       ├── minute_by_date/{YYYY-MM-DD}.parquet
│       ├── stock_list.json
│       └── trade_dates.json
│
├── factor_implementation_source_data_debug/    # 调试数据（1只股票）
│
├── factor_outputs/
│   ├── literature_reports/{DATE}/              # 测试因子产出
│   │   └── <报告名>/<因子名>/
│   │       ├── <因子名>.code.py                # 自包含因子代码（含完整实例化模板）
│   │       ├── <因子名>.parquet                # 测试结果（300只×600天）
│   │       └── <因子名>.meta.json              # 元数据（description/formulation等）
│   │
│   ├── 文献因子_全量/{DATE}/                   # 全量因子产出
│   │   └── <报告名>/<因子名>/
│   │       ├── <因子名>.code.py                # = literature_reports 的 .code.py（只改数据路径）
│   │       ├── <因子名>.parquet                # 全量结果（5435只×全历史，~80MB/因子）
│   │       ├── <因子名>.decile.png             # 十分组收益图
│   │       └── <因子名>.meta.json              # 含IC/IR/Barra/LLM审查
│   │
│   └── 文献因子_每日更新/                       # 每日增量更新产出
│       └── <报告名>/<因子名>/
│           ├── <因子名>.parquet                # 增量 parquet（随每日更新追加）
│           └── <因子名>.meta.json              # 状态元数据
│
├── barra_model/                                # Barra风险模型数据
│   ├── 因子收益率表(Long-Term Model).csv
│   ├── 因子收益率表(Trading Model).csv
│   ├── 因子暴露表(Long-Term Model).csv
│   ├── 特质收益率表(Trading Model).csv
│   └── 风险因子协方差矩阵表(...).csv
│
├── logs/                                       # 日志文件
│   └── daily_update.log
│
├── daily_update_config.json                    # 每日更新配置（历史记录）
└── daily_update_status.json                    # 每日更新运行状态
```

### 数据目录选择策略

脚本启动时按优先级自动检测可用数据目录：

```
1. FACTOR_DATA_DIR 环境变量
2. RDAGENT_FACTOR_DATA_DIR 环境变量
3. git_ignore_folder/factor_implementation_source_data/
4. /mnt/remote_e/_paper_factor_unified/...（CIFS挂载）
5. E:\\... / Z:\\...（Windows路径）
6. \\\\192.168.1.13\\...（SMB UNC路径）
```

检测到 `stock_data/daily/` 存在即确认。所有脚本使用统一的 `_detect_data_dir()` 逻辑。

### 涨停列表

`limit_up_daily.parquet` 由 `sync_data.py` 自动生成，列：`datetime, instrument`。生成规则：`pct_chg >= 9.5%`。每日更新前自动同步。

---

## 2. /factor 技能：两阶段架构

### Phase 1：提取 + 定义（每个内容一个 sub-agent）

最多 **5 个 agent 并行**，每个只做：

1. 读原文（PDF提取/网页抓取/直接使用文本）
2. 调用 `show-columns` 查看可用数据列
3. 定义最多 15 个因子（name, type, lookback, cols, formulation, description）
4. `save-extracted` 保存到 `extracted_reports/{DATE}/`
5. 返回因子列表给主 Claude

因子类型对照：

| type | 含义 | 模板 | 核心函数 |
|------|------|------|----------|
| `daily` | 日线逐股票 | DAILY_FRAMEWORK_TEMPLATE | `calc_factor_single_stock(df, trade_date, stock)` |
| `minute` | 分钟逐股票 | MINUTE_FRAMEWORK_TEMPLATE | `calc_factors_one_day(df, stock)` |
| `cross_section` | 日线截面 | CROSS_SECTION_FRAMEWORK_TEMPLATE | `calc_factor_cross_section(trade_date)` |
| `minute_cs` | 分钟截面 | MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE | `calc_factor_minute_raw(df, stock)` + `cross_section_transform(all_values)` |
| `deep_learning` | 深度学习 | DEEP_LEARNING_FRAMEWORK_TEMPLATE | `train_model(all_data, trade_date)` + `predict_batch(model, data_dict, trade_date)` |

### Phase 2：编码 + 测试（每个因子一个 sub-agent）

最多 **5 个 agent 并行**，每个只做：

1. 写核心函数（只实现计算逻辑，不写模板框架）
2. 调 `test-and-export` 命令（包装模板 → 跑300只测试数据 → 导出到 literature_reports）
3. 失败自动重试最多 3 次（普通错误改代码重试，超时降维优化后重试）

**test-and-export 内部流程：**

```
用户核心函数（如 /tmp/factor_xxx.py）
        │
        ▼
  FactorFBWorkspace._build_factor_code(template, user_code, lookback, cols)
        │  1. 模板缓存命中？→ 只替换 {user_code}
        │  2. 缓存未命中？→ 编译模板（列定义+lookback+双花括号解义）→ 写入 L1+L2 缓存
        ▼
  完整 .code.py（自包含，含数据加载、并行、涨停剔除）
        │
        ▼
  子进程执行（300只测试数据）
        │
        ▼
  result.parquet → {factor_name}.parquet → 复制到 literature_reports/{DATE}/{report}/{factor}/
```

### 模板缓存（2026-07-29 优化）

`_build_factor_code` 使用双层缓存避免重复编译：

| 层 | 存储 | 跨进程 | 命中速度 |
|----|------|--------|---------|
| L1 | `_TEMPLATE_CACHE` dict（内存） | 否 | 纳秒级 |
| L2 | `_template_cache/{sha256}.pickle`（磁盘） | **是** | 微秒级 |

key = `sha256(template全文 + lookback + cols_def)`，同一进程内的 RDAgent pipeline 迭代走 L1，子进程执行走 L2。每次 `test-and-export` 只做一次 `.replace('{user_code}', code)`。

### Step 3：部署 + 同步

```bash
# 部署到全量
python scripts/claude_factor_helper.py deploy-to-full \
  --code literature_reports/{DATE}/{report}/{factor}/{factor}.code.py \
  --date {DATE}

# 同步到远程
python scripts/claude_factor_helper.py sync-full --all --date {DATE}

# 标记完成
python scripts/claude_factor_helper.py mark-done --name "文件名.pdf"
```

---

## 3. 全量计算

### 单因子

`run_factor_full.py` 将测试通过的 `.code.py` 接入全量数据运行，包含完整评估流程：

```
                               ┌─ evaluate_factor.py（IC/IR/分组收益）
   .code.py → 全量计算 ────────┼─ plot_decile.py（十分组图）
  （5435只）                   ├─ barra_evaluate.py（Barra暴露）
                               └─ llm_review_factor.py（逻辑审查）
```

输出到 `文献因子_全量/{report}/{factor}/`。

### 批量

`run_all.py` 扫描 `文献因子_全量/` 下所有因子，按状态处理：

```
扫描因子
  ├─ 无 .parquet            → 全量计算（调用 factor_full_pipeline.run_full_pipeline）
  ├─ 有 .parquet 但日期落后  → 增量计算（只跑新日期，merge 回全量 parquet）
  └─ 已最新                  → 跳过
```

并行模式 `--workers 3`，本地模式默认跳过远程挂载和数据同步，`--remote` 启用。

---

## 4. 每日增量更新

`daily_update.py` 是日常运行入口，由 cron 定时触发（`scripts/setup_cron.sh` 管理）。

### 流程

```
1. 读增量 parquet → last_date（首次：从全量复制原始 parquet 作为起点）
2. 读 trade_dates.json → latest_date
3. 若 latest_date <= last_date → 跳过（已最新）
4. 复制 .code.py → 注入增量 patch
5. 设 FACTOR_INCREMENTAL_START_DATE → 子进程执行
6. 裁掉 lookback 重叠行（date > last_date）→ concat 到增量 parquet
7. 评估 + 绘图 + 同步远程
```

### 增量 patch 机制

`.code.py` 中已有完整的 TRADE_DATES 列表。增量运行时设环境变量 `FACTOR_INCREMENTAL_START_DATE`，在模板的 TRADE_DATES 定义后注入 patch：

```python
_INC_START = os.environ.get("FACTOR_INCREMENTAL_START_DATE")
if _INC_START:
    _pos = max(0, np.searchsorted(TRADE_DATES, _INC_START) - LOOKBACK_DAYS)
    TRADE_DATES = TRADE_DATES[_pos:]
```

即：保留 `lookback_days` 天的历史数据用于计算，裁掉更早的数据。结果合并时再裁掉这 `lookback_days` 的重叠行。

### cron 配置

```bash
scripts/setup_cron.sh status      # 查看状态
scripts/setup_cron.sh enable HH:MM # 启用（如 enable 16:30）
scripts/setup_cron.sh disable     # 禁用
```

执行链：`sync_data.py`（同步数据 + 生成涨停列表）→ `daily_update.py --workers 3`

---

## 5. 模板系统

5 种模板定义在 `rdagent/components/coder/factor_coder/factor.py` 的 `FactorFBWorkspace` 类中：

| 模板常量 | 并行策略 | 共享方式 | 说明 |
|----------|----------|----------|------|
| `DAILY_FRAMEWORK_TEMPLATE` | `ThreadPoolExecutor` | 主线程加载 per-stock，只读共享 | 每只股票调用一次函数 |
| `MINUTE_FRAMEWORK_TEMPLATE` | `ThreadPoolExecutor` | 主线程加载 `_WDATA`，线程共享 | 每分钟文件加载最近LOOKBACK_DAYS个文件 |
| `CROSS_SECTION_FRAMEWORK_TEMPLATE` | `ProcessPoolExecutor(loky)` | 各进程 lazy-load | 每chunk重建pool，自动释放RSS |
| `MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE` | `joblib(threading)` | 各线程独立加载 `minute_by_date` | 不支持滑动窗口（每worker独立加载） |
| `DEEP_LEARNING_FRAMEWORK_TEMPLATE` | `ThreadPoolExecutor` | 主线程加载 | 训练+预测分离 |

所有模板注入用 `.replace()` 而非 `.format()`，避免 `{xxx}` 冲突。

### 输出文件名

所有模板使用相同模式：`Path(__file__).stem.removesuffix('.code').parquet`。当代码文件为 `MorningVolumeRatio.py` 时输出 `MorningVolumeRatio.parquet`。全流程统一以 `{factor_name}.parquet` 命名。

### 涨停剔除

如果 `limit_up_daily.parquet` 存在，所有模板在保存前自动剔除涨停日数据。剔除逻辑：遍历涨停列表每个日期，对 `pct_chg >= 9.5%` 的股票将其在该日期的因子值设为 NaN。

---

## 6. 数据同步

### sync_data.py

从远程 E 盘（`192.168.1.13`）同步原始数据，流程：

```
1. smbclient 拉取 market_daily_daily_new（日线CSV）
2. 解析为 per-stock parquet
3. smbclient 拉取 market_minute_daily_new（分钟CSV）
4. 解析为 per-stock + by-date parquet
5. 裁剪 300只×600天 测试子集
6. 注册新列到 factor_field_schema.json
7. 更新 data_field_dictionary.md
8. 生成 limit_up_daily.parquet（涨停列表）
```

首次全量 `--full`，日常增量直接运行（根据文件时间戳增量拉取）。

---

## 7. 远程存储

| 路径 | 内容 |
|------|------|
| `\\192.168.1.13\E\paper_factors\文献因子_全量\` | 全量因子产出 |
| `\\192.168.1.13\E\paper_factors\文献因子_每日更新\` | 每日更新产出 |
| `\\192.168.1.13\E\_paper_factor_unified\factor_implementation_source_data\` | 全量数据源 |

同步方式：
- CIFS 挂载：`/mnt/remote_e`（自动挂载，多版本协商）
- sshfs：`scripts/sync_utils.py` 备用
- SMB 直连：`smbclient`（数据同步用）

---

## 8. CLI 命令速查

### 因子处理

```bash
python scripts/claude_factor_helper.py scan-pending              # 扫描未处理内容
python scripts/claude_factor_helper.py test-and-export ...       # 测试+导出
python scripts/claude_factor_helper.py deploy-to-full ...        # 部署全量
python scripts/claude_factor_helper.py sync-full --all --date YYYYMMDD  # 同步远程
python scripts/claude_factor_helper.py trigger-full ...          # 全量流水线
python scripts/claude_factor_helper.py run-full ...              # 单因子全量
python scripts/claude_factor_helper.py mark-done ...             # 标记完成
```

### 批量运行

```bash
python scripts/run_all.py                                        # 本地批量，默认当天日期子目录
python scripts/run_all.py --remote                               # 远程模式（挂载+同步+计算）
python scripts/run_all.py 20260727                              # 指定日期子目录
python scripts/run_all.py --workers 3                            # 3因子并行
python scripts/run_all.py --force                                # 强制重跑
python scripts/run_all.py --dry-run                              # 仅查看计划
```

### 每日更新

```bash
python scripts/daily_update.py                                   # 更新所有因子
python scripts/daily_update.py --factor report/factor            # 单因子
python scripts/daily_update.py --dry-run                         # 仅检查
python scripts/daily_update.py --skip-eval                       # 跳过评估
python scripts/daily_update.py --workers 5                       # 并行数
```

### 数据同步

```bash
python scripts/sync_data.py           # 增量同步
python scripts/sync_data.py --full    # 全量同步
python scripts/sync_data.py --check   # 仅检查
```

### Claude Code 技能

```
/factor     — 因子提取全流程（扫描→提取→编码→测试→部署→同步）
/getdata    — 增量数据同步（SMB直连+自动检测新列）
/clean      — 删除所有因子产出+Python缓存
```

---

## 9. 项目目录结构

```
paper-factor/
├── papers/
│   ├── inbox/                    # 放入待处理的研报 PDF
│   ├── website/
│   │   └── sources.json          # 网站文章 URL 列表
│   └── ideas/
│       └── ideas.json            # 文本因子想法
├── scripts/
│   ├── claude_factor_helper.py   # 核心 CLI 工具（所有因子操作命令）
│   ├── factor_utils.py           # 共享工具函数（数据加载、子进程、合并）
│   ├── run_all.py                # 全量/增量批量运行
│   ├── run_factor_full.py        # 单因子全量流水线
│   ├── daily_update.py           # 每日增量更新
│   ├── sync_data.py              # 数据同步（SMB 拉取原始数据）
│   ├── sync_utils.py             # 远程同步工具（CIFS/sshfs）
│   ├── evaluate_factor.py        # 因子评估（IC/IR/分组收益）
│   ├── plot_decile.py            # 十分组收益图绘制
│   ├── barra_evaluate.py         # Barra 风险模型分析
│   └── llm_review_factor.py      # LLM 逻辑审查
├── rdagent/
│   └── components/coder/factor_coder/
│       └── factor.py             # 5种模板 + FactorFBWorkspace + 模板缓存
├── rdagent/app/qlib_rd_loop/
│   └── factor_full_pipeline.py   # 全量流水线执行器
├── .claude/
│   └── skills/factor/
│       ├── SKILL.md              # /factor 技能定义（两阶段架构）
│       └── knowledge/            # 领域知识（涨跌停规则、列定义等）
├── .claude/skills/factor/knowledge/
│   ├── minute.md                 # 分钟线知识
│   ├── minute_cs.md              # 分钟截面知识
│   └── zhishu.md                 # 指数数据知识
├── git_ignore_folder/            # 所有大数据文件（见上节）
└── scripts/setup_cron.sh         # cron 定时任务管理
```
