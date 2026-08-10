# paper-factor

LLM 驱动的量化因子挖掘系统。从研报 PDF / 网站文章中提取因子定义 → 自动编码测试 → 全量计算 → 每日增量更新。

**本地执行，不依赖 Docker，不依赖远程存储。**

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
  输出: 因子产出/测试/{DATE}/{report}/{factor}/
          ├── {factor}.code.py       # 自包含代码（模板+用户函数）
          ├── {factor}.parquet       # 测试结果（300只×300天）
          └── {factor}.meta.json     # 元数据
        │
        ▼  [deploy-to-full]
  原样复制 .code.py，接入全量数据目录
  输出: 因子产出/全量/{DATE}/{report}/{factor}/
          ├── {factor}.code.py
          ├── {factor}.parquet       # 全量结果（5435只×全历史）
          ├── {factor}.decile.png    # 十分组收益图
          └── {factor}.meta.json     # 含IC/IR/Barra等评估指标
        │
        ▼  [每日增量更新]
  run_all.py 自动检测新交易日 → 增量计算 → 合并
```

---

## 快速开始（老板电脑）

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url> paper-factor
cd paper-factor

# 安装依赖
pip install -e .

# 或使用 requirements.txt
pip install -r requirements.txt
```

### 2. 配置 `.env`

复制 `.env.example` 为 `.env`，填入必要信息：

```bash
# ── LLM API（因子提取/审查用） ──
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1
CHAT_MODEL=openai/gpt-4o

# ── JQData（聚宽数据，分钟因子需要） ──
JQDATA_USERNAME=your_username
JQDATA_PASSWORD=your_password

# ── 因子执行配置 ──
FACTOR_CoSTEER_EXECUTION_BACKEND=local
```

### 3. 准备数据

数据仓库是本地目录结构，需要从原有环境拷贝或通过 `/getdata` 逐步导入：

```
数据仓库/
├── 行情数据/
│   ├── 日线/
│   │   ├── 全量/stock_data/daily/{code}.parquet   # 5435只 × 价量20列
│   │   └── 测试/stock_data/daily/{code}.parquet   # 300只 × 300天
│   └── 分钟线/
│       ├── 全量/stock_data/
│       │   ├── minute/{code}.parquet              # 5435只
│       │   └── minute_by_date/{date}.parquet      # per-date
│       └── 测试/stock_data/                        # 同结构，300只
├── 基本面数据/
│   ├── 全量/stock_data/daily/{code}.parquet       # 5435只 × 18列
│   └── 测试/stock_data/daily/{code}.parquet       # 300只 × 300天
└── 板块数据/                                       # 行业分类等
```

**最简单的方式**：从已有环境直接拷贝整个 `数据仓库/` 目录。或者：

```bash
# 通过 /getdata 导入（每次把新数据放到 新建文件/ 下）
python scripts/import_new_data.py --check    # 预览
python scripts/import_new_data.py            # 执行导入
```

---

## 数据目录结构

```
paper-factor/
├── 数据仓库/                          # 所有数据（本地，不入git）
│   ├── 行情数据/日线/{全量,测试}/      # 日线价量数据（20列）
│   ├── 行情数据/分钟线/{全量,测试}/    # 分钟数据
│   ├── 基本面数据/{全量,测试}/        # 基本面数据（18列）
│   ├── 板块数据/                      # 行业分类
│   ├── barra_model/                   # Barra风险模型数据
│   └── 因子产出/                      # 所有因子产出
│       ├── 测试/{DATE}/               # 测试因子产出
│       └── 全量/{DATE}/               # 全量因子产出
├── 新建文件/                          # 新增数据入口（用户放置）
│   ├── 日线/dailyData.parquet
│   ├── 分钟线/YYYYMMDD.parquet
│   ├── 基本面/因子名.parquet
│   ├── barra/                         # Barra模型数据（可选，更新时放）
│   └── 描述.txt
├── git_ignore_folder/                  # 配置和中间产物（不入git）
│   └── logs/                           # 运行日志
├── scripts/                            # 所有脚本
│   ├── claude_factor_helper.py         # 核心CLI（因子操作命令）
│   ├── factor_utils.py                 # 共享工具函数
│   ├── import_new_data.py              # 数据导入（/getdata后端）
│   ├── run_all.py                      # 全量/增量批量运行
│   ├── run_factor_full.py              # 单因子全量流水线
│   ├── evaluate_factor.py              # 因子评估（IC/IR）
│   ├── plot_decile.py                  # 十分组收益图
│   ├── barra_evaluate.py               # Barra风险分析
│   └── llm_review_factor.py            # LLM逻辑审查
├── rdagent/                            # 核心引擎
│   └── components/coder/factor_coder/
│       ├── factor.py                   # 5种模板 + FactorFBWorkspace
│       ├── config.py                   # 因子执行配置
│       └── prompts.yaml                # LLM提示词（含数据列定义）
├── data/
│   └── schema.json                     # 字段注册表
├── .env                                # 本地配置（LLM密钥等）
├── .env.example                        # 配置模板
├── pyproject.toml                      # 项目依赖
└── start.sh                            # 环境检查脚本
```

### 数据目录选择策略

脚本启动时自动检测数据目录，优先级：

```
1. FACTOR_DATA_DIR 环境变量
2. RDAGENT_FACTOR_DATA_DIR 环境变量
3. 数据仓库/行情数据/日线/全量/
```

---

## 工作流详解

### 1. 数据导入（/getdata）

```bash
# 查看 新建文件/ 下有什么新文件
python scripts/import_new_data.py --check

# 执行导入（自动分类行情/基本面/分钟数据）
python scripts/import_new_data.py

# 预览不执行
python scripts/import_new_data.py --dry-run
```

导入规则：
- **行情列**（价量20列）→ `数据仓库/行情数据/日线/全量/`
- **基本面列**（18列）→ `数据仓库/基本面数据/全量/`
- **分钟数据** → `数据仓库/行情数据/分钟线/全量/`
- **新列**自动注册到 `data/schema.json` + prompt 更新
- 测试数据固定 300 天，只补新列不补新日期/新股票

### 2. 因子提取 + 测试（/factor 技能）

在 Claude Code 中执行 `/factor`，两阶段：

**Phase 1：提取** — 并行读取研报 → 定义因子（name, type, lookback, formulation）
**Phase 2：编码** — 每个因子写核心函数 → `test-and-export`（300只测试数据验证）

输出到 `数据仓库/因子产出/测试/{DATE}/`

### 3. 部署到全量

```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code 数据仓库/因子产出/测试/{DATE}/{report}/{factor}/{factor}.code.py \
  --date {DATE}
```

原样复制 `.code.py` 到 `因子产出/全量/{DATE}/`，只改数据路径。

### 4. 全量计算

**单因子：**

```bash
python scripts/run_factor_full.py \
  数据仓库/因子产出/全量/{DATE}/{report}/{factor}/{factor}.code.py
```

含评估 + 十分组图 + Barra + LLM审查。

**批量：**

```bash
# 扫描所有因子，自动全量/增量
python scripts/run_all.py

# 指定并行度
python scripts/run_all.py --workers 3

# 强制重跑
python scripts/run_all.py --force

# 仅预览
python scripts/run_all.py --dry-run
```

### 5. 每日增量更新

```bash
# 更新所有因子（全量+增量，自动检测新交易日）
python scripts/run_all.py

# 指定日期子目录
python scripts/run_all.py 2026-08-09

# 并行
python scripts/run_all.py --workers 3

# 仅查看计划
python scripts/run_all.py --dry-run
```

---

## 因子类型与模板

| 类型 | 模板 | 核心函数 | 并行策略 |
|------|------|----------|----------|
| `daily` | DAILY_FRAMEWORK_TEMPLATE | `calc_factor_single_stock(df, trade_date, stock)` | ThreadPoolExecutor |
| `minute` | MINUTE_FRAMEWORK_TEMPLATE | `calc_factors_one_day(df, stock)` | ThreadPoolExecutor |
| `cross_section` | CROSS_SECTION_FRAMEWORK_TEMPLATE | `calc_factor_cross_section(trade_date)` | ProcessPoolExecutor(loky) |
| `minute_cs` | MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE | `calc_factor_minute_raw` + `cross_section_transform` | joblib(threading) |
| `deep_learning` | DEEP_LEARNING_FRAMEWORK_TEMPLATE | `train_model()` + `predict()` | ThreadPoolExecutor |

所有模板在 `rdagent/components/coder/factor_coder/factor.py` 的 `FactorFBWorkspace` 类中。

---

## CLI 命令速查

### 因子处理

```bash
python scripts/claude_factor_helper.py scan-pending                  # 扫描未处理内容
python scripts/claude_factor_helper.py test-and-export ...           # 测试+导出
python scripts/claude_factor_helper.py deploy-to-full ...            # 部署全量
python scripts/claude_factor_helper.py mark-done --name "文件名.pdf"  # 标记完成
```

### 批量运行

```bash
python scripts/run_all.py                           # 默认当天日期子目录
python scripts/run_all.py 2026-08-09                # 指定日期子目录
python scripts/run_all.py --workers 3               # 3因子并行
python scripts/run_all.py --force                   # 强制重跑
python scripts/run_all.py --dry-run                 # 仅查看计划
```

### 数据导入

```bash
python scripts/import_new_data.py --check           # 预览新建文件
python scripts/import_new_data.py                   # 执行导入
python scripts/import_new_data.py --dry-run         # 预览不执行
```

### Claude Code 技能

```
/factor     — 因子提取全流程（扫描→提取→编码→测试→部署→标记完成）
/getdata    — 从 新建文件/ 导入新增数据（行情/基本面/分钟线）
/clean      — 删除所有因子产出 + Python 缓存
```

---

## 依赖说明

核心依赖（`pip install -e .` 自动安装）：

- LLM 调用：`litellm`, `openai`
- 数据处理：`pandas`, `numpy`, `pyarrow`, `scikit-learn`
- 聚宽数据：`jqdatasdk`
- 工具：`filelock`, `python-dotenv`, `rich`, `tqdm`, `loguru`
- PDF 解析：`pymupdf`

**不依赖 Docker。** 执行后端为 `local`，直接在宿主机运行。

---

## 重要规则

1. **改模板后必须先跑测试数据验证**，并清 pickle 缓存
2. **测试数据固定 300 天、固定不动**，导入只补新列不加新日期
3. **标准化只能是截面 zscore**，绝不能是时序标准化（未来数据泄露）
4. `.code.py` 是自包含的，不要手动修改
5. 改模板后必须清缓存（`cache_with_pickle` 装饰器）
6. 所有模板注入用 `.replace()` 而非 `.format()`