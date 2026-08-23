# paper-factor

LLM 驱动的量化因子挖掘系统。从研报 PDF / 文章提取因子定义 → 自动编码测试 → 全量计算。

**本地执行，不依赖 Docker。**

---

## 整体流程

```
研报PDF / 想法文本
        │
        ▼  [Phase 1: 提取]
  提取因子定义（5个agent并行）
  输出: 因子产出/测试/{DATE}/{report}/factor_definitions.json
        │
        ▼  [Phase 2: 编码+测试+部署]
  每个因子写核心函数 → test-and-export（300只×300天测试数据）
  测试通过后立即部署到全量目录
  输出: 因子产出/测试/{DATE}/{report}/{factor}/
          ├── {factor}.code.py       # 自包含代码
          ├── {factor}.parquet       # 测试结果
          └── {factor}.meta.json
  输出: 因子产出/全量/{DATE}/{report}/{factor}/
          └── {factor}.code.py       # 自动部署的全量代码
        │
        ▼  [run_all / run_factor_full]
  全量计算（5435只×全历史）+ 评估/绘图/Barra
  输出: 因子产出/全量/{DATE}/{report}/{factor}/
          ├── {factor}.parquet       # 全量结果
          ├── {factor}.decile.png    # 十分组收益图
          └── {factor}.meta.json     # 含IC/IR/Barra等评估指标
```

---

## 全新复现步骤

要在一台新电脑上完全复现这个项目，需要做以下事情：

### 第一步：克隆代码

```
git clone <仓库地址> paper-factor
cd paper-factor
```

### 第二步：获取数据仓库

数据仓库是项目运行的核心，包含行情数据、非行情数据、因子产出等，约 61GB。
从已有环境用 rsync 复制：

```
rsync -avhP user@dev-machine:/path/to/paper-factor//mnt/d/paper-factor-data/数据仓库/ .//mnt/d/paper-factor-data/数据仓库/
```

或者用硬盘拷贝整个 `/mnt/d/paper-factor-data/数据仓库/` 目录到项目根目录。

**如果只做因子开发测试（不跑全量）：**
只需要 `/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/` 和 `/mnt/d/paper-factor-data/数据仓库/非行情数据/测试/` 两个目录（约 25MB），300 只股票 × 300 天。

### 第三步：数据符号链接（workspace/）

`workspace/` 里是数据目录的符号链接（已提交到 git，`setup.sh` 也会重建）。它包含：

```
workspace/
├── factor_implementation_source_data/     # → 指向数据仓库的符号链接
│   └── stock_data/daily/                  # ln -s /mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量
├── factor_implementation_source_data_1000/ # → 指向测试数据
│   └── stock_data/daily/                  # ln -s /mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试
└── ideas/                                 # 因子想法暂存
```

创建方式（git clone 后已带符号链接；如需重建）：

```bash
mkdir -p workspace/logs workspace/ideas
ln -sfn ..//mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量 workspace/factor_implementation_source_data
ln -sfn ..//mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试 workspace/factor_implementation_source_data_1000
```

运行时生成的 `workspace/RD-Agent_workspace/`（代码执行工作区）与 `workspace/logs/`（日志）
不进 git，运行时自动创建。

### 第四步：安装依赖

```
pip install -r requirements.txt
```

### 第五步：配置 .env

复制 `.env.example` 为 `.env`，填入：

- **JQDATA_USERNAME / JQDATA_PASSWORD**：聚宽账号，分钟因子计算需要
- **FACTOR_DATA_DIR**：数据目录，默认指向本地 `/mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量`
- **OPENAI_API_KEY / OPENAI_API_BASE**：LLM API，因子提取/审查用
- **CHAT_MODEL**：使用的模型

### 第六步：验证

```bash
# 检查数据完整性
python -c "from pathlib import Path; d=Path('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量/stock_data/daily'); print(f'{len(list(d.glob(\"*.parquet\")))} 只股票')"

# 检查 workspace 符号链接
ls -la workspace/factor_implementation_source_data

# 跑一个测试因子
python scripts/run_factor_full.py "/mnt/d/paper-factor-data/数据仓库/因子产出/全量/20260810/GRU量价趋势预测因子/GRUPriceVolumePredictor/GRUPriceVolumePredictor.code.py"
```

---

## 数据目录结构

```
/mnt/d/paper-factor-data/数据仓库/                          # 所有数据（本地，不入git，约61GB）
├── 行情数据/
│   ├── 日线/
│   │   ├── 全量/stock_data/daily/{code}.parquet   # 5435只 × 全历史
│   │   └── 测试/stock_data/daily/{code}.parquet   # 300只 × 300天
│   └── 分钟线/
│       ├── 全量/stock_data/
│       │   ├── minute/{code}.parquet              # 5435只
│       │   └── minute_by_date/{date}.parquet      # 按日期
│       └── 测试/stock_data/                        # 300只 × 300天
├── 非行情数据/                                    # 非行情（基本面/截面因子等）
│   ├── 全量/stock_data/daily/{code}.parquet       # 5435只 × 18列
│   └── 测试/stock_data/daily/{code}.parquet       # 300只 × 300天
├── 板块数据/                                       # 行业分类等
├── barra_model/                                    # Barra风险模型数据
└── 因子产出/
    ├── 测试/{DATE}/...                             # 测试因子产出
    └── 全量/{DATE}/...                             # 全量因子产出
```

## 数据目录选择策略

脚本按以下优先级确定数据目录：

```
1. FACTOR_DATA_DIR 环境变量（最优先）
2. RDAGENT_FACTOR_DATA_DIR 环境变量
3. /mnt/d/paper-factor-data/数据仓库/行情数据/日线/全量/（相对路径）
```

---

## 工作流

### 数据导入（/getdata）

把新增数据文件放到 `/mnt/d/paper-factor-data/新建文件/` 目录，然后：

```bash
python scripts/import_new_data.py --check    # 预览
python scripts/import_new_data.py            # 执行导入
python scripts/import_new_data.py --dry-run  # 预览不执行
```

自动分类行情/非行情/分钟数据，新列注册到 `data/schema.json`。
新增截面因子时，需先在 `/mnt/d/paper-factor-data/新建文件/新因子描述.csv` 中描述因子含义（CSV 无表头，`因子名,描述文本`）。

### 因子提取 + 测试（/factor 技能）

在 Claude Code 中执行 `/factor`，两阶段：

**Phase 1：提取** — 并行读取研报 → 定义因子（name, type, lookback, formulation）
**Phase 2：编码+部署** — 每个因子写核心函数 → `test-and-export`（300只测试数据验证）→ 测试通过后立即 `deploy-to-full` 到全量目录

输出到 `/mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/` 和 `/mnt/d/paper-factor-data/数据仓库/因子产出/全量/{DATE}/`

### 全量计算

```bash
python scripts/claude_factor_helper.py deploy-to-full \
  --code /mnt/d/paper-factor-data/数据仓库/因子产出/测试/{DATE}/{report}/{factor}/{factor}.code.py \
  --date {DATE}
```

### 全量计算

**单因子：**

```bash
python scripts/run_factor_full.py \
  /mnt/d/paper-factor-data/数据仓库/因子产出/全量/{DATE}/{report}/{factor}/{factor}.code.py
```

含评估 + 十分组图 + Barra + LLM审查。

**批量：**

```bash
python scripts/run_all.py                    # 默认当天日期子目录
python scripts/run_all.py --workers 3        # 3因子并行
python scripts/run_all.py --force            # 强制重跑
python scripts/run_all.py --dry-run          # 仅查看计划
```

或者在 Claude Code 中执行 `/all`。

---

## 因子类型与模板

| 类型 | 模板 | 核心函数 | 并行策略 |
|------|------|----------|----------|
| `daily` | DAILY_FRAMEWORK_TEMPLATE | `calc_factor_single_stock(df, trade_date, stock)` | ThreadPoolExecutor |
| `minute` | MINUTE_FRAMEWORK_TEMPLATE | `calc_factors_one_day(df, stock)` | ThreadPoolExecutor |
| `cross_section` | CROSS_SECTION_FRAMEWORK_TEMPLATE | `calc_factor_cross_section(trade_date)` | ProcessPoolExecutor(loky) |
| `minute_cs` | MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE | `calc_factor_minute_raw` + `cross_section_transform` | joblib(threading) |
| `deep_learning` | DEEP_LEARNING_FRAMEWORK_TEMPLATE | `train_model()` + `predict()` | GPU batch |

所有模板在 `rdagent/components/coder/factor_coder/factor.py` 的 `FactorFBWorkspace` 类中。

---

## 因子产出对比

### 日线因子（测试 vs 全量）

| 指标 | 测试 | 全量 |
|------|------|------|
| 股票数 | 300 只 | 5435 只 |
| 天数 | 300 天 | 全历史（~2000 天） |
| 数据大小 | 16MB | 1.2GB |
| 运行时间 | 几秒 | 几分钟 |

### 分钟因子（测试 vs 全量）

| 指标 | 测试 | 全量 |
|------|------|------|
| 股票数 | 300 只 | 5435 只 |
| 天数 | 300 天 | 全历史（~2027 天） |
| 数据大小 | 8.2GB | 35GB |
| 运行时间 | 几十秒 | 几十分钟 |

---

## CLI 命令速查

### 因子处理

```bash
python scripts/claude_factor_helper.py scan-pending                  # 扫描未处理内容
python scripts/claude_factor_helper.py test-and-export ...           # 测试+导出
python scripts/claude_factor_helper.py deploy-to-full ...            # 部署全量
python scripts/claude_factor_helper.py mark-done --name "文件名.pdf"  # 标记完成
```

### 数据导入

```bash
python scripts/import_new_data.py --check           # 预览新建文件
python scripts/import_new_data.py                   # 执行导入
```

### Claude Code 技能

```
/factor     — 因子提取全流程（扫描→提取→编码→测试→部署→标记完成）
/getdata    — 从 /mnt/d/paper-factor-data/新建文件/ 导入新增数据
/all        — 全量/增量运行所有因子
/clean      — 删除所有因子产出 + Python 缓存
```

---

## 依赖

- LLM 调用：`litellm`, `openai`
- 数据处理：`pandas`, `numpy`, `pyarrow`, `scikit-learn`
- 聚宽数据：`jqdatasdk`
- 工具：`filelock`, `python-dotenv`, `rich`, `tqdm`, `loguru`
- PDF 解析：`pymupdf`
- 深度学习：`torch`（GPU 可选）

**不依赖 Docker。** 执行后端为 `local`。

---

## 重要规则

1. **改模板后必须先跑测试数据验证**，并清 pickle 缓存
2. **测试数据固定 300 天、固定不动**，导入只补新列不加新日期
3. **标准化只能是截面 zscore**，绝不能是时序标准化（未来数据泄露）
4. `.code.py` 是自包含的，不要手动修改
5. 分钟因子需要 JQData 账号（聚宽）
6. 全量运行设 `FACTOR_LOOKBACK_CAP=99999` ≈ 无上限