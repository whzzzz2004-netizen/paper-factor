# paper-factor

LLM 驱动的量化因子挖掘系统。从研报 PDF / 网站文章中提取因子描述 → 自动生成代码 → 300 只股票测试 → 5435 只全量计算 → 评估/绘图/Barra 分析 → 同步到远程。

## 整体流程

```
PDF/网站 → LLM提取因子 → CoSTEER生成代码 → 测试（300 stocks） → 全量（5435 stocks） → 评估 → 远程同步
```

1. **因子提取**: LLM 从研报 PDF（`papers/inbox/`）或网站文章（`papers/website/sources.json`）中提取因子定义
2. **代码生成**: CoSTEER（多轮 LLM 代码生成）根据因子描述编写 Python 函数
3. **模板注入**: `inject_files()` 自动将用户函数包装为完整可运行脚本（数据加载、并行计算、输出处理）
4. **测试**: 在 300 只股票的子集上运行，验证因子可计算
5. **导出**: 测试通过后导出到 `literature_reports/`，并提交全量流水线
6. **全量**: 复制 `.code.py` 到 `文献因子_全量/`，在 5435 只股票上运行
7. **评估**: IC/IR 分析、分位数图、Barra 暴露分析、LLM 审查
8. **同步**: 通过 SMB 同步到远程 Windows 机器

## 目录结构

```
paper-factor/
├── paper_factor_cli/
│   └── main.py                  # CLI 入口（start 命令）
├── papers/
│   ├── inbox/                   # 放入待处理的研报 PDF
│   ├── website/
│   │   └── sources.json         # 网站/公众号文章 URL 列表
│   └── ideas/                   # 因子改进想法
├── rdagent/
│   ├── app/
│   │   └── qlib_rd_loop/
│   │       ├── factor_from_report.py    # 主流程：因子提取+测试+导出 (~1300行)
│   │       ├── factor_full_pipeline.py  # 全量流水线：计算+评估+Barra+同步 (~850行)
│   │       ├── conf.py                  # 配置（LLM 模型、prompts 路径）
│   │       ├── prompts.yaml             # LLM 提示词模板
│   │       └── paper_factor_knowledge.md # 领域知识
│   ├── components/
│   │   └── coder/factor_coder/
│   │       ├── factor.py         # ⭐ 核心模板文件（~1450行）
│   │       │                     # 5 个框架模板 + FactorFBWorkspace 类
│   │       ├── config.py         # 数据路径、执行后端配置
│   │       ├── evaluators.py     # 因子评估器
│   │       └── evolving_strategy.py
│   └── scenarios/qlib/
│       ├── experiment/
│       │   ├── factor_experiment.py
│       │   └── workspace.py
│       └── factor_experiment_loader/
│           ├── pdf_loader.py     # PDF 加载
│           └── json_loader.py
├── scripts/
│   ├── full.py                   # 全量计算 CLI（最早版本，与 full_pipeline 重叠）
│   ├── run_factor_full.py        # 单因子全量运行
│   ├── run_all_full.py           # 批量运行所有因子
│   ├── evaluate_factor.py        # 评估已产出的因子
│   ├── extract_website_factors.py # 网站/公众号因子提取
│   ├── sync_data.py              # 从远程同步数据
│   └── ...                       # 其他工具脚本
├── git_ignore_folder/
│   ├── factor_outputs/
│   │   ├── literature_reports/   # 测试阶段输出（300 stocks）
│   │   ├── 文献因子_全量/         # 全量输出（5435 stocks）
│   │   ├── processed_reports.json # 已处理完成的研报列表
│   │   ├── extracted_reports/    # LLM 提取的因子定义
│   │   ├── availability_cache/   # 因子可用性缓存
│   │   └── task_refinement_cache/
│   └── factor_implementation_source_data/  # ⭐ 全量数据目录（5435 stocks）
│       └── stock_data/
│           ├── daily/            # 日线 parquet（每只股票一个文件）
│           ├── minute_by_date/   # 分钟 parquet（每个交易日一个文件）
│           └── minute/           # 分钟 parquet（每只股票一个文件，旧格式）
├── .env                          # API keys（不提交）
├── pyproject.toml                # 项目配置 + CLI entrypoint
└── README.md
```

## 数据

| 数据 | 目录 | 规模 |
|------|------|------|
| 日线 | `stock_data/daily/{stock}.parquet` | 5435 stocks × ~2027 天 |
| 分钟（按天） | `stock_data/minute_by_date/{date}.parquet` | ~2029 个交易日文件 |
| 分钟（按股，旧） | `stock_data/minute/{stock}.parquet` | 逐步弃用 |
| 测试数据 | `factor_implementation_source_data_1000/` | **300 stocks**（名字误导！） |
| Barra 模型 | `barra_model/` | 风格因子暴露数据 |

日线 parquet 列：`open, close, high, low, volume, factor, market_cap, industry_sw, turnover, return`

分钟 parquet 列：`open, high, low, close, volume, vwap, factor, return, trade_number, avg_turnover`

## 因子模板（factor.py）

`rdagent/components/coder/factor_coder/factor.py` 定义了 5 个框架模板。LLM 只编写用户函数体，框架代码自动注入：

| 模板 | 用户需实现的函数 | 数据源 | 场景 |
|------|------------------|--------|------|
| `DAILY_FRAMEWORK_TEMPLATE` | `calc_factor_single_stock(df, trade_date, stock)` | 日线 per-stock | 动量、反转、均线等 |
| `MINUTE_FRAMEWORK_TEMPLATE` | `calc_factors_one_day(sub, stock)` | 分钟 per-day | 日内因子 |
| `CROSS_SECTION_FRAMEWORK_TEMPLATE` | `calc_factor_cross_section(trade_date)` | 日线 per-stock | 截面因子 |
| `MINUTE_CROSS_SECTION_FRAMEWORK_TEMPLATE` | `calc_factor_minute_raw` + `cross_section_transform` | 分钟 per-day | 分钟截面因子 |
| `DEEP_LEARNING_FRAMEWORK_TEMPLATE` | `train_model()` + `predict()` | 日线 per-stock | GRU/LSTM 等 |

### 模板注入机制

`FactorFBWorkspace.inject_files()`（factor.py ~L1370）：
1. 检测用户代码中出现的函数名 → 判断模板类型
2. 调用 LLM 推断用户函数需要哪些数据列
3. 调用 `_build_factor_code()` 用 `.replace()` 填充模板占位符
4. 生成的完整 `.code.py` 写入 workspace

### 模板内部架构

每个模板生成的 `.code.py` 包含：

- **列过滤**：运行时用 `inspect.getsource()` + 正则自动检测用户函数用到的列，只加载需要的列（减少 60%-80% 内存）
- **并行计算**：`ProcessPoolExecutor`（fork）+ 全局变量 `_WDATA`（COW 共享），避免序列化开销
- **Checkpoint**：每 chunk 保存 `checkpoints/chk_*.parquet`，支持断点续跑，产出后自动清理
- **输出**：pivot 为 Date × Code 宽表，保存为 `result.parquet`

## CLI 命令

| 命令 | 说明 |
|------|------|
| `start` | 全流程：处理未完成的研报 → 网站因子 → 全量补缺 → 等待完成 |
| `run` | 同上，无强制后端 |

### start 命令参数

```bash
start                          # 处理 papers/inbox/ 中所有未完成的 PDF
start -f path/to/paper.pdf     # 处理单个 PDF
start --test-only              # 只跑测试，不跑全量
start --max-factors 10         # 每篇研报最多提取 10 个因子
```

### start 的执行顺序

1. **PDF 处理**：扫描 `papers/inbox/`，对每个未完成的 PDF：
   - LLM 提取因子定义 → CoSTEER 生成代码 → 300 stocks 测试
   - 测试通过 → 导出到 `literature_reports/` → 提交全量流水线
2. **网站处理**：读取 `papers/website/sources.json`，对每个未处理的 URL：
   - 抓取内容 → LLM 提取因子 → 相同测试流程
3. **全量补缺**：扫描 `literature_reports/` 中缺少全量结果的因子，提交后台计算
4. **等待完成**：阻塞直到所有后台全量任务完成

## 全量流水线

`FactorFullPipelineExecutor`（`factor_full_pipeline.py`）在后台串行执行：

1. **类型检测**：扫描 `.code.py` 中的函数名确定因子类型
2. **全量计算**：复制 `.code.py` → 设置 `FACTOR_DATA_DIR` → 子进程执行
3. **后处理**：读取 `result.parquet` → IC/IR 分析 → 分位数图 → Barra 暴露分析 → 写入 `meta.json`
4. **LLM 审查**：对比原始研报摘录检查代码实现，IC 太低或 Barra 偏离过大时触发重写
5. **远程同步**：通过 SMB 同步到 `192.168.1.13` 的 `E:\paper_factors\`

## 关键配置

### 数据路径（config.py）

```python
data_folder = "git_ignore_folder/factor_implementation_source_data"       # 全量（5435 stocks）
data_folder_debug = "git_ignore_folder/factor_implementation_source_data_1000"  # 测试（300 stocks！）
```

环境变量 `FACTOR_DATA_DIR` 可覆盖数据路径。

### N_WORKERS 默认值

| 因子类型 | 默认值 | 说明 |
|---------|--------|------|
| 日线单股 | `os.cpu_count() or 16` | CPU 密集型 |
| 分钟 | 16 | 已验证 189 天回看 × 16 核稳定 |
| 截面 | 4 | 防 OOM（每 chunk 加载所有股票） |
| 分钟截面 | 16 | 已验证稳定 |

环境变量 `FACTOR_N_WORKERS` 可覆盖。

### 执行后端

当前统一使用 `local`（直接子进程执行）。Docker 已废弃（WSL2 bind mount 同步有问题）。

## 产出物

每个因子在全量运行后产出（以 `daily_momentum` 为例）：

```
文献因子_全量/温和收益的动量与极端收益的反转效应/daily_momentum/
├── result.parquet       # 因子值宽表（Date × Code）
├── daily_momentum.code.py  # 完整的自包含运行代码
├── ic_series.png        # IC 序列图
├── decile_plot.png      # 分位数组合收益图
├── barra_exposure.png   # Barra 风格暴露分析
└── meta.json            # 评估指标汇总
```

## 远程同步

- 目标：`pc@192.168.1.13` / `E:\paper_factors\`
- 方式：`sshpass` + `subprocess.Popen` 自动挂载 sshfs
- 依赖：`conda install -c conda-forge sshpass`
- 脚本：`scripts/sync_utils.py`

## 环境要求

- Python 3.10+
- 操作系统：Linux（WSL2 可用）
- 内存：至少 32GB（全量分钟因子需要 ~6.6GB）
- JQData 账号（用于获取指数成分股等数据）

## 一键运行全量因子

只需要两步：

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 一键跑（自动挂载远程 E 盘 + 扫描 + 全量计算 + 出图 + Barra）
python scripts/run_all_pending_full.py
```

脚本会自动：
1. 挂载 `//192.168.1.13/E` 到 `/mnt/remote_e`（需同局域网）
2. 设置 `FACTOR_DATA_DIR` 指向远程数据
3. 扫描 `文献因子_全量/` 下所有有 `.code.py` 但无 `.parquet` 的因子
4. 逐个执行全量计算 → 评估 IC/IR → 分位数图 → Barra 分析 → 写入 meta.json
5. 完成后输出统计

不支持远程 E 盘？用 `--no-mount` 后手动设 `FACTOR_DATA_DIR`：

```bash
export FACTOR_DATA_DIR=/你的本地数据路径
python scripts/run_all_pending_full.py --no-mount
```

其他选项：

```bash
python scripts/run_all_pending_full.py --dry-run      # 只列出待运行因子
python scripts/run_all_pending_full.py --report "报告名"  # 只跑某份报告
```

```bash
# 安装
git clone <repo>
cd paper-factor
pip install -e .

# 配置
cp .env.example .env
# 编辑 .env 填入 LLM API keys 和 JQData 账号

# 放入研报
cp some_paper.pdf papers/inbox/

# 运行
start
```

## 关键设计决策

- **`.code.py` 即全量代码**：测试阶段生成的 `.code.py` 已包含完整模板，全量时原样复制、只改数据路径
- **fork COW 共享内存**：分钟因子用 `ProcessPoolExecutor(fork)`，主进程加载数据后子进程通过 Copy-On-Write 共享，避免 pickle 序列化
- **按需列加载**：运行时通过 AST 分析用户函数用到的列，只加载需要的 parquet 列，分钟数据从 ~30GB 降到 ~6.6GB
- **Checkpoint 断点续跑**：每 chunk 保存 checkpoint，崩溃后自动从最大 checkpoint 继续