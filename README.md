# paper-factor

LLM 驱动的量化因子挖掘系统。从研报 PDF / 网站文章中提取因子 → 自动编码测试 → 全量计算 → 远程同步。

## 快速开始（从零到跑通 /factor）

### 1. 前置条件

| 依赖 | 说明 | 验证 |
|------|------|------|
| **网络** | 能访问 `192.168.1.13:445`（SMB） | `ping 192.168.1.13` |
| **smbclient** | SMB 文件传输 | `smbclient -L //192.168.1.13 -U pc` |
| **Python 3.10+** | 运行环境 | `python3 --version` |
| **Claude Code CLI** | AI 因子提取引擎（WSL 安装见下方） | `claude --version` |

### 2. 安装 Claude Code（WSL 环境）

```bash
# 安装 Node.js（如未装）
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装 Claude Code
npm install -g @anthropic-ai/claude-code

# 验证
claude --version

# 首次使用需登录
claude --login
```

### 3. 克隆并安装依赖

```bash
git clone <repo-url> paper-factor
cd paper-factor

# Python 依赖
pip install -r requirements.txt
# 或 conda:
# conda env create -f environment.yml
# conda activate paper-factor

# smbclient（Ubuntu/Debian）
sudo apt-get install smbclient
# macOS
# brew install samba
```

### 3. 同步数据

```bash
python3 scripts/sync_data.py --full
```

这条命令自动完成：
- 从远程 E 盘拉取日线 + 分钟线 + 财务 + 行业数据
- 转换为 per-stock parquet 格式（5435 只）
- 裁剪 300 只 × 600 天测试子集到 `factor_implementation_source_data_1000`
- 注册新列到 schema，更新 prompt 文件
- 生成涨停列表

> 首次全量同步较慢（~30 分钟）。日常增量用 `python3 scripts/sync_data.py` 即可。

### 4. 运行 /factor 技能

```bash
# 进入项目目录后启动 Claude Code
claude

# 在 Claude Code 中执行：
#   /factor
#
# 这会自动：
#   1. 扫描 papers/inbox/、websites、ideas 中的未处理内容
#   2. 5 个 agent 并行提取因子 → 编码 → 测试
#   3. 导出到 literature_reports/ 并部署到 文献因子_全量/
#   4. 同步到远程 E 盘
```

### 5. 可选：全量计算

```bash
# 某个因子全量跑
python scripts/claude_factor_helper.py run-full \
  --code 文献因子_全量/<报告>/<因子>/<因子>.code.py \
  --factor-name <因子名> \
  --report-name "<报告名>"

# 或批量跑所有因子
python scripts/claude_factor_helper.py run-all-full
```

## 整体架构

```
                    ┌──────────────────┐
                    │  远程 E 盘        │
                    │  192.168.1.13     │
                    │  ─ market_daily   │
                    │  ─ dailyData      │
                    │  ─ market_minute  │
                    │  ─ paper_factors\ │
                    │    文献因子_全量   │
                    └────────┬─────────┘
                             │ SMB (smbclient)
                             ▼
  ┌────────┐    ┌──────────────────────┐    ┌─────────────────┐
  │ /factor│───▶│ 测试 (300只×600天)   │───▶│ literature_reports/
  │ skill  │    │ _1000 目录           │    │ 每个因子: code.py│
  └────────┘    └──────────────────────┘    │        meta.json │
                                            │        parquet   │
  ┌────────┐    ┌──────────────────────┐    └────────┬────────┘
  │deploy  │───▶│ 文献因子_全量/        │─────────────▶ 远程 E 盘
  │to-full │    │ (本地 或 /mnt/remote_e)│   sync-full
  └────────┘    └──────────────────────┘
```

## 数据目录结构与格式

### git_ignore_folder 完整结构

```
git_ignore_folder/
├── factor_implementation_source_data/        # 全量数据源（5435只股票）
│   ├── stock_data/
│   │   ├── daily/{code}.parquet              # 日线 per-stock parquet（5435个文件）
│   │   │                                      # 列: open, close, high, low, volume, factor,
│   │   │                                      #      pct_chg, pre_close, turnover_rate,
│   │   │                                      #      market_cap, pe_ttm, pb, roe, roa, ... 共120+列
│   │   ├── minute/{code}.parquet             # 分钟 per-stock parquet（5435个文件）
│   │   │                                      # 列: open, close, high, low, volume, vwap, factor, return
│   │   ├── minute_by_date/{YYYY-MM-DD}.parquet # 分钟 by date（2029个文件，每天一个）
│   │   │                                      # 列: datetime, instrument, open, close, high, low, volume, vwap, factor, return
│   │   ├── stock_list.json                   # 全量股票列表（5435只）
│   │   ├── trade_dates.json                  # 交易日列表（2027天）
│   │   └── industry.json                     # 申万一级行业分类
│   ├── daily_pv.h5                           # 日线 H5 格式（MultiIndex [datetime, instrument]）
│   ├── minute_pv.h5                          # 分钟 H5 格式（备用）
│   ├── limit_up_daily.parquet                # 涨停列表
│   ├── factor_field_schema.json              # 字段注册表
│   └── data_field_dictionary.md              # 完整字段字典
│
├── factor_implementation_source_data_1000/   # 测试数据子集（300只×600天）
│   └── stock_data/                           # 同全量结构，但只有300只股票
│       ├── daily/{code}.parquet
│       ├── minute/{code}.parquet
│       ├── minute_by_date/{YYYY-MM-DD}.parquet
│       ├── stock_list.json
│       └── trade_dates.json
│
├── factor_outputs/
│   ├── literature_reports/YYYYMMDD/          # 测试因子产出
│   │   └── <报告名>/<因子名>/
│   │       ├── <因子名>.code.py              # 自包含的因子代码（含模板）
│   │       ├── <因子名>.parquet              # 测试结果（300只×600天）
│   │       └── <因子名>.meta.json            # 因子元数据
│   │
│   ├── 文献因子_全量/YYYYMMDD/               # 全量因子产出（本地缓存）
│   │   └── <报告名>/<因子名>/
│   │       ├── <因子名>.code.py              # 自包含的因子代码（全量数据路径）
│   │       ├── <因子名>.parquet              # 全量结果（5435只×2027天，~80MB）
│   │       ├── <因子名>.decile.png           # 十分位收益图
│   │       └── <因子名>.meta.json            # 因子元数据（含评估指标）
│   │
│   └── 文献因子_每日更新/YYYYMMDD/           # 每日增量更新产出
│       └── <报告名>/<因子名>/
│           └── <因子名>.parquet              # 增量更新的 parquet
│
├── barra_model/                              # Barra 风险模型数据
├── logs/                                     # 日志文件
│   └── daily_update.log
├── daily_update_config.json                  # 每日更新配置
└── daily_update_status.json                  # 每日更新状态
```

### 如何从远程复制数据到本地

远程数据存储在 `192.168.1.13` 的 `E:\` 盘，通过 SMB 协议访问。

**方法 1：使用 sync_data.py（推荐）**

```bash
# 全量同步（首次使用）
python scripts/sync_data.py --full

# 增量同步（日常更新）
python scripts/sync_data.py

# 仅检查远程有无新数据
python scripts/sync_data.py --check
```

`sync_data.py` 自动完成：
- 从远程 E 盘拉取日线 + 分钟线 + 财务 + 行业数据
- 转换为 per-stock parquet 格式（5435只）
- 裁剪 300只 × 600天 测试子集
- 注册新列到 schema，更新 prompt 文件
- 生成涨停列表

**方法 2：手动挂载 CIFS**

```bash
# 挂载远程 E 盘
sudo mkdir -p /mnt/remote_e
sudo mount -t cifs //192.168.1.13/E /mnt/remote_e \
  -o username=pc,password=,rw,uid=$(id -u),gid=$(id -g),iocharset=utf8,file_mode=0755,dir_mode=0755,noperm

# 手动复制数据
cp -r /mnt/remote_e/_paper_factor_unified/factor_implementation_source_data/stock_data/daily/ \
  git_ignore_folder/factor_implementation_source_data/stock_data/daily/

# 复制因子产出
cp -r "/mnt/remote_e/paper_factors/文献因子_全量/" \
  git_ignore_folder/factor_outputs/文献因子_全量/

# 卸载
sudo umount /mnt/remote_e
```

**方法 3：使用 smbclient**

```bash
# 下载单个文件
smbclient //192.168.1.13/E -U pc -c 'get paper_factors\文献因子_全量\报告\因子\因子.parquet /tmp/因子.parquet'

# 递归下载目录（需 tar 配合）
smbclient //192.168.1.13/E -U pc -c 'tar c paper_factors\文献因子_全量\报告\' | tar x
```

### 日线 parquet 文件格式

每只股票的日线 parquet 包含全部历史数据，列如下：

| 列名 | 含义 | 备注 |
|------|------|------|
| datetime | 交易日 | DatetimeIndex |
| open | 开盘价 | 前复权 |
| close | 收盘价 | 前复权 |
| high | 最高价 | 前复权 |
| low | 最低价 | 前复权 |
| volume | 成交量(股) | |
| factor | 复权因子 | 前复权因子 |
| pct_chg | 涨跌幅(%) | 含隔夜跳空 |
| pre_close | 前收盘价 | |
| turnover_rate | 换手率(%) | |
| market_cap | 总市值(元) | |
| circulating_market_cap | 流通市值(元) | |
| pe_ttm | 市盈率(TTM) | |
| pb | 市净率 | |
| roe | 净资产收益率(%) | |
| roa | 总资产净利率(%) | |
| revenue_yoy | 营收同比(%) | |
| profit_yoy | 净利润同比(%) | |
| ... | 共120+列 | 含基本面、技术指标等 |

### 全量因子产出格式

每个因子的 parquet 文件为宽表格式：

- **index**: trade_date（字符串格式 `YYYY-MM-DD`）
- **columns**: 股票代码（int64）
- **values**: 因子值（float64，含 NaN）
- 行列已排序，`index.name = "trade_date"`，`columns.name = "stock_code"`
- 涨停日期的因子值已被剔除（设为 NaN）
- 每个 parquet 约 80MB（2027天 × 5435只）

## 日期子目录结构

每次运行按当天日期新建 `YYYYMMDD/` 子目录：

```
factor_outputs/
├── literature_reports/20260726/<报告>/<因子>/
├── 文献因子_全量/20260726/<报告>/<因子>/
└── 文献因子_每日更新/20260726/<报告>/<因子>/
```

`run_all.py --date YYYYMMDD` 只处理指定日期子目录下的因子。`daily_update.py` 同理。

## 向量化日线模板

2026-07-26 新增 `calc_factor_series` 向量化模式，将单因子计算从 O(n²) 优化到 O(n)：

- **旧模式**：`calc_factor_single_stock(df, trade_date, stock)` — 每只股票调用 2027 次
- **新模式**：`calc_factor_series(df, stock)` → `pd.Series` — 每只股票调用 1 次
- 性能提升：~28分钟/因子 → ~5分钟/因子（5435只股票）
- 旧模式自动回退：模板检测 `calc_factor_series` 不存在时自动使用旧模式

## 技能命令

### `/factor` — 因子提取全流程

扫描待处理内容 → 多 agent 并行提取 → 编码测试 → 导出部署 → 同步远程。

支持的输入：
- `papers/inbox/*.pdf` — 研报 PDF
- `papers/website/sources.json` — 网站文章
- `papers/ideas/` — 文本想法

输出目录：`git_ignore_folder/factor_outputs/literature_reports/<报告>/<因子>/`

### `/getdata` — 增量数据同步

从远程 E 盘同步最新数据，自动检测新列并更新 prompt 文件。

### `/clean` — 清理所有因子产出

删除测试因子、全量因子、Python 缓存。

## 命令行工具

核心入口：`python scripts/claude_factor_helper.py <命令> [参数]`

| 命令 | 用途 |
|------|------|
| `scan-pending` | 扫描未处理的内容 |
| `test-and-export` | 测试因子并导出到 literature_reports |
| `deploy-to-full` | 部署测试因子到全量目录 |
| `sync-full` | 同步全量因子到远程 E 盘 |
| `trigger-full` | 触发全量流水线（计算+评估+同步） |
| `run-full` | 运行单因子全量计算 |
| `run-all-full` | 批量运行所有因子全量计算 |
| `show-columns` | 查看可用数据列 |
| `find-similar` | 在因子记忆库中查找同类因子 |
| `retrieve-knowledge` | 检索领域知识（涨停规则等） |
| `mark-done` | 标记内容为已处理 |

数据同步：`python scripts/sync_data.py`

| 参数 | 用途 |
|------|------|
| `--check` | 检查远程有无新数据 |
| `--full` | 全量同步（覆盖） |
| `--dry-run` | 只看变更不执行 |
| `--update-prompts-only` | 仅更新 prompt 文件 |

## 目录结构

```
paper-factor/
├── papers/
│   ├── inbox/                    # 放入待处理的研报 PDF
│   ├── website/
│   │   └── sources.json          # 网站文章 URL 列表
│   └── ideas/                    # 文本因子想法
├── scripts/
│   ├── claude_factor_helper.py   # 核心 CLI 工具
│   ├── sync_data.py              # 数据同步（SMB）
│   └── sync_utils.py             # 远程同步工具
├── rdagent/
│   └── components/coder/factor_coder/factor.py  # 模板（5种类型）
├── .claude/
│   └── skills/factor/            # /factor 技能定义
│       ├── SKILL.md
│       └── knowledge/            # 领域知识（涨跌停规则、列定义等）
├── data/
│   └── schema.json               # 字段注册表
└── git_ignore_folder/
    └── factor_outputs/
        ├── literature_reports/   # 测试产出
        └── 文献因子_全量/         # 全量产出（本地缓存）
```

## 常见问题

### smbclient 连不上

```bash
# 检查连通性
ping 192.168.1.13
smbclient -L //192.168.1.13 -U pc
# 如果端口 445 被屏蔽，检查 VPN / 防火墙
```

### 测试数据没有被同步

```bash
python3 scripts/sync_data.py --full  # 强制全量同步
```

### 全量因子没有部署到远程

```bash
# 先确保本地有测试通过
python scripts/claude_factor_helper.py deploy-to-full --code literature_reports/<报告>/<因子>/<因子>.code.py
# 然后同步到远程
python scripts/claude_factor_helper.py sync-full --report "<报告名>"
```

### /factor 运行一半断了

重新运行 `/factor`。`scan-pending` 会跳过已标记完成的内容，只处理未完成的。