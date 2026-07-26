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

## 关键概念

### 数据目录

| 目录 | 内容 | 用途 |
|------|------|------|
| `factor_implementation_source_data/` | 5435 只全量数据 | 全量计算 |
| `factor_implementation_source_data_1000/` | 300 只 × 600 天子集 | 因子测试 |
| `git_ignore_folder/factor_outputs/literature_reports/` | 测试产出 | 因子调试 |
| `git_ignore_folder/factor_outputs/文献因子_全量/` | 全量产出（本地缓存） | 本地备份 |
| 远程 `E:\paper_factors\文献因子_全量\` | 全量产出（主存储） | 远程同步 |

### 数据格式

- **日线**：per-stock parquet，`stock_data/daily/{code}.parquet`
- **分钟 per-stock**：`stock_data/minute/{code}.parquet`
- **分钟 by date**：`stock_data/minute_by_date/{YYYY-MM-DD}.parquet`
- 每只股票的 parquet 包含全部历史，追加新日期

### 因子类型

| 类型 | 函数签名 | 数据源 | 说明 |
|------|----------|--------|------|
| `daily` | `calc_factor_single_stock(df, trade_date, stock)` | 日线 per-stock | 截面/时间序列 |
| `minute` | `calc_factors_one_day(df, stock)` | 分钟 per-stock | 日内因子 |
| `cross_section` | `calc_factor_cross_section(all_data, trade_date)` | 日线 per-stock | 截面比较 |
| `minute_cs` | `calc_factor_minute_raw(df, stock)` + `cross_section_transform(all_values)` | 分钟 by date | 日内截面 |
| `deep_learning` | `train_model(all_data, trade_date)` + `predict_batch(model, data_dict, trade_date)` | 日线 per-stock | 深度学习 |

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