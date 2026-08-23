# 在新电脑上完整复现本项目

> 目标：在老板/其他电脑上完整复现 paper-factor 项目，能跑 `/factor`、全量计算、增量更新。
>
> **核心结论：代码用 git clone，数据仓库直接拷贝（或通过 `/getdata` 逐步导入）。**

---

## 复现需要什么

| 东西 | 在哪 | 怎么获取 |
|------|------|----------|
| 代码（scripts/rdagent/.claude/templates） | git 仓库 | `git clone` |
| 数据源（日线/分钟/非行情 parquet） | `/mnt/d/paper-factor-data/数据仓库/` | 从已有环境直接拷贝，或逐步导入 |
| 因子产出（测试/全量） | `/mnt/d/paper-factor-data/数据仓库/因子产出/` | 从已有环境拷贝，或重新计算 |
| 研报 PDF | `papers/inbox/` | 手动拷贝 |
| 密钥配置 | `.env` | 手动创建 |

---

## 前置条件

- Python ≥ 3.10（建议 conda 环境）
- 能访问原有环境的数据仓库（或准备从零导入）

---

## Step 1 — 获取代码

```bash
git clone git@github.com:whzzzz2004-netizen/paper-factor.git
cd paper-factor
```

（或 HTTPS：`git clone https://github.com/whzzzz2004-netizen/paper-factor.git`）

---

## Step 2 — 安装依赖

```bash
conda create -n rdagent python=3.10 -y
conda activate rdagent
pip install -r requirements.txt
pip install -e .          # 完整依赖（pyproject.toml）
```

---

## Step 3 — 准备数据

### 方式 A：从已有环境拷贝（推荐）

直接把整个 `/mnt/d/paper-factor-data/数据仓库/` 目录从原有环境拷贝到项目根目录：

```bash
# 从原有环境（WSL/Ubuntu）
cp -r /path/to/old/paper-factor/数据仓库 ./数据仓库
```

需要大约 200GB 磁盘空间（全量日线 + 分钟 + 非行情 + 因子产出）。

### 方式 B：从零逐步导入

如果没有现成数据，可以通过 `/getdata` 技能逐步导入：

1. 准备好原始数据文件（CSV/parquet），放到 `/mnt/d/paper-factor-data/新建文件/` 目录下
2. 运行 `python scripts/import_new_data.py --check` 预览
3. 运行 `python scripts/import_new_data.py` 执行导入

详细数据目录结构见 README.md 的「数据目录结构」章节。

---

## Step 4 — 配置 `.env`

复制模板并填写：

```bash
cp .env.example .env
```

`.env` 里需要的项：

| 变量 | 用途 | 是否必须 |
|------|------|----------|
| `OPENAI_API_KEY` | LLM 因子提取/审查 | 是（/factor 需要） |
| `OPENAI_API_BASE` | API 地址 | 按需 |
| `CHAT_MODEL` | 模型名（如 `openai/gpt-4o`） | 按需 |
| `JQDATA_USERNAME` | 聚宽账号（分钟因子模板需要） | 分钟因子按需 |
| `JQDATA_PASSWORD` | 聚宽密码 | 分钟因子按需 |

---

## Step 5 — 验证

```bash
# 数据完整性检查
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/trade_dates.json')); print(f'日线测试: {len(sl)}只×{len(td)}天')"
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/trade_dates.json')); print(f'分钟测试: {len(sl)}只×{len(td)}天')"

# 扫描待处理研报
python scripts/claude_factor_helper.py scan-pending

# 查看可用数据列
python scripts/claude_factor_helper.py show-columns
```

测试数据正常输出应为 `300只×300天`。

---

## 常见操作

```bash
# 处理新研报（端到端）
#   用 Claude Code 打开项目，执行 /factor

# 全量计算 / 增量更新
python scripts/run_all.py                      # 自动扫描并计算
python scripts/run_all.py --workers 3          # 3因子并行
python scripts/run_all.py --dry-run            # 仅查看计划

# 数据导入
python scripts/import_new_data.py --check      # 预览新建文件
python scripts/import_new_data.py              # 执行导入
```

---

## 常见问题

**Q: 只想看已有因子结果，不跑计算？**
A: 直接看 `/mnt/d/paper-factor-data/数据仓库/因子产出/全量/{DATE}/{report}/{factor}/` 下的 `.parquet` 和 `.decile.png`。

**Q: 跑 `/factor` 需要研报 PDF 吗？**
A: 处理新研报需要 PDF（放到 `papers/inbox/`）。已 mark-done 的研报不需要。

**Q: 数据从哪里来？**
A: 本项目不提供原始数据。需要从已有数据源（如聚宽、Tushare、券商数据终端）获取，通过 `import_new_data.py` 导入。

**Q: 测试数据为什么是 300 只×300 天？**
A: 固定不变。测试数据只用于验证因子代码逻辑正确性，不用于最终结果。全量计算使用全量数据目录（5435 只×全历史）。