# 在新电脑上完整复现本项目

> 目标：在老板/其他电脑上完整复现 paper-factor 项目，能跑 `/factor`、全量计算、增量更新。
>
> **核心结论：代码用 git clone，数据仓库直接拷贝（或通过 `/getdata` 逐步导入）。**

---

## 数据盘路径（D 盘 / G 盘）

本机数据在 **D 盘**，另一台电脑的数据在 **G 盘**：

| 机器 | Windows 路径 | WSL 路径 |
|---|---|---|
| 当前开发机 | `D:\paper-factor-data` | `/mnt/d/paper-factor-data` |
| 另一台电脑 | `G:\paper-factor-data` | `/mnt/g/paper-factor-data` |

数据导入脚本按以下顺序自动解析：

1. `PAPER_FACTOR_DATA_ROOT` 环境变量（显式指定）
2. 自动探测：**D 盘存在 `数据仓库` → 用 D 盘**
3. 否则用 G 盘

所以 **Windows 原生导入**直接在本机运行：

```bat
D:\paper-factor-data\scripts\getdata.bat
```

或在 `D:\paper-factor-data\scripts` 下：

```bat
python getdata.py
```

需要强制指定时：

```bat
set PAPER_FACTOR_DATA_ROOT=G:\paper-factor-data
```

WSL：

```bash
export PAPER_FACTOR_DATA_ROOT=/mnt/g/paper-factor-data
```

查看当前实际使用的数据根：

```bash
python scripts/claude_factor_helper.py data-root
```

> ⚠️ 如果 `.env` 里还写着旧盘的 `FACTOR_DATA_DIR` / `RDAGENT_FACTOR_DATA_DIR`，
> 请改成当前数据根或直接删除这两行，否则旧路径会覆盖自动探测。
>
> Windows 原生导入不依赖 bash/WSL；`repo_path.txt` 若仍指向 WSL 仓库路径，
> `getdata` 会跳过“刷新提示词”，不影响数据导入。需要刷新提示词时，
> 设置 `PAPER_FACTOR_REPO` 为 Windows 仓库路径，或修改 `repo_path.txt`。

---

## 复现需要什么

| 东西 | 在哪 | 怎么获取 |
|------|------|----------|
| 代码（scripts/rdagent/.claude/templates） | git 仓库 | `git clone` |
| 数据源（日线/分钟/非行情 parquet） | `<数据根>/数据仓库/`（D 盘或 G 盘） | 从已有环境直接拷贝，或逐步导入 |
| 因子产出（测试/全量） | `<数据根>/数据仓库/因子产出/`（D 盘或 G 盘） | 从已有环境拷贝，或重新计算 |
| 研报 PDF | `<数据根>/papers/inbox/`（D 盘或 G 盘） | 手动拷贝 |
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

1. 准备好原始数据文件（CSV/parquet），放到 `/mnt/d/paper-factor-data/原始数据/` 目录下
2. 运行 `python /mnt/d/paper-factor-data/scripts/import_new_data.py --check` 预览
3. 运行 `python /mnt/d/paper-factor-data/scripts/import_new_data.py` 执行导入

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
| `TUSHARE_TOKEN` | Tushare token（指数行情/成分股/估值） | 指数类因子需要 |

---

## Step 5 — 验证

```bash
# 数据完整性检查
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/日线/测试/stock_data/daily/trade_dates.json')); print(f'日线测试: {len(sl)}只×{len(td)}天')"
python3 -c "import json; sl=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/stock_list.json')); td=json.load(open('/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/trade_dates.json')); print(f'分钟测试: {len(sl)}只×{len(td)}天')"

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
python /mnt/d/paper-factor-data/scripts/import_new_data.py --check      # 预览原始数据
python /mnt/d/paper-factor-data/scripts/import_new_data.py              # 执行导入
```

---

## 常见问题

**Q: 只想看已有因子结果，不跑计算？**
A: 直接看 `/mnt/d/paper-factor-data/数据仓库/因子产出/全量/{DATE}/已完成/{report}/{factor}/` 下的 `.parquet` 和 `.decile.png`（还没算完的在同日期 `未完成/` 下）。

**Q: 跑 `/factor` 需要研报 PDF 吗？**
A: 需要（放到 `/mnt/d/paper-factor-data/papers/inbox/`，支持 `.pdf` 和 `.md`）。

**Q: 数据从哪里来？**
A: 本项目不提供原始数据。需要从已有数据源（如聚宽、Tushare、券商数据终端）获取，通过 `import_new_data.py` 导入。

**Q: 测试数据为什么是 300 只×300 天？**
A: 固定不变。测试数据只用于验证因子代码逻辑正确性，不用于最终结果。全量计算使用全量数据目录（5435 只×全历史）。

### WSL 跑全量因子时终端闪退

**症状**：跑 `run_all.py` / `run_factor_full.py` 时，WSL 里的 Ubuntu 终端窗口突然关闭，Windows 本身还在。

**先分清两种情况**：
- 所有 WSL 终端一起关闭 → 多半是 WSL 虚拟机被 Windows 终止（内存超限）。
- 只有当前任务/窗口消失，其他终端正常 → 多半是 Claude Code 的后台任务回收器杀了后台 shell。

**常见原因**（按可能性排序）：

1. **WSL 内存被跑满**：截面因子每个 worker 会缓存全市场约 10GB；WSL2 默认最多使用 Windows 内存的约 50%，超出后 Windows 可能直接停掉整个发行版。默认 `FACTOR_FULL_N_WORKERS=4` 在 16GB/32GB 机器上都可能超限。
2. **Claude Code 后台任务回收器**：后台 shell + 内存压力 + 长时间无真人交互时，会静默杀掉后台任务。2026-09-25 的跑批就是这样在 54 分钟被杀掉的。
3. **磁盘写满**：WSL 的 vhdx 或数据盘（D/G）空间不足，写入 parquet 时失败，终端退出。
4. **GPU 驱动崩溃**：深度学习因子训练时驱动异常，桌面/终端可能直接退出。

**排查（WSL 内）**：

```bash
free -h
dmesg -T | grep -iE 'oom|out of memory|killed process' | tail -20
df -h / /mnt/d /mnt/g 2>/dev/null
grep -n "PRESSURE_REAP\|FACTOR_FULL_N_WORKERS" ~/.claude/settings.json
```

**排查（Windows PowerShell）**：

```powershell
Get-Content $env:USERPROFILE\.wslconfig
Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory
Get-WinEvent -LogName System -MaxEvents 100 |
  Where-Object { $_.Message -match 'WSL|vmmem|memory' } |
  Select TimeCreated,Message
```

**修复**：

- 按机器内存设置 worker 数（截面因子单 worker 约 10GB）：
  - 16GB 内存：`FACTOR_FULL_N_WORKERS=1`
  - 32GB 内存：`FACTOR_FULL_N_WORKERS=2`
- 关闭后台任务回收器：`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`
- 这两项已写进本仓库的 `.claude/settings.json` → `env`，另一台机器 `git pull` 后
  重启 Claude Code CLI 即自动生效。若那台机器内存小于 32GB，把
  `FACTOR_FULL_N_WORKERS` 改成 `1`（16GB）或先按 24GB 档验证。
- 在 `%USERPROFILE%\.wslconfig` 给 WSL 留足内存（例如 32GB 机器设 `memory=24GB`），但不要超过物理内存的 75%。
- 清理 WSL vhdx 和数据盘空间；深度学习因子异常时检查 `nvidia-smi` 与 `dmesg` 里的 GPU Xid 错误。

> 本机实测：2026-09-25 的 `run_all.py` 跑到第 19 个截面因子时被杀（无内核 OOM，cgroup 峰值 30.75GB/31.2GB）。设置上述两项后重跑，49/49 成功。