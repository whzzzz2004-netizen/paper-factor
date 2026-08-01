# 在新电脑上完整复现本项目

> 目标：在老板/其他电脑上完整复现 paper-factor 项目，能跑 `/factor`、全量计算、每日更新。
>
> **核心结论：代码用 git clone，数据不用搬 —— 远程 E 盘（192.168.1.13）已有完整数据镜像，直接读。**

---

## 复现需要什么

| 东西 | 在哪 | 怎么获取 |
|------|------|----------|
| 代码（scripts/rdagent/.claude/templates） | git 仓库 | `git clone` |
| 数据源（日线/分钟 parquet，~96G） | `git_ignore_folder/`（gitignore 排除） | **读远程 E 盘镜像** |
| 因子产出（测试/全量/每日更新） | `git_ignore_folder/factor_outputs/` | **读远程 E 盘镜像** |
| 研报 PDF（25个，`papers/**/*.pdf`） | 本地 `papers/`（gitignore 排除） | 手动上传/拷贝 |
| 密钥配置 | `.env`（gitignore 排除） | 手动创建 |

---

## 前置条件

- **能访问远程 E 盘**：`192.168.1.13`（内网），账号 `pc` / 密码 `123456`
- Python ≥ 3.10（建议 conda 环境）

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
pip install -e .          # 完整依赖（pyproject.toml：jqdatasdk 等）
```

---

## Step 3 — 挂载远程 E 盘

### Windows（推荐，映射网络驱动器）

资源管理器 → 此电脑 → 右键 → **映射网络驱动器**：

- 文件夹：`\\192.168.1.13\E`
- 盘符：选 `Z:`（或任意空闲盘符）
- 连接凭据：`pc` / `123456`

### WSL / Linux（CIFS 挂载）

```bash
sudo mkdir -p /mnt/remote_e
sudo mount -t cifs //192.168.1.13/E /mnt/remote_e \
  -o vers=3.0,username=pc,password=123456,uid=$(id -u),gid=$(id -g)
```

> 开机自动挂载：写入 `/etc/fstab`（`nofail` 防止开机卡住）。

---

## Step 4 — 配置数据目录

系统会自动探测数据目录（候选顺序见 `_detect_data_dir()`），但建议显式设置环境变量，最稳妥：

```bash
# Windows 挂了 Z: 盘
export FACTOR_DATA_DIR="Z:/_paper_factor_unified/factor_implementation_source_data"

# 或 CIFS 挂载到 /mnt/remote_e
export FACTOR_DATA_DIR="/mnt/remote_e/_paper_factor_unified/factor_implementation_source_data"
```

> 不设也能跑：`_detect_data_dir()` 会依次探测
> `FACTOR_DATA_DIR` → `RDAGENT_FACTOR_DATA_DIR` → 本地 `git_ignore_folder/...` →
> `/mnt/remote_e/_paper_factor_unified/...` → `E:\...` → `Z:\...` → `\\192.168.1.13\E\...`。
> 生成的全量 `.code.py` 也内置了同样的多级降级链。

---

## Step 5 — 配置 `.env`（密钥）

复制模板并填写：

```bash
cp .env.example .env
```

`.env` 里需要的项（**敏感，gitignore 已排除，不会推上 GitHub**）：

| 变量 | 用途 | 是否必须 |
|------|------|----------|
| `FACTOR_DATA_DIR` | 数据目录（.env.example 里的示例） | 建议设置 |
| `OPENAI_API_KEY` / `OPENAI_API_BASE` | LLM 因子审查（ModelVerse） | 可选，不设则跳过审查 |
| `JQDATA_USERNAME` / `JQDATA_PASSWORD` | 聚宽取数（模板 `get_jq_data`） | 按需 |
| `TUSHARE_TOKEN` | Tushare 取数 | 按需 |

> **注意**：`.env` 只在 rdagent 子系统（`rdagent/core/conf.py`）import 时自动加载（LLM/模型配置）。
> `scripts/` 下的 `/factor` 主流程**不读 `.env`**，它们靠 `os.environ`（shell 导出）里的
> `FACTOR_DATA_DIR` 等变量。所以环境变量要在 shell 里 export（见 Step 4）。

---

## Step 6 — 验证

```bash
python scripts/sync_data.py --check        # 远程连接正常，日线/分钟已最新
python scripts/claude_factor_helper.py scan-pending   # 扫描待处理研报
python scripts/claude_factor_helper.py show-columns   # 查看可用数据列
```

---

## 常见操作速查

```bash
# 处理新研报（端到端）
#   用 Claude Code 打开项目，执行 /factor

# 全量计算 / 增量更新
python scripts/run_all.py                      # 默认当天日期子目录
python scripts/daily_update.py                 # 每日增量更新

# 同步到远程
python scripts/claude_factor_helper.py sync-full --all --date YYYYMMDD
```

---

## 常见问题

**Q: 远程数据镜像过期怎么办？**
A: 运行 `python scripts/sync_data.py`（增量同步），它会从远程 E 盘根目录的
`market_daily_daily_new` / `market_minute_daily_new` 拉取原始数据并重建本地 parquet。
注意：这需要远程 E 盘根目录本身数据是最新的。

**Q: 老板电脑只想看已有因子结果，不想跑计算？**
A: 直接读远程 `\\192.168.1.13\E\paper_factors\文献因子_全量\`（已有全部因子产出含评估图表），
连代码 clone 都不需要。

**Q: 跑 `/factor` 需要研报 PDF 吗？**
A: 处理**新**研报需要 PDF（手动放到 `papers/inbox/`）。已 mark-done 的研报不需要——
因子代码和元数据已存在。
