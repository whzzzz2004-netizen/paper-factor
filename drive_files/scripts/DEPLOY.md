# DEPLOY 部署说明 — 给另一台机器的 Claude

目标：把数据仓库从**旧版状态**（含大量空基本面列、价值因子1/2、旧注册）迁移到**新版 getdata 机制**（域自治增量、无 ffill、只保留真实截面因子）。

## 一、要复制的（一次性）

把本文件夹 `scripts\` **完整**复制到目标机的 **`G:\paper-factor-data\scripts\`**。

scripts 下包含：
- `import_new_data.py` — 新版导入引擎（核心）
- `getdata.py` — 一键入口
- `reset_to_clean.py` — 清理旧版数据的脚本
- `deploy_payload/` — 关键的**元数据 json 备份**（见下）

## 二、deploy_payload/ 里每个文件该放哪（目标机）

这些 json 都是脚本运行时依赖的小文本文件，缺失也能由 getdata 自愈，但**建议全部按下面放好**，保证一致。

### A. 根配置（放 G:\paper-factor-data\ 下）
| 备份文件 | 应放到目标机 |
|---------|------------|
| deploy_payload/schema.json | G:\paper-factor-data\schema.json |

### B. 数据仓库状态/因子注册（放 G:\paper-factor-data\数据仓库\ 下）
| 备份文件 | 应放到目标 |
|---------|-----------|
| 数据仓库__import_state.json | 数据仓库\import_state.json |
| …__日线__全量__factor_field_schema.json | 数据仓库\行情数据\日线\全量\factor_field_schema.json |
| …__日线__测试__factor_field_schema.json | 数据仓库\行情数据\日线\测试\factor_field_schema.json |

### C. 日线元数据（放各自 stock_data/daily 下）
| 备份文件 | 对应目标 |
|---------|----------|
| …__日线__全量__stock_data__daily__trade_dates.json | 数据仓库\行情数据\日线\全量\stock_data\daily\trade_dates.json |
| …__日线__全量__stock_data__daily__stock_list.json | …\全量\stock_data\daily\stock_list.json |
| …__日线__全量__stock_data__daily__industry.json | …\全量\stock_data\daily\industry.json |
| …__日线__测试__stock_data__daily__trade_dates.json | 数据仓库\行情数据\日线\测试\stock_data\daily\trade_dates.json |
| …__日线__测试__stock_data__daily__stock_list.json | …\测试\stock_data\daily\stock_list.json |
| …__日线__测试__stock_data__daily__industry.json | …\测试\stock_data\daily\industry.json |

### D. 分钟元数据（放各自 stock_data 下）
| 备份文件 | 对应目标 |
|---------|----------|
| …__分钟线__全量__stock_data__trade_dates.json | …\分钟线\全量\stock_data\trade_dates.json |
| …__分钟线__全量__stock_data__stock_list.json | …\分钟线\全量\stock_data\stock_list.json |
| …__分钟线__全量__stock_data__minute_by_date__trade_dates.json | …\minute_by_date\trade_dates.json |
| …__分钟线__全量__stock_data__minute_by_date__stock_list.json | …\minute_by_date\stock_list.json |
| …__分钟线__测试__stock_data__trade_dates.json | …\分钟线\测试\stock_data\trade_dates.json |
| …__分钟线__测试__stock_data__stock_list.json | …\分钟线\测试\stock_data\stock_list.json |

命名规律：文件名里的 `数据仓库__` → `数据仓库\`，`__` → `\`，即目标路径。

## 三、目标机操作步骤（Claude）

### 前提
1. 装真 Python（非 WindowsApps 商店壳）：`pip install pandas pyarrow numpy`
2. 确认目录结构：G:\paper-factor-data\原始数据 与 G:\paper-factor-data\数据仓库

### 步骤 1：放置元数据
按上面表格把 deploy_payload 的 16 个 json 还原到位。

### 步骤 2：清理旧版非行情因子残留（可选但推荐）
```powershell
cd G:\paper-factor-data\scripts
python reset_to_clean.py
```
会：删 价值因子1/2 源文件、清空非行情 per-stock 列、schema/ff 只留 desc 里的因子、state 清空。

### 步骤 3：跑 getdata
```powershell
cd G:\paper-factor-data\scripts
python getdata.py
```
预期（只有 analyst_coverage + EPS_Predict）：
```
【截面因子】共 2 个
  [analyst_coverage] 新增因子 全量导入：2015-01-05 ~ 2026-09-11 — …
  [EPS_Predict] 新增因子 全量导入：2022-05-01 ~ 2026-08-31 — …
  → 合并写入 5475 只股票
```

## 四、常见陷阱
- `repo_path.txt` 指向不存在的路径 → --update-prompts 静默跳过（不阻塞导入）
- 因子源 index 是 string（object）会自动转 datetime
- 没有这些 json 也能跑（getdata 自重建），只是首次全量
- reset_to_clean.py 支持 `--dry-run` 预览

## 五、注意
deploy_payload 中 trade_dates 等是以「本机当前状态」为准；若目标机数据日期范围不同，getdata 会再扩展（因子源并集）。