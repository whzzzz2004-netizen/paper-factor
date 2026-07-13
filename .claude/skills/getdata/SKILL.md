---
name: getdata
description: 从远程E盘同步增量数据（SMB直连），自动检测新列 → 更新 schema → 更新 prompt
---

# /getdata — 增量数据同步

从远程 E 盘 (Windows SMB 共享) 同步增量数据，转换为 per-stock parquet 格式。
基于 `smbclient`（不需要挂载，不需要 sudo，不会卡死）。

## 流程

**每次同步必须按此顺序执行：**

```
1. python3 scripts/sync_data.py --check
   → 看远程有无新数据

2. 用 smbclient 直接读取远程 数据说明.txt，理解当前数据布局：
   smbclient //192.168.1.13/E -U pc%123456 -c "get 数据说明.txt /tmp/数据说明.txt"
   → Read /tmp/数据说明.txt
   → 如果说明里提到新文件夹/新文件格式 → 先修改 sync_data.py 再继续

3. python3 scripts/sync_data.py
   → 执行同步（自动检测新列、注册 schema、更新 prompt）

4. 如输出 NEW_COLUMNS_DETECTED → 根据说明理解新列含义 → 更新 schema + factor_field_schema → --update-prompts-only
```

## 使用方式

```bash
# 查看远程状态（推荐先执行）
python3 scripts/sync_data.py --check

# 自动同步增量
python3 scripts/sync_data.py

# 全量同步（覆盖所有数据）
python3 scripts/sync_data.py --full

# 只看变更不执行
python3 scripts/sync_data.py --dry-run
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `scripts/sync_data.py` | 同步脚本（全部逻辑） |
| `data/schema.json` | 字段注册表，定义所有可用列及其来源 |
| `git_ignore_folder/factor_implementation_source_data/` | 全量数据目录 |
| `git_ignore_folder/factor_implementation_source_data_1000/` | 测试数据目录（300只） |
| `*/factor_field_schema.json` | LLM 数据可用性检查用的字段含义表，新列自动同步 |

## 远程数据源

| 数据 | 远程路径 | 说明 |
|------|----------|------|
| 日线(基础) | `E:\market_daily_daily_new\` | 按日期 parquet, 含价量+EMA |
| 日线(财务) | `E:\dailyData.parquet` | 全量 parquet, 含基本面+财务 |
| 分钟线 | `E:\market_minute_daily_new\` | 按日期 parquet |
| 行业分类 | `E:\jq_swIndu_comp.csv` | 申万一级行业 |
| 集合竞价换手率 | `E:\jhjjHsl.csv` | jhjj_hsl 列数据 |

## 新列与新数据源

`sync_data.py` 自动完成：
1. 同步开始时下载远程 `数据说明.txt` → `data/数据说明.txt`
2. 扫描远程下载文件的列名，跟 `data/schema.json` 对比发现新列
3. 新列注册到 schema.json（description 暂填列名本身）和 `factor_field_schema.json`（含义待补充）
4. 更新所有 prompt 文件（通过 `<!-- DAILY_COLUMNS -->` / `<!-- MINUTE_COLUMNS -->` 标记位）
5. 如有新列，输出 `⚠️ NEW_COLUMNS_DETECTED: [...]`

### Agent 必做步骤

**同步完成后**，如果出现 `NEW_COLUMNS_DETECTED`：

1. 读 `/tmp/数据说明.txt`（已在步骤 2 下载），理解每个新列的实际含义
2. 更新 `data/schema.json`：将新列的 `description` 改为实际含义
3. 更新两个 `factor_field_schema.json`：`short_name` 改为中文含义，`note` 更新为完整说明
4. 运行 `python3 scripts/sync_data.py --update-prompts-only` 刷新 prompt 文件

**数据说明提到了新文件夹/新文件格式时**（sync_data.py 暂不支持的）：
- Agent 需要修改 `sync_data.py`，新增对应的同步函数
- 将新函数接入 `main()` 流程
- 参考现有函数（如 `sync_daily_incremental`）的模式

### 被更新的 prompt 文件

| 文件 | 日线 | 分钟 |
|------|------|------|
| `rdagent/components/coder/factor_coder/prompts.yaml` | ✅ DAILY_COLUMNS | ✅ MINUTE_COLUMNS |
| `.claude/skills/factor/knowledge/daily.md` | ✅ DAILY_COLUMNS | |
| `.claude/skills/factor/knowledge/cross_section.md` | ✅ DAILY_COLUMNS | |
| `.claude/skills/factor/knowledge/deep_learning.md` | ✅ DAILY_COLUMNS | |
| `.claude/skills/factor/knowledge/minute.md` | | ✅ MINUTE_COLUMNS |
| `.claude/skills/factor/knowledge/minute_cs.md` | | ✅ MINUTE_COLUMNS |

### 描述生成规则

列描述优先使用 `factor_field_schema.json` 的 `short_name` + `note`（含中文含义与备注），
fallback 到 `data/schema.json` 的 `description`。

## 连接问题

如果 smbclient 连不上：
1. 检查网络：`ping 192.168.1.13`
2. 检查 Windows 防火墙是否阻止 SMB (端口 445)
3. 检查 E 盘共享是否正常：`smbclient -L //192.168.1.13 -U pc`
