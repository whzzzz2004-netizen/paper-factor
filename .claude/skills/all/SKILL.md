---
name: all
description: 全量/增量运行所有因子（串联，自动重扫直到队列清空）
---

# /all — 全量/增量运行所有因子（自动重扫）

启动 `run_all.py` 批量运行所有因子（全量计算 + 评估 + 绘图 + Barra）。
**自动重扫**：处理完一轮后自动重扫目录，如果 `/factor` 在此期间 deploy 了新因子，继续处理；队列清空后自动退出。

## 用法

```bash
# 默认当天日期子目录
python3 scripts/run_all.py

# 指定日期子目录
python3 scripts/run_all.py 2026-08-09

# 指定研报
python3 scripts/run_all.py --report 研报名

# 强制重跑
python3 scripts/run_all.py --force

# 仅查看计划
python3 scripts/run_all.py --dry-run
```

## 说明

- 扫描 `/mnt/d/paper-factor-data/数据仓库/因子产出/全量/{DATE}/` 下所有因子，逐个运行 `run_factor_full.py`
- 含评估（IC/IR）、十分组收益图、Barra 风险分析、LLM 审查
- 自动检测新交易日做增量更新
- `FACTOR_LOOKBACK_CAP=99999` 已内置，约等于无上限
- **自动重扫**：处理完一轮后自动重扫，捡新因子 → 继续处理 → 队列清空后退出

## 执行

**必须重定向到日志文件**，不要把 run_all 的 stdout 直接留在对话里（长跑、且个别分钟因子日志可达数千行）：

```bash
setsid python3 scripts/run_all.py {args} > /tmp/run_all.log 2>&1 < /dev/null &
```

- run_all 自身只打印**每个因子的摘要行**（`📈 进度 N/M`）+ 最终汇总，子进程日志只写 `/tmp/{因子}.run.log`，不回显。
- **实时进度**：`cat /tmp/run_all_progress.json`（每个因子结束覆盖写一次：轮次/进度/成功/失败/跳过/失败清单/耗时）。
- **失败原因**：进度文件里的 `failures[]`，或看该因子的 `/tmp/{因子}.run.log` 末尾。
- **判断是否卡死**（而非慢）：对比进度文件 `updated_at` 是否长期不变；有些因子本来就要跑很久，**不要因为慢就杀它**。

## ⚠️ 失败因子处理（重要）

**单个因子失败不会中断整条队列**，会记入 `total_fail` 与进度文件的 `failures[]`，然后继续下一个。

**断点续跑**：`find_pending_factors` 用「有没有 parquet」判状态——已成功的因子重跑时直接跳过。所以修复失败的因子后，**直接再跑一次 `run_all.py` 同一天**即可，不会重算已完成的。

全量跑完（`🏁 全部完成` 且 `队列已清空`）后，逐因子检查全量产出：parquet 是否存在、非空率是否正常（minute/daily 应 >60%，cross_section 应 >50%）、是否有 decile.png。失败的因子 → 派 agent 分析根因 → 修核心函数 → 重新 `test-and-export` + `deploy-to-full` → 重跑该因子全量。

常见失败模式：
- **数据 NaN**（如非行情列空）→ 检查 `非行情数据/` 是否有效
- **模板 bug**（如 cross_section 索引列读取）→ 修模板后清 `_template_cache`
- **因子算法**（全量规模退化、日期对齐、O(N³) 过慢）→ 改核心函数

## 不跑全量时

用户可能只要求"确保全量能正常产出"（不实际跑完）：此时修复因子后跑 `run_factor_full.py` 单因子验证非空率正常即可，不必等全量完整结束。