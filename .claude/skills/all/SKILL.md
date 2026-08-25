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

```bash
python3 scripts/run_all.py {args}
```

## ⚠️ 失败因子处理（重要）

**全量跑完（所有其他因子都完成后）**，若发现有因子失败或产出异常（parquet 全空/非空率过低/无图像），**必须派 agent 介入修复**，不能直接接受：

1. **等 run_all 完全结束**（`🏁 全部完成` 且 `队列已清空`）
2. 逐因子检查全量产出：parquet 是否存在、非空率是否正常（minute/daily 应 >60%，cross_section 应 >50%）、是否有 decile.png
3. **失败的因子 → 派 agent 分析根因**（数据问题 / 模板问题 / 因子算法问题）→ 修复核心函数 → 重新 `test-and-export` + `deploy-to-full` → 重跑该因子全量（`run_factor_full.py`）
4. 修复后重新验证非空率正常，全部通过才算完成

常见失败模式：
- **数据 NaN**（如非行情列空）→ 检查 `非行情数据/` 是否有效
- **模板 bug**（如 cross_section 索引列读取）→ 修模板后清 `_template_cache`
- **因子算法**（全量规模退化、日期对齐、O(N³) 过慢）→ 改核心函数

## 不跑全量时

用户可能只要求"确保全量能正常产出"（不实际跑完）：此时修复因子后跑 `run_factor_full.py` 单因子验证非空率正常即可，不必等全量完整结束。