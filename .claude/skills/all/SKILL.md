---
name: all
description: 全量/增量运行所有因子（封装 run_all.py）
---

# /all — 全量/增量运行所有因子

启动 `run_all.py` 批量运行所有因子（全量计算 + 评估 + 绘图 + Barra）。

## 用法

```bash
# 默认当天日期子目录
python3 scripts/run_all.py

# 指定日期子目录
python3 scripts/run_all.py 2026-08-09

# 3 因子并行
python3 scripts/run_all.py --workers 3

# 强制重跑
python3 scripts/run_all.py --force

# 仅查看计划
python3 scripts/run_all.py --dry-run
```

## 说明

- 扫描 `数据仓库/因子产出/全量/{DATE}/` 下所有因子，逐个运行 `run_factor_full.py`
- 含评估（IC/IR）、十分组收益图、Barra 风险分析、LLM 审查
- 自动检测新交易日做增量更新
- `FACTOR_LOOKBACK_CAP=99999` 已内置，约等于无上限

## 执行

```bash
python3 scripts/run_all.py {args}
```

如需指定参数，直接传入：

```bash
python3 scripts/run_all.py --workers 3 --force
```