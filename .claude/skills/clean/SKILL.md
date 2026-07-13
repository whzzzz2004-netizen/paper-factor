---
name: clean
description: 删除所有因子产出（测试因子 / 全量因子 / Python 缓存）
---

# /clean — 删除所有产出

删除所有因子产出，包括测试因子、全量因子和 Python 字节码缓存。

## 效果

| 项目 | 路径 | 操作 |
|------|------|------|
| 全量因子 | `factor_outputs/文献因子_全量/` | 清空 |
| 测试因子 | `factor_outputs/literature_reports/` | 清空 |
| 已处理名单 | `factor_outputs/processed_reports.json` | 删除 |
| label 缓存 | `factor_implementation_source_data/label_full.parquet` | 删除 |
| Python 缓存 | `__pycache__` / `*.pyc` | 删除 |

## 使用

```bash
/clean
```

## 注意事项

- 不删除 `papers/inbox/` 中的源 PDF
- 不删除数据目录 `factor_implementation_source_data/` 中的原始行情数据
- 跑完 `/clean` 后需要重新跑 `start`
