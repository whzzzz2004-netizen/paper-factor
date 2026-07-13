# 涨停规则

A 股不同板块涨停板比例不同，按股票代码前缀判断：

| 板块 | 代码前缀 | 涨停比例 |
|------|----------|----------|
| 主板 | 60/00/30/20/等 | 10% |
| 科创板 | 688 | 20% |
| 创业板 | 300 | 20% |
| 北交所 | 8 | 30% |
| ST | - | 5%（st 状态） |

## 计算公式

```python
def get_limit_pct(stock):
    if stock.startswith("688"):
        return 0.20
    elif stock.startswith("300"):
        return 0.20
    elif stock.startswith("8"):
        return 0.30
    else:
        return 0.10
```

涨停价 = `pre_close * (1 + limit_pct)`

## 概念

- **一字板**：OHLC 全部等于涨停价（或接近涨停价）
- **非一字板**：close 等于涨停价，但 OHLC 不全等于涨停价
- **判断涨停**：`close >= pre_close * (1 + limit_pct)`
- **跌停判断同理**：`close <= pre_close * (1 - limit_pct)`
