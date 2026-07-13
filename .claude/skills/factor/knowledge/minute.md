# 分钟因子模板

## 函数签名
```python
def calc_factors_one_day(df, stock):
    # df: 一批回看窗口内的全部分钟 bar 数据，DatetimeIndex
    # stock: 股票代码
    # 返回: pd.Series, index=datetime.date, values=因子值
    # 用 groupby('_date') 按天聚合，返回所有日期的值（不要只取最后一个）
```

## 可用数据列（8 列）

<!-- MINUTE_COLUMNS -->
  - close: 收盘价（日线收盘价）
  - factor: 复权因子（前复权因子）
  - high: 最高价（日线最高价）
  - low: 最低价（日线最低价）
  - money: 成交额
  - open: 开盘价（日线开盘价）
  - volume: 成交量（单位：股）
<!-- /MINUTE_COLUMNS -->

- 多天数据用 `df.index.date` 分组
- **无 `pct_chg`、`pre_close`、基本面等日线列**

## 特殊约束

1. **返回所有日期的值**：groupby('_date') 后每天都要出值，不能只取最后一天
2. 返回 `pd.Series(index=date_series, values=values)`，不是 dict
3. 日内涨跌用 `return` 列（不含隔夜跳空）
4. 无复权概念，直接用 `close * factor`
5. **禁止读日线数据**（如 pe_ttm、market_cap 等日线列不存在）

## lookback

- 只用当天数据时 lookback_days=1（不是 0）
- 回看天数按日历日估算（不是交易日数）

## 模板特点

分钟模板用 fork COW（Copy-On-Write）方式加载数据：主进程预加载后 fork 子进程共享。N_WORKERS 默认 16，环境变量 FACTOR_N_WORKERS 可覆盖。自动 checkpoint 恢复。
