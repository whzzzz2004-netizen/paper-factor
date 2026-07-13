# 分钟截面因子模板

## 函数签名
```python
def calc_factor_minute_raw(df, stock):
    """单股票分钟数据 → dict {"因子名": 原始值}

    Example: return {"MinuteRealizedVol": 0.015}
    """

def cross_section_transform(all_values):
    """dict {股票代码: 原始值} → dict {股票代码: {"因子名": 处理后值}}

    注意：all_values 的值是标量（不是嵌套 dict）。

    Example input:  {"000001": 0.015, "000002": 0.022, "600519": 0.008}
    Example output: {"000001": {"MinuteRealizedVol": 0.3},
                     "000002": {"MinuteRealizedVol": 0.8},
                     "600519": {"MinuteRealizedVol": 0.1}}
    # 返回标量也可以（会自动包装成 {"因子名": 值}）：
    Example output: {"000001": 0.3, "000002": 0.8, "600519": 0.1}
```

## 可用数据列（8 列，同分钟线）

<!-- MINUTE_COLUMNS -->
  - close: 收盘价（日线收盘价）
  - factor: 复权因子（前复权因子）
  - high: 最高价（日线最高价）
  - low: 最低价（日线最低价）
  - money: 成交额
  - open: 开盘价（日线开盘价）
  - volume: 成交量（单位：股）
<!-- /MINUTE_COLUMNS -->

## 特殊约束

1. 分两阶段：Phase 1 算每只股票的原始值，Phase 2 做截面变换
2. `cross_section_transform` 接收 `all_values`（dict {股票代码: 原始值}），输出标准化/排名后的值
3. 数据源是 `minute_by_date` 格式（按日期文件夹组织，每文件含当天所有股票），不是 per-stock
4. 无 `pct_chg`、`pre_close` 等日线列
