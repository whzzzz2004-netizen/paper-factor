# 分钟截面因子模板

## 注意：pandas 兼容性
- **禁止使用** `resample('5T')`、`resample('15T')` 等 `'T'` 后缀——pandas 2.3+ 已删除 `'T'` 别名
- 必须用 `resample('5min')`、`resample('15min')` 代替

## 函数签名
```python
def calc_factor_minute_raw(df, stock):
    """单股票分钟数据 → dict {"因子名": 原始值}

    Example: return {"MinuteRealizedVol": 0.015}
    """

def cross_section_transform(all_values):
    """dict {股票代码: 原始值} → dict {股票代码: {"因子名": 处理后值}}

    注意：all_values 的值是标量（不是嵌套 dict）。
    ⚠️ 必须返回 dict，绝不能返回 pd.Series！模板会对返回值调用 .values()/.items()（方法），
       pd.Series 的 .values 是 ndarray 属性、不可调用 → 报 TypeError。

    Example input:  {"000001": 0.015, "000002": 0.022, "600519": 0.008}
    Example output: {"000001": {"MinuteRealizedVol": 0.3},
                     "000002": {"MinuteRealizedVol": 0.8},
                     "600519": {"MinuteRealizedVol": 0.1}}
    # 返回标量也可以（会自动包装成 {"因子名": 值}）：
    Example output: {"000001": 0.3, "000002": 0.8, "600519": 0.1}
```

## 可用数据列（同分钟线）

> 完整列名及含义由主进程通过 `show-columns --type minute` 预跑后内联进 Phase 2 prompt（`{MINUTE_COLS_TEXT}`），此处不再重复列出。（注意：当前 /factor 已禁用 minute_cs 类型，统一走 minute 模板。）

## 特殊约束

1. 分两阶段：Phase 1 算每只股票的原始值，Phase 2 做截面变换
2. `cross_section_transform` 接收 `all_values`（dict {股票代码: 原始值}），输出标准化/排名后的值
3. 数据源是 `minute_by_date` 格式（按日期文件夹组织，每文件含当天所有股票），不是 per-stock
4. 无 `pct_chg`、`pre_close` 等日线列
5. **raw 只算当日单日值，标准化放截面**：`calc_factor_minute_raw` 每只股票在滑动窗口内被调用，只返回该股票当日的原始指标（如当日异动占比、当日日内相关系数）；跨股票的 zscore/排名一律在 `cross_section_transform` 里做。**禁止在 raw 内对单只股票做跨日纵向 mean/std 归一化**（既泄露又超重，测试会超时）。
6. **性能**：分钟数据每股每天约 240 行（MultiIndex[instrument, datetime]）。用 numpy 向量化（groupby+transform、rolling、`.values`），禁止逐分钟 `for` 循环。每个 minute_cs 因子测试约 6~8 分钟，写错一次重试成本高。

## lookback 天数限制（重要）

- **lookback_days 最大 120（约 6 个月）**，即使论文用 1 年也要截断
- 分钟截面模板按天并行，每 chunk 重建进程池 → worker 缓存重置
- 高 lookback 导致：
  - 每 chunk 前 lookback 天都要重新读盘
  - 每个 worker 内存 ≈ lookback × 列数 × 7.8MB
  - 全量 5000 只股票下 OOM 风险增加
- 例外：如果因子只依赖当天分钟数据（不跨天），可以用 lookback_days=1
