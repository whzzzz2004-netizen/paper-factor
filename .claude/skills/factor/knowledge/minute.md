# 分钟因子模板

**所有分钟因子一律用 minute 模板**——包括分钟数据上的行业分位/排名等截面操作
（模板内置可选的 `cross_section_transform` 钩子，见下）。没有第二个分钟模板。

## 函数签名

模板有两条路径：

- `calc_factors_one_day(df, stock)` → 非向量化，模板按 LOOKBACK 滑动窗口逐天调用。适合单日截面因子、LOOKBACK≤21
- `calc_factor_series(df, stock)` → 向量化，一次接收全部数据，快 10~30 倍。适合滚动累积/平均类、LOOKBACK 大的因子

**判断原则**：能拆成"先算每日值、再跨日 rolling" → `calc_factor_series`；依赖窗口内全量数据 → `calc_factors_one_day`。
不确定时两种都写（模板会自动优先走向量化）。

## 可选截面钩子：`cross_section_transform`（分钟 + 行业分位/排名时用）

需要在**同一天的股票之间**做运算（行业内分位、全市场排名、行业中性化）时，
**不要换模板、不要改类型**——仍然用 `minute`，额外定义这个函数：

```python
def cross_section_transform(all_values):
    # all_values: {股票代码: 当日原始值}（只含非 NaN）
    # 返回:      {股票代码: 变换后值 或 {"因子名": 值}}
```

模板在拼出「日期 × 全股票」宽表后**逐日**调用它，因此**不额外占内存**
（每天只处理一行，不是把分钟数据全量载入）。

```python
def calc_factor_series(df, stock):
    # 先算 per-stock 原始值
    ...

def cross_section_transform(all_values):
    s = pd.Series(all_values, dtype=float).dropna()
    ind = pd.Series({k: INDUSTRY_DICT.get(k, "未知") for k in s.index})
    rank = s.groupby(ind).rank(pct=True)     # 行业内百分位
    return rank.to_dict()
```

不定义这个函数 → 整段跳过，零开销。

```python
def calc_factors_one_day(df, stock):
    # df: 一批回看窗口内的全部分钟 bar 数据，DatetimeIndex（模板已转换，直接用 df.index）
    # stock: 股票代码
    # 返回: pd.Series, index=datetime.date, values=因子值
    # 用 groupby('_date') 按天聚合，返回所有日期的值（不要只取最后一个）
```

## 注意：pandas 兼容性

- **禁止使用** `resample('5T')`、`resample('15T')` 等 `'T'` 后缀——pandas 2.3+ 已删除 `'T'` 别名
- 必须用 `resample('5min')`、`resample('15min')` 代替

## 可用数据列

> **以你自己跑的 `show-columns --type minute` 输出为准**，那是完整且唯一的字段清单。

- 多天数据用 `df.index.date` 分组
- 字段可能随数据更新而变化，**不要凭记忆假定有哪些列**

## 分钟 · 市场数据模式（当因子需要全市场数据时）

**Step 0：检查预计算文件（测试 + 全量都要）**
```bash
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试/stock_data/minute_by_date/market_minute_return.parquet
ls /mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量/stock_data/minute_by_date/market_minute_return.parquet
```

**Step 1：缺失就立即预计算（先跑再写函数！两个目录都要跑）**
```bash
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/测试"
python /mnt/d/paper-factor-data/scripts/precompute_market_proxy.py --data-dir "/mnt/d/paper-factor-data/数据仓库/行情数据/分钟线/全量"
```

**Step 2：模块级加载（一次性，不要在逐日函数内重复读取）**
```python
MARKET_RETURN = pd.read_parquet(MINUTE_BY_DATE_DIR / "market_minute_return.parquet")
MARKET_RETURN = MARKET_RETURN.groupby(level=0).first()  # 去重

def calc_factors_one_day(df, stock):
    dt_idx = df.index          # DatetimeIndex，模板已转换
    market_ret = MARKET_RETURN.reindex(dt_idx)["market_return"].values
    # ... 后续计算
```

## 额外可用数据（框架已注入，不是 parquet 列，**不算缺字段**）

> `INDUSTRY_DICT` 由模板自动加载，函数内直接可用。它**不出现在 `show-columns --type minute` 输出里**，
> 但属于合法可用数据——需要行业分类时**不要判缺列**。
> 注：分钟线数据目录下没有 `industry.json`，模板会自动回退到同级的「日线」目录读取，已实测可用。

- 字典：`INDUSTRY_DICT[股票代码] = 行业名`（如 `INDUSTRY_DICT["000001"] == "银行I"`）
- 申万一级行业标准，共 31 个行业；测试集 292/300 只、全量 5207 只
- 用法：`industry = INDUSTRY_DICT.get(stock, "未知")`
- 用途：行业中性化、行业分组统计、行业内排名、行业动量、行业轮动

**指数行情、指数成分股等其他在线数据本地不可用**，按缺字段处理，不要用相近数据代理。

## 特殊约束

1. **返回所有日期的值**：groupby('_date') 后每天都要出值，不能只取最后一天
2. 返回 `pd.Series(index=date_series, values=values)`，不是 dict
3. 日内涨跌用 `return` 列（不含隔夜跳空）
4. 无复权概念，直接用 `close * factor`
5. **禁止读日线 parquet 数据列**（分钟模板只提供分钟列）；但 `INDUSTRY_DICT` 是框架注入的额外数据，可以用
6. **禁止分钟级 for 循环**（用向量化操作）
7. **字段只认 `show-columns --type minute` 输出**（外加上面的「额外可用数据」）：有则用；无但能由清单内的列精确推导则推导着用；其余判缺列。**缺字段不能近似、不能代理**，也不要去别处找

## lookback

- 只用当天数据时 lookback_days=1（不是 0）
- 回看天数按日历日估算（不是交易日数）
- **最大 120（约 6 个月）**，即使论文用 1 年也要截断
  - 原因：分钟模板用 fork COW，高 lookback → 主进程加载数据量过大 → OOM

## 模板特点

分钟模板用 fork COW（Copy-On-Write）加载数据：主进程预加载后 fork 子进程共享。N_WORKERS 默认 16，环境变量 FACTOR_N_WORKERS 可覆盖。自动 checkpoint 恢复。
