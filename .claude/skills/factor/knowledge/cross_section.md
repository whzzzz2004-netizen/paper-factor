# 截面因子模板

## 函数签名
```python
def calc_factor_cross_section(all_data, trade_date):
    # all_data: dict {股票代码: DataFrame}
    # trade_date: 目标交易日
    # 返回: dict {股票代码: {"因子名": 值}}
```

**窗口行数**：每只股票的 DataFrame 共 `lookback + 1` 行，**最后一行 = T 日**。
所以 `close.pct_change(lookback)` / `close.shift(lookback)` 在末尾恰好有值，
`iloc[0]` 是 T-lookback 日、`iloc[-1]` 是 T 日。**不要**写 `len(df) < lookback + 1` 之类的
行数过滤去跳过股票——行数天然够，多写过滤会把所有股票滤空、产出全 NaN。
（与 `daily` / `deep_learning` 模板口径一致。）

## 可用数据列（同日线）

> 完整列名及含义由主进程通过 `show-columns --type daily_single` 预跑后内联进 Phase 2 prompt（`{DAILY_COLS_TEXT}`），此处不再重复列出。
> （cross_section 用的就是日线+非行情那套列，**没有** `--type cross_section` 这个参数值。）

## 额外可用数据（框架已注入，不是 parquet 列，**不算缺字段**）

> `INDUSTRY_DICT` 由模板自动加载，函数内直接可用。它**不出现在 `show-columns` 输出里**，
> 但属于合法可用数据——需要行业分类时**不要判缺列**。
> `show-columns` 命令的输出末尾也会重复打印这一段，以那次输出为准。

### `INDUSTRY_DICT` — 申万一级行业分类

- 字典：`INDUSTRY_DICT[股票代码] = 行业名`（如 `INDUSTRY_DICT["000001"] == "银行I"`）
- 数据源：`{日线数据目录}/industry.json`，申万一级行业标准，共 31 个行业
  （银行I、房地产I、医药生物I、电子I、计算机I、食品饮料I …，带 `I` 后缀表示一级行业）
- 用法：`industry = INDUSTRY_DICT.get(stock, "未知")`
- 用途：行业中性化、行业分组统计、行业内排名、行业动量、行业轮动
- 覆盖：测试集 292/300 只、全量 5207 只；缺失的股票用 `.get(stock, "未知")` 兜底

**用法示例**（行业分组是核心逻辑时，正常实现）：

```python
def calc_factor_cross_section(all_data, trade_date):
    vals = {}
    for stock, df in all_data.items():
        vals[stock] = _my_raw_value(df)
    s = pd.Series(vals)
    ind = pd.Series({k: INDUSTRY_DICT.get(k, "未知") for k in s.index})
    # 行业内排名（行业轮动/行业内选股类因子的核心逻辑）
    rank_in_ind = s.groupby(ind).rank(pct=True)
    return {k: {"因子名": v} for k, v in rank_in_ind.items()}
```

> ⚠️ **纯「后处理」性质的截面操作（排名/标准化/行业中性化）跳过**——
> 只输出个股原始值，最后统一做。这里写示例是因为**行业分组本身是因子核心逻辑**
> （如行业动量、行业轮动、行业偏离度）时需要用到 `INDUSTRY_DICT`，而非鼓励做后处理。

**指数行情、指数成分股等其他在线数据本地不可用**，需要时按缺字段处理（写 `{name}.missing.json`），不要用相近数据代理。

## 特殊约束

1. `all_data` 是 dict，用 `stock_df = all_data[stock]` 访问单只股票
2. 可以访问所有股票的 lookback 窗口数据，做横截面计算
3. 返回 dict {股票代码: {"因子名": 值}}，需要遍历 all_data 的 key
4. 支持行业中性化：用 `INDUSTRY_DICT` 查询股票所属行业
5. 模板用 ProcessPoolExecutor，N_WORKERS 默认 4
