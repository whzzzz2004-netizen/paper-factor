# 日线因子模板

## 函数签名
```python
def calc_factor_single_stock(df, trade_date, stock):
    # df: 单只股票截至 trade_date 的全部日线数据，DatetimeIndex，已排序
    # trade_date: 目标交易日 (pd.Timestamp)
    # stock: 股票代码（如 "600519"）
    # 返回: dict {"因子名": 值}，条件不满足返回 {"因子名": np.nan}
```

## 可用数据列

> 完整列名及含义由主进程通过 `show-columns --type daily_single` 预跑后内联进 Phase 2 prompt（`{DAILY_COLS_TEXT}`），此处不再重复列出。

- 索引：DatetimeIndex
- **日线无 `return` 列**，计算收益率用 `pct_chg`（单位是 %，3.5 表示 +3.5%）或 `close.pct_change()`
- 复权价比较：`close * factor`；单日涨跌直接用 `pct_chg`
- **非行情数据可用**：`gross_margin`, `revenue_yoy`, `profit_yoy`, `roe`, `roa`, `pe_ttm`, `pb`, `debt_to_asset`, `ocf_per_credit_score_margin` 等财务指标可直接用于日线因子。财务数据为季度更新，日频为前向填充（财报发布后更新，不变直至下期）

## 额外工具

- `INDUSTRY_DICT[stock]` → 申万一级行业名（如 "银行I"）
- `INDUSTRY_MEMBERS` 行业成分股字典
- `get_jq_data(symbol, data_type)` → 指数行情/成分股

## 特殊约束

- T 日 = df.iloc[-1]，df 共 lookback_days 行，最后一行是 T 日
- 日频窗口必须是整数交易日数
- `df.index.date` 不放循环内（每次访问重建整个数组）
- 布尔序列 shift() 后必须 fillna(False)
- np.inf/-np.inf → np.nan

## 模板代码特点

日线模板用 `joblib.Parallel(n_jobs=N_JOBS, backend="loky")` 并行算股票，单股票顺序算交易日。
