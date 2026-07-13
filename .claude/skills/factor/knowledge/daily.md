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

<!-- DAILY_COLUMNS -->
  - EMA10: 10日指数移动均线（10交易日指数移动平均收盘价）
  - EMA20: 20日指数移动均线（20交易日指数移动平均收盘价）
  - EMA5: 5日指数移动均线（5交易日指数移动平均收盘价）
  - adjusted_profit: 扣非净利润（扣除非经常性损益后的净利润，单位：元。按报告期 forward-fill 到每日。）
  - amount: 成交金额(元, 成交量×收盘价近似)
  - circulating_market_cap: 流通市值（单位：亿元，衡量可交易市值）
  - close: 收盘价（日线收盘价）
  - debt_to_asset: 资产负债率（Debt-to-Asset Ratio, 衡量财务杠杆）
  - factor: 复权因子（前复权因子）
  - float_shares: 流通股本（单位：万股，A股市场流通股本）
  - gross_margin: 毛利率（Gross Margin, 衡量产品盈利能力）
  - gross_profit: 毛利润（毛利润（营业总收入 - 营业成本），单位：元。银行等金融公司用营业利润代替。按报告期 forward-fill 到每日。）
  - high: 最高价（日线最高价）
  - jhjj_hsl: 集合竞价换手率（集合竞价时段换手率，单位：%）
  - low: 最低价（日线最低价）
  - market_cap: 总市值（单位：亿元，衡量公司规模）
  - net_margin: 净利率（Net Profit Margin, 衡量综合盈利效率）
  - ocf_per_share: 每股经营现金流（Operating Cash Flow per Share, 衡量现金创造能力）
  - open: 开盘价（日线开盘价）
  - pb: 市净率（Price-to-Book, 衡量估值相对于账面价值）
  - pct_chg: 涨跌幅（单位：%）
  - pe_ttm: 市盈率TTM（滚动市盈率, 衡量估值水平）
  - pre_close: 前收盘价（前一交易日收盘价）
  - profit_yoy: 净利润同比增速（Profit Year-over-Year, 衡量盈利成长性）
  - revenue_yoy: 营收同比增速（Revenue Year-over-Year, 衡量成长性）
  - roa: 总资产收益率（Return on Assets, 衡量资产回报效率）
  - roe: 净资产收益率（Return on Equity, 衡量盈利能力）
  - total_shares: 总股本（单位：万股，包含A股、B股和H股）
  - turnover_rate: 换手率（单位：%）
  - volume: 成交量（单位：股）
<!-- /DAILY_COLUMNS -->

- 索引：DatetimeIndex
- **日线无 `return` 列**，计算收益率用 `pct_chg`（单位是 %，3.5 表示 +3.5%）或 `close.pct_change()`
- 复权价比较：`close * factor`；单日涨跌直接用 `pct_chg`
- **基本面数据可用**：`gross_margin`, `revenue_yoy`, `profit_yoy`, `roe`, `roa`, `pe_ttm`, `pb`, `debt_to_asset`, `ocf_per_credit_score_margin` 等财务指标可直接用于日线因子。财务数据为季度更新，日频为前向填充（财报发布后更新，不变直至下期）

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
