# 截面因子模板

## 函数签名
```python
def calc_factor_cross_section(all_data, trade_date):
    # all_data: dict {股票代码: DataFrame}
    # trade_date: 目标交易日
    # 返回: dict {股票代码: {"因子名": 值}}
```

## 可用数据列（36 列，同日线）

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

## 额外工具

- `INDUSTRY_DICT[stock]` → 申万一级行业名
- `get_jq_data(symbol, data_type)` → 指数行情/成分股

## 特殊约束

1. `all_data` 是 dict，用 `stock_df = all_data[stock]` 访问单只股票
2. 可以访问所有股票的 lookback 窗口数据，做横截面计算
3. 返回 dict {股票代码: {"因子名": 值}}，需要遍历 all_data 的 key
4. 支持行业中性化：用 `INDUSTRY_DICT` 查询股票所属行业
5. 模板用 ProcessPoolExecutor，N_WORKERS 默认 4
