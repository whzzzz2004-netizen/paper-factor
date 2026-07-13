# 深度学习因子模板

## 函数签名（新版 — 推荐）

```python
def train_model(all_data, trade_date):
    """训练并返回模型。每年只调用一次（框架自动按年分组）。"""
def predict_batch(model, data_dict, trade_date):
    """GPU批量推理 → 返回 (factor_name, {stock: value})。所有股票拼接为一个大batch。"""
```

## 函数签名（旧版 — 兼容回退）

```python
def predict(model, df, trade_date, stock):
    """逐股票推理 → dict {"因子名": 值}（不用GPU batch，慢）"""
```

## 框架行为

- 框架按年份分组交易日，**每年只调用一次 `train_model`**（不再是每交易日一次）
- `predict_batch` 接收所有股票的数据切片 `{stock: DataFrame}`，应在内部堆叠为一个 batch tensor 调用 `model(batch)`
- 如果用户函数定义了 `predict_batch`，框架优先使用；否则回退到旧版逐股票 `predict`
- `predict_batch` 返回 `(factor_name_string, {stock_code: float_value})`

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

## 特殊约束

1. GPU 仅 4GB，必须分 batch 推理：`predict_batch` 内对 batch 切块（如 max_batch=500）
2. `train_model` 接收所有股票数据，每年只调用一次
3. `predict_batch` 接收 `{stock: DataFrame}`，返回 `(factor_name, {stock: value})`
4. 注意显存管理：`torch.cuda.empty_cache()`，控制 batch_size
5. 保留旧版 `predict` 作为 fallback（框架自动检测）
