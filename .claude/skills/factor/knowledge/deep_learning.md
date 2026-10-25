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

## 可用数据列（同日线）

> 完整列名及含义由主进程通过 `show-columns --type daily_single` 预跑后内联进 Phase 2 prompt（`{DAILY_COLS_TEXT}`），此处不再重复列出。
> （deep_learning 用的就是日线+非行情那套列，**没有** `--type deep_learning` 这个参数值。）

## 额外可用数据（框架已注入，不是 parquet 列，**不算缺字段**）

> `INDUSTRY_DICT` 与 `get_ts_index_data(...)` 由模板自动注入，函数内直接可用。它们**不出现在 `show-columns` 输出里**，
> 但属于合法可用数据——需要行业分类 / 指数数据时**不要判缺列**。
> `show-columns` 命令的输出末尾也会重复打印这一段，以那次输出为准。

### `INDUSTRY_DICT` — 申万一级行业分类

- 字典：`INDUSTRY_DICT[股票代码] = 行业名`（如 `INDUSTRY_DICT["000001"] == "银行I"`）
- 申万一级行业标准，共 31 个行业；测试集 292/300 只、全量 5207 只
- 用法：`industry = INDUSTRY_DICT.get(stock, "未知")`
- 用途：行业中性化、行业 embedding、行业分组特征


### `get_ts_index_data(symbol, data_type)` — Tushare 指数数据（行情 / 成分股 / 估值）

首次调用联网下载并缓存为 parquet，之后读缓存。`symbol` 用 **Tushare 代码**（不是聚宽格式）：

常用：`000300.SH`(沪深300)、`000905.SH`(中证500)、`000016.SH`(上证50)、`000852.SH`(中证1000)、
`000906.SH`(中证800)、`000688.SH`(科创50)、`000001.SH`(上证指数)、`399001.SZ`(深证成指)、
`399006.SZ`(创业板指)、`932000.CSI`(中证2000)。

| data_type | 返回 | 说明 |
|---|---|---|
| `'price'` | DatetimeIndex 升序，列 `open/high/low/close/pre_close/change/pct_chg/vol/amount` | 指数日线行情 |
| `'components'` | 列 `stock`(6位代码)/`weight` | 截至 `end_date` 最近一期（月末）成分股 |
| `'valuation'` | 列 `pe/pe_ttm/pb/turnover_rate/total_mv/float_mv` 等 | 指数每日估值 |
| `'global'` | 日线行情 | 全球指数（symbol 如 `HSI`/`SPX`） |

```python
idx = get_ts_index_data('000300.SH', 'price')               # 沪深300 行情
stocks = get_ts_index_data('000905.SH', 'components')       # 中证500 成分股（截至 end_date）
pe = get_ts_index_data('000300.SH', 'valuation')['pe_ttm']  # 沪深300 每日 PE_TTM
```

- 用途：市场收益、超额收益、Beta、相对强弱、CAPM、指数估值、成分股筛选
- 需要 `TUSHARE_TOKEN` 环境变量（`.env` 已配置）
- ⚠️ **前瞻约束**：默认 `end_date` 是本地数据最后交易日；按 T 日计算时必须传
  `end_date=trade_date`，或自行截断 `idx[idx.index <= trade_date]`。成分股同理，
  不要用今天的成分股回填历史（幸存者偏差）

## 特殊约束

1. GPU 仅 4GB，必须分 batch 推理：`predict_batch` 内对 batch 切块（如 max_batch=500）
2. `train_model` 接收所有股票数据，每年只调用一次
3. `predict_batch` 接收 `{stock: DataFrame}`，返回 `(factor_name, {stock: value})`
4. 注意显存管理：`torch.cuda.empty_cache()`，控制 batch_size
5. 保留旧版 `predict` 作为 fallback（框架自动检测）
