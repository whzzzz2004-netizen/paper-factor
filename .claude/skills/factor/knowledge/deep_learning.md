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

> `INDUSTRY_DICT` 由模板自动加载，函数内直接可用。它**不出现在 `show-columns` 输出里**，
> 但属于合法可用数据——需要行业分类时**不要判缺列**。
> `show-columns` 命令的输出末尾也会重复打印这一段，以那次输出为准。

### `INDUSTRY_DICT` — 申万一级行业分类

- 字典：`INDUSTRY_DICT[股票代码] = 行业名`（如 `INDUSTRY_DICT["000001"] == "银行I"`）
- 申万一级行业标准，共 31 个行业；测试集 292/300 只、全量 5207 只
- 用法：`industry = INDUSTRY_DICT.get(stock, "未知")`
- 用途：行业中性化、行业 embedding、行业分组特征

**指数行情、指数成分股等其他在线数据本地不可用**，需要时按缺字段处理（写 `{name}.missing.json`），不要用相近数据代理。

## 特殊约束

1. GPU 仅 4GB，必须分 batch 推理：`predict_batch` 内对 batch 切块（如 max_batch=500）
2. `train_model` 接收所有股票数据，每年只调用一次
3. `predict_batch` 接收 `{stock: DataFrame}`，返回 `(factor_name, {stock: value})`
4. 注意显存管理：`torch.cuda.empty_cache()`，控制 batch_size
5. 保留旧版 `predict` 作为 fallback（框架自动检测）
