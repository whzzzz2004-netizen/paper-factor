# 截面因子模板

## 函数签名
```python
def calc_factor_cross_section(all_data, trade_date):
    # all_data: dict {股票代码: DataFrame}
    # trade_date: 目标交易日
    # 返回: dict {股票代码: {"因子名": 值}}
```

## 可用数据列（同日线）

> 完整列名及含义由主进程通过 `show-columns --type daily_single` 预跑后内联进 Phase 2 prompt（`{DAILY_COLS_TEXT}`），此处不再重复列出。
> （cross_section 用的就是日线+非行情那套列，**没有** `--type cross_section` 这个参数值。）

## 额外工具

- `INDUSTRY_DICT[stock]` → 申万一级行业名
- `get_jq_data(symbol, data_type)` → 指数行情/成分股

## 特殊约束

1. `all_data` 是 dict，用 `stock_df = all_data[stock]` 访问单只股票
2. 可以访问所有股票的 lookback 窗口数据，做横截面计算
3. 返回 dict {股票代码: {"因子名": 值}}，需要遍历 all_data 的 key
4. 支持行业中性化：用 `INDUSTRY_DICT` 查询股票所属行业
5. 模板用 ProcessPoolExecutor，N_WORKERS 默认 4
