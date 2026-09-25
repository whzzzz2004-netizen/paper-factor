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

> **以你自己跑的 `show-columns --type daily_single` 输出为准**，那是完整且唯一的字段清单。
> 主进程也会把该输出内联进 Phase 2 prompt（`{DAILY_COLS_TEXT}`），此处不重复列出。

- 索引：DatetimeIndex
- 字段可能随数据更新而变化（新字段会由 getdata 导入并自动出现在 show-columns 里），**不要凭记忆假定有哪些列**
- 收益率：优先用 `close.pct_change()`（复权价口径 `close * factor`）

## 额外工具

- `INDUSTRY_DICT[stock]` → 申万一级行业名（如 "银行I"）
- `INDUSTRY_MEMBERS` 行业成分股字典
- `get_jq_data(symbol, data_type)` → 指数行情/成分股

## 字段纪律

**字段只认 `show-columns --type daily_single` 的输出。**
- 输出里有 → 直接用（列名照抄，不要臆造）
- 输出里没有 → 判缺列，写 `{name}.missing.json` 后返回
- **不要**为了找某个字段去翻数据仓库 / 源码 / memory / barra / 备份——字段清单只由 show-columns 决定，翻别处纯浪费 token

## 特殊约束

- T 日 = df.iloc[-1]，df 共 lookback_days 行，最后一行是 T 日
- 日频窗口必须是整数交易日数
- `df.index.date` 不放循环内（每次访问重建整个数组）
- 布尔序列 shift() 后必须 fillna(False)
- np.inf/-np.inf → np.nan

## 模板代码特点

日线模板用 `joblib.Parallel(n_jobs=N_JOBS, backend="loky")` 并行算股票，单股票顺序算交易日。
