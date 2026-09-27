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

**指数行情、指数成分股等其他在线数据本地不可用**，需要时按缺字段处理（写 `{name}.missing.json`），不要用相近数据代理。

## 字段纪律

**字段只认 `show-columns --type daily_single` 的输出。**
- 输出里有 → 直接用（列名照抄，不要臆造）
- 输出里没有，但能由清单内的列**精确**推导 → 推导着用（如 `close.pct_change()`、`close * factor`）
- 其余情况 → **判缺列**，写 `{name}.missing.json` 后返回
- **缺字段不能近似、不能代理**：不许把缺的列假设成常数，不许用语义相近的列代替
- **不要**为了找某个字段去翻数据仓库 / 源码 / memory / barra / 备份——字段清单只由 show-columns 决定，翻别处纯浪费 token

## 特殊约束

- T 日 = df.iloc[-1]，df 共 lookback_days 行，最后一行是 T 日
- 日频窗口必须是整数交易日数
- 日线收益率用 `close.pct_change()`（复权价口径 `close * factor`）
- **`df.index.date` 返回 ndarray**，没有 `.isin()` 方法，要判断日期是否属于某集合用 `np.isin(date_arr, list)`
- `df.index.date` 不放循环内（每次访问重建整个数组）
- 布尔序列 shift() 后必须 fillna(False)
- np.inf/-np.inf → np.nan
- 禁止未来数据

## 模板代码特点

日线模板用 `joblib.Parallel(n_jobs=N_JOBS, backend="loky")` 并行算股票，单股票顺序算交易日。
