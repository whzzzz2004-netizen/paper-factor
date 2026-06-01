# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Adjusted daily price, volume, and turnover data.                 |
| "minute_pv.h5" | Minute-level OHLCV/VWAP data.                                    |


# For different data, We have some basic knowledge for them

## Daily price and volume data (daily_pv.h5)
MultiIndex: ['datetime', 'instrument']

Price/volume columns:
$open: open price of the stock on that day.
$close: close price of the stock on that day.
$high: high price of the stock on that day.
$low: low price of the stock on that day.
$volume: volume of the stock on that day.
$factor: adjustment factor (复权因子) of the stock on that day.
$pct_chg: price change percentage (涨跌幅%) of the stock on that day.
$pre_close: previous close price (前收盘价) of the stock on that day.
$turnover_rate: daily turnover rate (换手率%). NOTE: the column name is $turnover_rate, NOT $turnover.

Fundamental factor columns (111 pre-computed factors):
$盈利因子1~$盈利因子14: profitability factors
$现金流因子1~$现金流因子9: cash flow factors
$价值因子1~$价值因子12: value factors
$成长因子1~$成长因子16: growth factors
$运营因子1~$运营因子22: operational factors
$杠杆因子1~$杠杆因子8: leverage factors
$质量因子1~$质量因子15: quality factors
$其他因子1~$其他因子13: other factors (cumulative-based)
$股本因子1~$股本因子2: share capital factors

## Minute price and volume data (minute_pv.h5)

MultiIndex: ['datetime', 'instrument']

Columns:
$open: minute open price.
$close: minute close price.
$high: minute high price.
$low: minute low price.
$volume: minute traded volume.
$vwap: minute volume weighted average price.
$factor: adjustment factor (复权因子).
$return: minute return (分钟收益率).
