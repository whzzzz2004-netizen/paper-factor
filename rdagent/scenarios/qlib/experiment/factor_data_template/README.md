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
| "minute_pv.h5"  | Minute-level OHLCV/VWAP data used by the pipeline. |


# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: open price of the stock on that day.
$close: close price of the stock on that day.
$high: high price of the stock on that day.
$low: low price of the stock on that day.
$volume: volume of the stock on that day.
$factor: adjustment factor (复权因子) of the stock on that day.
$market_cap: total market capitalization (总市值) of the stock on that day.
$industry_sw: Shenwan (申万) industry classification code of the stock.
$turnover_rate: daily turnover rate (换手率).
$turnover: alias of daily turnover rate for factor compatibility.

## Minute price and volume data
$open: minute open price.
$close: minute close price.
$high: minute high price.
$low: minute low price.
$volume: minute traded volume.
$vwap: minute volume weighted average price.

The expected schema for `minute_pv.h5` is:
- MultiIndex: `datetime`, `instrument`
- Columns: `$open`, `$close`, `$high`, `$low`, `$volume`, `$vwap`
