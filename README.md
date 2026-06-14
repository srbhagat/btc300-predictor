# BTCUSDT 300-Point Cloud Predictor V2

Fixed Streamlit Cloud version.

## Fixes
- Uses new cache filename to ignore old bad cache.
- Validates candle count.
- Validates latest archive close against CoinDCX live price.
- Rebuilds stale or partial databases.
- Downloads Binance Vision 15m archive and resamples to 1h, 4h, 1d.
- Avoids Binance Futures API, which can return HTTP 451 on Streamlit Cloud.