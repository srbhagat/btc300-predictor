# BTCUSDT 300-Point Cloud Predictor

Cloud-safe version for Streamlit Cloud.

It downloads 15m BTCUSDT futures archive data from Binance Vision and resamples internally to 1h, 4h, and 1d.
This avoids Binance Futures API HTTP 451 issues on Streamlit Cloud.