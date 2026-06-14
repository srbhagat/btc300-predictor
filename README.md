# BTCUSDT 300-Point Final Weighted Predictor

This is the recommended final version.

## Final architecture

- Binance Vision archive for historical BTCUSDT futures candles
- 3-year default history, not 5 years
- CoinDCX live futures price for actual trade reference
- Current-month daily archive candles where available
- 1h, 4h and 1d internally resampled from 15m data
- Recent-data weighted probability

## Why 3 years and weighted probability?

For BTC 300-point trades, very old data should not count equally.
The app weights matches like this:

- Last 90 days: 1.00
- Last 1 year: 0.70
- Last 3 years: 0.35
- Older: 0.15

This balances statistical sample size with current market relevance.

## Safe support/resistance

If archive latest price is close to CoinDCX live price, the app uses recent swing support/resistance.

If archive latest price is far from CoinDCX live price, it does NOT show stale levels as real levels. It switches to ATR-based live zone and labels it.

## Upload to GitHub

Replace these files:

- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
