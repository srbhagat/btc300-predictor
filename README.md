# BTCUSDT 300-Point Final CoinDCX Adaptive Predictor

This version keeps the fixed 300-point target and improves only the historical matcher.

## What changed

The previous matcher was too strict and returned 0 historical matches.

This version uses adaptive matching:

1. Strict match
2. Relaxed match
3. Wide match

It does not force trades. It only widens historical comparison when strict matching gives too few examples.

## Data

- CoinDCX-only candles
- CoinDCX live futures price
- Forward time-window database builder
- No Binance

## Strategy unchanged

- TP fixed at 300 points
- SL default 250 points
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Weighted probability
- Quality score
- A/B/C/D grade
- Strict NO TRADE filter

## Deploy

Replace GitHub files:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
