# BTCUSDT 300-Point Best CoinDCX Predictor

Best stable version for CoinDCX trading.

## Data source
This app uses CoinDCX only:
- CoinDCX public candle data
- CoinDCX futures live price

No Binance data is used.

## Why this version
- No Binance 451 errors
- No archive/live mismatch
- Support/resistance comes from CoinDCX candles
- BTC price comes from CoinDCX
- Works on Streamlit Cloud and iPhone

## Strategy
- TP fixed at 300 points
- Default SL 250 points
- Strict NO TRADE filter
- Weighted historical probability
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Quality score and grade

## Deploy
Upload:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
