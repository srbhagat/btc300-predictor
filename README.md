# BTCUSDT 300-Point Final Corrected CoinDCX Predictor

This is the corrected final app.

## Main fix

The previous CoinDCX-only app created blank columns:

- quote_volume
- trades

Then `add_indicators()` used `df.dropna()`, which deleted all rows because those columns were NaN.

This corrected version drops NaN only from required indicator columns, so CoinDCX candle data remains usable.

## Data

- CoinDCX public candles only
- CoinDCX futures live price only
- No Binance
- No 451 errors
- No archive/live mismatch

## Strategy

- Fixed TP = 300 points
- Default SL = 250 points
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Weighted probability
- Quality score
- Grade system
- Strict NO TRADE filter

## Deploy

Replace GitHub files:

- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
