# BTCUSDT 300-Point Final Hybrid Live Predictor

This is the improved final version.

## Data architecture

1. Binance Vision archive  
   Used for deep historical BTCUSDT futures candles.

2. Binance Futures API  
   Used to fill latest missing 15m candles up to the current moment where available.

3. CoinDCX live futures price  
   Used as the current trading reference because trades are executed on CoinDCX.

## Important

If Binance Futures API is blocked on Streamlit Cloud, the app does not crash. It falls back to:
- Binance archive + daily archive
- CoinDCX live price
- ATR live-zone support/resistance fallback

## Predictor logic

- Fixed TP = 300 points
- Default SL = 250 points
- Weighted probability
- Quality score
- A/B/C/D grade
- Sample confidence
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Safe support/resistance

## Deploy

Upload these files to GitHub:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
