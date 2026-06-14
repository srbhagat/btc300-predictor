# BTCUSDT 300-Point Final Weighted Pro Predictor

This keeps the 300-point target fixed.

## Added improvements
- BOS detection
- CHOCH detection
- Liquidity sweep detection
- Volume spike confirmation
- High-volume zone support/resistance context
- Improved quality score
- Same strict NO TRADE filter

## Data architecture
- Binance Vision archive for historical BTCUSDT futures candles
- 3-year default history
- CoinDCX live futures price for actual trading reference
- Recent-data weighted probability

## Fixed TP
Target remains fixed at 300 points.
Stop loss remains adjustable, default 250 points.

## Upload
Replace:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
