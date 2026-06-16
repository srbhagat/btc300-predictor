# BTCUSDT 300-Point Production Final Predictor

Safe final version.

## Built from the working adaptive version

This keeps the CoinDCX forward database builder and adaptive matcher.

## Included

- CoinDCX-only data
- Forward time-window database builder
- Adaptive matching: Strict → Relaxed → Wide
- Fixed 300-point target
- Default 250-point stop loss
- Weighted probability
- Directional edge gap
- Readable grade
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Strict NO TRADE filter

## Deploy

Replace:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.

If candle count is not 38,000+, tick Force rebuild once.
