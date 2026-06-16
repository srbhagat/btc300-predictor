# BTCUSDT 300-Point Final Tuned Predictor

This is the new fine-tuned final version.

## Built from the working 38k-candle version

This version keeps the forward CoinDCX database builder that produced 38,000+ candles,
then adds the calibrated quality score.

## Included

- CoinDCX-only data
- Forward time-window database builder
- Adaptive historical matching: Strict → Relaxed → Wide
- Fixed 300-point target
- Default 250-point stop loss
- Weighted probability
- Calibrated quality score
- A/B/C/D grade
- Sample confidence
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Strict NO TRADE filter

## Score weighting

- Probability: 40%
- Sample size: 20%
- Trend/structure: 20%
- Confirmation: 20%

Fixed 300-point TP blocking penalties are still applied.

## Deploy

Replace GitHub files:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.

Use Force rebuild once after deployment so the new cache is built cleanly.
