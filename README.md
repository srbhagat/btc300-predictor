# BTCUSDT 300-Point Final Calibrated Predictor

Final calibrated version.

## Included
- CoinDCX-only data
- Forward time-window database builder
- Adaptive historical matching
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

## Final score weighting
- Probability: 40%
- Sample size: 20%
- Trend/structure: 20%
- Confirmation: 20%

Fixed-300TP blocking penalties are still applied.

## Deploy
Replace:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
