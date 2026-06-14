# BTCUSDT 300-Point Final Score Predictor

This version keeps the trade logic fair and mathematical.

## What changed
- Quality score is now balanced:
  - 50% weighted probability
  - 20% sample size
  - 15% trend/structure
  - 15% confirmations
- Adds confidence grade:
  - A - Strong
  - B - Good
  - C - Watch
  - D - Weak
- Adds sample confidence:
  - High sample
  - Good sample
  - Moderate sample
  - Low sample
  - Very low sample

## What did NOT change
- TP remains fixed at 300 points.
- SL default remains 250 points.
- Predictor still says NO TRADE if edge is weak.
- Binance archive is used for history.
- CoinDCX live price is used for current trading reference.

## Deploy
Replace these files in GitHub:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
