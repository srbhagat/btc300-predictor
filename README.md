# BTCUSDT 300-Point Final Normalized Predictor

Fixes:
- Normalized Long/Short probabilities now sum to 100%.
- Raw probabilities are still shown separately.
- Bad 997-candle cache is auto-rebuilt.
- TP remains fixed at 300 points.
- SL default is now 400 points.
- TP feasibility display added.

Deploy:
Replace app.py, requirements.txt, README.md and reboot Streamlit.
