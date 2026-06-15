# BTCUSDT 300-Point Final CoinDCX Forward Predictor

This version fixes the 997-candle issue.

## Main fix

Previous app used backward pagination and often stopped after about 997 candles.

This version uses forward time-window pagination:

- start from the requested history date
- request a 1000-candle time window
- move forward
- repeat until now

CoinDCX candle API supports:
- pair
- interval
- startTime
- endTime
- limit up to 1000

## Also fixed

- Indicator calculation no longer deletes rows because quote_volume/trades are blank.
- Cache filename changed so old bad cache is ignored.
- Database coverage is displayed.

## Strategy unchanged

- CoinDCX-only data
- Fixed TP = 300 points
- Default SL = 250 points
- Strict NO TRADE filter
- MTF bias
- BOS / CHOCH
- Liquidity sweep
- Volume zone
- Weighted probability
- Quality score and grade

## Deploy

Replace GitHub files:
- app.py
- requirements.txt
- README.md

Then reboot Streamlit Cloud.
