
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="BTC 300 Fast Accurate India", page_icon="₿", layout="wide")

BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OI = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_OI_HIST = "https://fapi.binance.com/futures/data/openInterestHist"
COINDCX_TRADES = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/trades"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def interval_minutes(tf):
    return {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf]


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def cache_file(tf):
    return os.path.join(DATA_DIR, f"BTCUSDT_{tf}_cache.csv")


def load_cache(tf):
    path = cache_file(tf)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def save_cache(tf, df):
    df.to_csv(cache_file(tf), index=False)


def fetch_klines(tf, start_ms=None, end_ms=None, limit=1500):
    p = {"symbol": "BTCUSDT", "interval": tf, "limit": limit}
    if start_ms is not None:
        p["startTime"] = int(start_ms)
    if end_ms is not None:
        p["endTime"] = int(end_ms)
    r = requests.get(BINANCE_KLINES, params=p, timeout=20)
    r.raise_for_status()
    return r.json()


def klines_to_df(rows):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]
    if not rows:
        return pd.DataFrame(columns=cols + ["datetime"])
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades",
              "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    return df.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def update_tf(tf, years, box=None):
    old = load_cache(tf)
    tf_ms = interval_minutes(tf) * 60 * 1000
    now = datetime.now(timezone.utc)

    if old.empty:
        start_ms = to_ms(now - timedelta(days=int(years * 365.25)))
        frames = []
        mode = "building"
    else:
        start_ms = int(old["open_time"].max()) + tf_ms
        frames = [old]
        mode = "updating"

    end_ms = to_ms(now)
    if start_ms >= end_ms:
        if box:
            box.success(f"{tf}: ready with {len(old):,} candles")
        return old

    batch = 0
    while start_ms < end_ms:
        rows = fetch_klines(tf, start_ms=start_ms, end_ms=end_ms, limit=1500)
        if not rows:
            break
        dfb = klines_to_df(rows)
        if dfb.empty:
            break
        frames.append(dfb)
        batch += 1
        out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        save_cache(tf, out)
        if box:
            box.info(f"{tf}: {mode} batch {batch}, candles {len(out):,}")
        next_ms = int(dfb["open_time"].max()) + tf_ms
        if next_ms <= start_ms:
            break
        start_ms = next_ms
        time.sleep(0.05)

    out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    save_cache(tf, out)
    if box:
        box.success(f"{tf}: ready with {len(out):,} candles")
    return out


def ensure_db(years):
    boxes = {tf: st.empty() for tf in TIMEFRAMES}
    dfs = {}
    for tf in TIMEFRAMES:
        dfs[tf] = update_tf(tf, years, boxes[tf])
    return all(not dfs[tf].empty for tf in TIMEFRAMES), dfs


@st.cache_data(show_spinner=False)
def add_indicators_cached(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr_sum = df["atr14"].rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / atr_sum
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / atr_sum
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx14"] = dx.rolling(14).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.to_datetime(df["datetime"]).dt.date
    df["vwap"] = (typical * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(50).mean()
    df["momentum_4"] = df["close"] - df["close"].shift(4)
    df["momentum_12"] = df["close"] - df["close"].shift(12)
    df["trend"] = np.where(df["ema20"] > df["ema50"], "Bullish", "Bearish")
    df["major_trend"] = np.where(df["ema50"] > df["ema200"], "Bullish", "Bearish")

    hour = pd.to_datetime(df["datetime"]).dt.hour
    df["india_session"] = np.select(
        [
            (hour >= 5) & (hour < 12),
            (hour >= 12) & (hour < 18),
            (hour >= 18) & (hour < 24),
        ],
        ["Asia/India Morning", "London Open", "New York"],
        default="Late US"
    )

    df["range_pct_atr"] = (df["high"] - df["low"]) / df["atr14"]
    return df.dropna().reset_index(drop=True)


def fetch_coindcx_price():
    r = requests.get(COINDCX_TRADES, params={"pair": "B-BTC_USDT"}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return float(data[0]["price"])
    raise ValueError("No CoinDCX live price")


def funding_latest():
    try:
        r = requests.get(BINANCE_FUNDING, params={"symbol": "BTCUSDT", "limit": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return float(data[-1]["fundingRate"]) * 100 if data else np.nan
    except Exception:
        return np.nan


def oi_current():
    try:
        r = requests.get(BINANCE_OI, params={"symbol": "BTCUSDT"}, timeout=10)
        r.raise_for_status()
        return float(r.json()["openInterest"])
    except Exception:
        return np.nan


def oi_trend():
    try:
        r = requests.get(BINANCE_OI_HIST, params={"symbol": "BTCUSDT", "period": "15m", "limit": 32}, timeout=10)
        r.raise_for_status()
        data = r.json()
        vals = pd.Series([float(x["sumOpenInterest"]) for x in data])
        if len(vals) < 8:
            return "Unknown", np.nan
        chg = (vals.iloc[-1] - vals.iloc[-8]) / vals.iloc[-8] * 100
        if chg > 0.75:
            return "Rising", chg
        if chg < -0.75:
            return "Falling", chg
        return "Flat", chg
    except Exception:
        return "Unknown", np.nan


def current_structure(df, lookback=240):
    d = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []
    for i in range(3, len(d)-3):
        if d.loc[i, "high"] == d.loc[i-3:i+3, "high"].max():
            highs.append(d.loc[i, "high"])
        if d.loc[i, "low"] == d.loc[i-3:i+3, "low"].min():
            lows.append(d.loc[i, "low"])

    structure = "Mixed"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            structure = "Bullish HH-HL"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            structure = "Bearish LH-LL"

    support = float(min(lows[-8:])) if lows else float(d["low"].tail(80).min())
    resistance = float(max(highs[-8:])) if highs else float(d["high"].tail(80).max())

    close = float(d["close"].iloc[-1])
    bos = "None"
    if close > resistance:
        bos = "Bullish BOS"
    elif close < support:
        bos = "Bearish BOS"

    prev_high = d["high"].iloc[:-1].max()
    prev_low = d["low"].iloc[:-1].min()
    last = d.iloc[-1]
    sweep = "None"
    if last["high"] > prev_high and last["close"] < prev_high:
        sweep = "Bearish sweep"
    elif last["low"] < prev_low and last["close"] > prev_low:
        sweep = "Bullish sweep"

    return structure, bos, sweep, support, resistance


def mtf_votes(dfs):
    votes = []
    for tf, raw in dfs.items():
        d = add_indicators_cached(raw)
        x = d.iloc[-1]
        score = 0
        score += 1 if x["ema20"] > x["ema50"] else -1
        score += 1 if x["close"] > x["ema200"] else -1
        score += 1 if x["close"] > x["vwap"] else -1
        bias = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
        votes.append((tf, bias))
    bull = sum(1 for _, v in votes if v == "Bullish")
    bear = sum(1 for _, v in votes if v == "Bearish")
    if bull > bear:
        return "Bullish", votes
    if bear > bull:
        return "Bearish", votes
    return "Mixed", votes


def first_hit(df, idx, direction, tp_points, sl_points, horizon):
    entry = df.loc[idx, "close"]
    fut = df.iloc[idx+1:idx+1+horizon]
    if fut.empty:
        return None
    if direction == "LONG":
        tp, sl = entry + tp_points, entry - sl_points
        for _, r in fut.iterrows():
            hit_tp = r["high"] >= tp
            hit_sl = r["low"] <= sl
            if hit_tp and hit_sl:
                return "Ambiguous"
            if hit_tp:
                return "Win"
            if hit_sl:
                return "Loss"
    else:
        tp, sl = entry - tp_points, entry + sl_points
        for _, r in fut.iterrows():
            hit_tp = r["low"] <= tp
            hit_sl = r["high"] >= sl
            if hit_tp and hit_sl:
                return "Ambiguous"
            if hit_tp:
                return "Win"
            if hit_sl:
                return "Loss"
    return "No hit"


def historical_matches(df, latest, current_price, price_tol, balanced_mode=True):
    s = df[
        (df["close"] >= current_price * (1 - price_tol / 100)) &
        (df["close"] <= current_price * (1 + price_tol / 100))
    ].copy()

    s = s[(s["trend"] == latest["trend"]) & (s["major_trend"] == latest["major_trend"])]
    s = s[(s["rsi14"] >= latest["rsi14"] - 12) & (s["rsi14"] <= latest["rsi14"] + 12)]
    s = s[(s["atr14"] >= latest["atr14"] * 0.45) & (s["atr14"] <= latest["atr14"] * 1.65)]
    s = s[(s["adx14"] >= latest["adx14"] - 12) & (s["adx14"] <= latest["adx14"] + 12)]
    s = s[(s["vol_ratio"] >= latest["vol_ratio"] * 0.35) & (s["vol_ratio"] <= latest["vol_ratio"] * 2.25)]

    if balanced_mode:
        s = s[s["india_session"] == latest["india_session"]]

    return s


def analyze_side(df, current_price, direction, tp_points, sl_points, horizon, price_tol, balanced_mode):
    latest = df.iloc[-1]
    sample = historical_matches(df, latest, current_price, price_tol, balanced_mode)
    sample = sample[sample.index <= len(df) - horizon - 2]

    outcomes = []
    for idx in sample.index:
        out = first_hit(df, idx, direction, tp_points, sl_points, horizon)
        if out:
            outcomes.append(out)

    ser = pd.Series(outcomes)
    wins = int((ser == "Win").sum())
    losses = int((ser == "Loss").sum())
    no_hit = int((ser == "No hit").sum())
    ambiguous = int((ser == "Ambiguous").sum())
    decided = wins + losses
    prob = wins / decided * 100 if decided else 0

    return {
        "direction": direction,
        "matches": len(outcomes),
        "wins": wins,
        "losses": losses,
        "no_hit": no_hit,
        "ambiguous": ambiguous,
        "decided": decided,
        "probability": prob,
    }


def quality(direction, result, latest, mtf, structure, bos, sweep, support, resistance, current_price, tp, oi_state):
    score = 0
    reasons = []

    p = result["probability"]
    d = result["decided"]

    if p >= 70:
        score += 35; reasons.append("Very strong historical probability")
    elif p >= 65:
        score += 28; reasons.append("Strong historical probability")
    elif p >= 60:
        score += 20; reasons.append("Acceptable historical probability")
    elif p >= 55:
        score += 10; reasons.append("Mild historical edge")

    if d >= 300:
        score += 20; reasons.append("Large sample size")
    elif d >= 150:
        score += 14; reasons.append("Good sample size")
    elif d >= 60:
        score += 8; reasons.append("Minimum sample size")

    if (direction == "LONG" and mtf == "Bullish") or (direction == "SHORT" and mtf == "Bearish"):
        score += 12; reasons.append("MTF aligned")

    if (direction == "LONG" and "Bullish" in structure) or (direction == "SHORT" and "Bearish" in structure):
        score += 10; reasons.append("Market structure aligned")

    if (direction == "LONG" and bos == "Bullish BOS") or (direction == "SHORT" and bos == "Bearish BOS"):
        score += 8; reasons.append("Break of structure aligned")

    if (direction == "LONG" and sweep == "Bullish sweep") or (direction == "SHORT" and sweep == "Bearish sweep"):
        score += 8; reasons.append("Liquidity sweep aligned")

    if latest["adx14"] >= 22:
        score += 6; reasons.append("Trend strength acceptable")

    if (direction == "LONG" and current_price > latest["vwap"]) or (direction == "SHORT" and current_price < latest["vwap"]):
        score += 6; reasons.append("VWAP aligned")

    if (direction == "LONG" and oi_state == "Rising" and latest["momentum_4"] > 0) or (direction == "SHORT" and oi_state == "Rising" and latest["momentum_4"] < 0):
        score += 6; reasons.append("OI trend supports momentum")

    if direction == "LONG" and (resistance - current_price) < tp:
        score -= 18; reasons.append("Resistance too close")
    if direction == "SHORT" and (current_price - support) < tp:
        score -= 18; reasons.append("Support too close")

    return max(0, min(100, score)), reasons


def decide(long_r, short_r, lscore, sscore, min_prob, min_samples, min_gap, min_score):
    lp = long_r["probability"] if long_r["decided"] >= min_samples else 0
    sp = short_r["probability"] if short_r["decided"] >= min_samples else 0

    if long_r["decided"] < min_samples and short_r["decided"] < min_samples:
        return "NO TRADE", "Low sample size"

    if lp >= min_prob and lscore >= min_score and (lp - sp) >= min_gap:
        return "LONG", "Long probability and score passed"

    if sp >= min_prob and sscore >= min_score and (sp - lp) >= min_gap:
        return "SHORT", "Short probability and score passed"

    return "NO TRADE", "No high-quality edge"


st.title("BTCUSDT 300-Point Fast Accurate India")
st.caption("Designed for India timezone: fast refresh, 5-year database, no manual clicks, 300-point BTC futures decision support.")

with st.sidebar:
    st.header("Automatic settings")
    years = st.slider("Historical years", 1, 5, 5)
    refresh = st.selectbox("Auto-refresh seconds", [0, 60, 120, 300], index=1)

    st.header("Trade system")
    tp_points = st.number_input("Target points", value=300, min_value=50, step=50)
    sl_points = st.number_input("Stop-loss points", value=250, min_value=50, step=50)
    horizon = st.slider("Max 15m candles to hold", 4, 96, 32)

    st.header("Matching")
    price_tol = st.slider("Similar price zone ±%", 0.25, 8.0, 3.5, 0.25)
    balanced_mode = st.checkbox("India session matching", True)

    st.header("Signal strictness")
    min_samples = st.slider("Minimum decided samples", 30, 500, 60)
    min_prob = st.slider("Minimum probability %", 55, 80, 60)
    min_gap = st.slider("Minimum Long/Short gap %", 5, 35, 8)
    min_score = st.slider("Minimum quality score", 30, 90, 45)

ready_already = all(not load_cache(tf).empty for tf in TIMEFRAMES)
if refresh and ready_already:
    st_autorefresh(interval=refresh * 1000, key="refresh")

st.subheader("Automatic database")
ready, dfs = ensure_db(years)
if not ready:
    st.warning("Database is building. Keep this page open.")
    st.stop()

df15 = add_indicators_cached(dfs["15m"])

try:
    price = fetch_coindcx_price()
    price_src = "CoinDCX live"
except Exception:
    price = float(df15["close"].iloc[-1])
    price_src = "Binance latest close"

latest = df15.iloc[-1]
fund = funding_latest()
oi = oi_current()
oi_state, oi_change = oi_trend()
mtf, votes = mtf_votes(dfs)
structure, bos, sweep, support, resistance = current_structure(df15)

long_r = analyze_side(df15, price, "LONG", tp_points, sl_points, horizon, price_tol, balanced_mode)
short_r = analyze_side(df15, price, "SHORT", tp_points, sl_points, horizon, price_tol, balanced_mode)

lscore, lreasons = quality("LONG", long_r, latest, mtf, structure, bos, sweep, support, resistance, price, tp_points, oi_state)
sscore, sreasons = quality("SHORT", short_r, latest, mtf, structure, bos, sweep, support, resistance, price, tp_points, oi_state)

signal, reason = decide(long_r, short_r, lscore, sscore, min_prob, min_samples, min_gap, min_score)

c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC price", f"{price:,.2f}", price_src)
c2.metric("MTF bias", mtf)
c3.metric("Structure", structure)
c4.metric("Session", latest["india_session"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("RSI", f"{latest['rsi14']:.1f}")
c6.metric("ATR", f"{latest['atr14']:.1f}")
c7.metric("ADX", f"{latest['adx14']:.1f}")
c8.metric("Liquidity sweep", sweep)

c9, c10, c11, c12 = st.columns(4)
c9.metric("Support", f"{support:,.0f}")
c10.metric("Resistance", f"{resistance:,.0f}")
c11.metric("OI trend", oi_state, "" if np.isnan(oi_change) else f"{oi_change:.2f}%")
c12.metric("Funding", "N/A" if np.isnan(fund) else f"{fund:.4f}%")

st.subheader("Final signal")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Signal", signal)
s2.metric("Long probability", f"{long_r['probability']:.1f}%")
s3.metric("Short probability", f"{short_r['probability']:.1f}%")
s4.metric("Reason", reason)

q1, q2 = st.columns(2)
q1.metric("Long quality score", f"{lscore}/100")
q2.metric("Short quality score", f"{sscore}/100")

if signal == "LONG":
    st.success(f"LONG: Entry area {price:,.2f} | TP {price + tp_points:,.2f} | SL {price - sl_points:,.2f}")
elif signal == "SHORT":
    st.error(f"SHORT: Entry area {price:,.2f} | TP {price - tp_points:,.2f} | SL {price + sl_points:,.2f}")
else:
    st.info("NO TRADE: The app did not find enough edge for a 300-point trade.")

st.subheader("Evidence")
st.dataframe(pd.DataFrame([long_r, short_r]), use_container_width=True)

r1, r2 = st.columns(2)
with r1:
    st.write("Long reasons")
    st.write(lreasons if lreasons else ["No strong long factors"])
with r2:
    st.write("Short reasons")
    st.write(sreasons if sreasons else ["No strong short factors"])

st.write("Multi-timeframe votes:", dict(votes))
st.write(f"15m candles: {len(df15):,} | From {df15['datetime'].min()} to {df15['datetime'].max()}")

st.subheader("Latest chart")
st.line_chart(df15.tail(300).set_index("datetime")[["close", "ema20", "ema50", "ema200", "vwap"]])

st.warning("Decision support only. BTC futures are risky. This app cannot guarantee profit or accuracy.")
