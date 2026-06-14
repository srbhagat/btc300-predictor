
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="BTC 300 CoinDCX Only Predictor", page_icon="₿", layout="wide")

COINDCX_CANDLES = "https://public.coindcx.com/market_data/candles/"
COINDCX_FUTURES_TRADES = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/trades"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_FILE = os.path.join(DATA_DIR, "coindcx_B_BTC_USDT_15m_cache.csv")

PAIR = "B-BTC_USDT"


def interval_ms(interval: str) -> int:
    return {
        "1m": 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }[interval]


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CACHE_FILE)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def save_cache(df):
    df.to_csv(CACHE_FILE, index=False)


def normalize_candles(data):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # CoinDCX public candles use: open, high, low, close, volume, time.
    if "time" in df.columns:
        df = df.rename(columns={"time": "open_time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "open_time"})

    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    for c in ["open_time", "open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
            (df["low"] <= df[["open", "close", "high"]].min(axis=1))]

    df["open_time"] = df["open_time"].astype("int64")
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
    df["quote_volume"] = np.nan
    df["trades"] = np.nan

    return df[["open_time", "datetime", "open", "high", "low", "close", "volume", "quote_volume", "trades"]].drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def fetch_candles(start_ms=None, end_ms=None, interval="15m", limit=1000):
    params = {
        "pair": PAIR,
        "interval": interval,
        "limit": min(int(limit), 1000),
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(COINDCX_CANDLES, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return normalize_candles(r.json())


def fetch_live_price():
    # CoinDCX futures trade endpoint; still CoinDCX only.
    try:
        r = requests.get(COINDCX_FUTURES_TRADES, params={"pair": PAIR}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return float(data[0]["price"]), "CoinDCX futures live"
    except Exception:
        pass

    # fallback: latest candle close
    d = fetch_candles(interval="15m", limit=2)
    if not d.empty:
        return float(d["close"].iloc[-1]), "CoinDCX latest candle"
    return np.nan, "No live price"


def build_or_update_cache(years=5, interval="15m", force_rebuild=False):
    status = st.empty()
    progress = st.empty()

    old = pd.DataFrame() if force_rebuild else load_cache()
    step = interval_ms(interval)

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_target = int((pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years)).timestamp() * 1000)

    frames = []

    if not old.empty:
        frames.append(old)
        last_ms = int(old["open_time"].max())
        # If cache is recent, only update from last candle.
        start_ms = last_ms + step
        status.success(f"Cache found with {len(old):,} candles. Updating latest CoinDCX candles...")
    else:
        start_ms = start_target
        status.warning("Building CoinDCX-only candle database from scratch. First run can take time.")

    if start_ms >= now_ms - step:
        out = old.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        status.success(f"CoinDCX 15m database ready with {len(out):,} candles")
        return out

    total_span = max(1, now_ms - start_ms)
    batch = 0
    consecutive_empty = 0

    while start_ms < now_ms:
        end_ms = min(now_ms, start_ms + step * 1000)
        try:
            dfb = fetch_candles(start_ms=start_ms, end_ms=end_ms, interval=interval, limit=1000)
        except Exception as e:
            status.error(f"CoinDCX candles download failed: {e}")
            break

        batch += 1

        if dfb.empty:
            consecutive_empty += 1
            # Move forward to prevent infinite loop.
            start_ms = end_ms + step
            if consecutive_empty > 10:
                status.warning("Too many empty CoinDCX batches. Stopping database build.")
                break
        else:
            consecutive_empty = 0
            frames.append(dfb)
            start_ms = int(dfb["open_time"].max()) + step

        if frames:
            out_tmp = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
            if batch % 8 == 0:
                save_cache(out_tmp)
            done = min(1.0, max(0.0, (start_ms - (int(old["open_time"].max()) + step if not old.empty else start_target)) / total_span))
            progress.progress(done, text=f"CoinDCX batch {batch} | candles {len(out_tmp):,} | latest {out_tmp['datetime'].max()}")
            status.info(f"Downloading CoinDCX candles... batch {batch}, total candles {len(out_tmp):,}")
        time.sleep(0.08)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    save_cache(out)

    # Validate rough expected count, but don't fail if CoinDCX doesn't provide full 5 years.
    latest_dt = pd.to_datetime(out["datetime"].max())
    age_hours = (pd.Timestamp.now(tz=latest_dt.tz) - latest_dt).total_seconds() / 3600
    if age_hours > 48:
        status.warning(f"CoinDCX database is not very fresh. Latest candle: {latest_dt}")
    else:
        status.success(f"CoinDCX 15m database ready with {len(out):,} candles")

    return out


def resample_ohlcv(df, rule):
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.set_index("datetime").sort_index()

    out = pd.DataFrame()
    out["open"] = d["open"].resample(rule).first()
    out["high"] = d["high"].resample(rule).max()
    out["low"] = d["low"].resample(rule).min()
    out["close"] = d["close"].resample(rule).last()
    out["volume"] = d["volume"].resample(rule).sum()
    out = out.dropna().reset_index()
    out["open_time"] = (pd.to_datetime(out["datetime"]).dt.tz_convert("UTC").astype("int64") // 10**6)
    out["quote_volume"] = np.nan
    out["trades"] = np.nan
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def add_indicators(df):
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
    df["trend"] = np.where(df["ema20"] > df["ema50"], "Bullish", "Bearish")
    df["major_trend"] = np.where(df["ema50"] > df["ema200"], "Bullish", "Bearish")

    hour = pd.to_datetime(df["datetime"]).dt.hour
    df["india_session"] = np.select(
        [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 18), (hour >= 18) & (hour < 24)],
        ["Asia/India Morning", "London Open", "New York"],
        default="Late US"
    )
    return df.dropna().reset_index(drop=True)


def mtf_bias(dfs):
    votes = []
    for tf, raw in dfs.items():
        d = add_indicators(raw)
        if d.empty:
            continue
        x = d.iloc[-1]
        score = 0
        score += 1 if x["ema20"] > x["ema50"] else -1
        score += 1 if x["close"] > x["ema200"] else -1
        score += 1 if x["close"] > x["vwap"] else -1
        votes.append((tf, "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"))
    bull = sum(1 for _, v in votes if v == "Bullish")
    bear = sum(1 for _, v in votes if v == "Bearish")
    return ("Bullish" if bull > bear else "Bearish" if bear > bull else "Mixed"), votes


def structure_levels(df, live_price):
    d = df.tail(480).reset_index(drop=True)
    highs, lows = [], []
    for i in range(3, len(d)-3):
        if d.loc[i, "high"] == d.loc[i-3:i+3, "high"].max():
            highs.append(float(d.loc[i, "high"]))
        if d.loc[i, "low"] == d.loc[i-3:i+3, "low"].min():
            lows.append(float(d.loc[i, "low"]))

    structure = "Mixed"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            structure = "Bullish HH-HL"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            structure = "Bearish LH-LL"

    swing_lows = sorted([x for x in lows if x < live_price], reverse=True)
    swing_highs = sorted([x for x in highs if x > live_price])
    support = swing_lows[0] if swing_lows else float(d["low"].tail(80).min())
    resistance = swing_highs[0] if swing_highs else float(d["high"].tail(80).max())

    return structure, support, resistance


def first_hit(df, idx, direction, tp_points, sl_points, horizon):
    entry = df.loc[idx, "close"]
    fut = df.iloc[idx+1:idx+1+horizon]
    if fut.empty:
        return None
    if direction == "LONG":
        tp, sl = entry + tp_points, entry - sl_points
        for _, r in fut.iterrows():
            if r["high"] >= tp and r["low"] <= sl:
                return "Ambiguous"
            if r["high"] >= tp:
                return "Win"
            if r["low"] <= sl:
                return "Loss"
    else:
        tp, sl = entry - tp_points, entry + sl_points
        for _, r in fut.iterrows():
            if r["low"] <= tp and r["high"] >= sl:
                return "Ambiguous"
            if r["low"] <= tp:
                return "Win"
            if r["high"] >= sl:
                return "Loss"
    return "No hit"


def analyze(df, price, direction, tp, sl, horizon, tol, session_match):
    latest = df.iloc[-1]
    s = df[(df["close"] >= price*(1-tol/100)) & (df["close"] <= price*(1+tol/100))].copy()
    s = s[(s["trend"] == latest["trend"]) & (s["major_trend"] == latest["major_trend"])]
    s = s[(s["rsi14"] >= latest["rsi14"]-12) & (s["rsi14"] <= latest["rsi14"]+12)]
    s = s[(s["atr14"] >= latest["atr14"]*0.45) & (s["atr14"] <= latest["atr14"]*1.65)]
    s = s[(s["adx14"] >= latest["adx14"]-12) & (s["adx14"] <= latest["adx14"]+12)]
    s = s[(s["vol_ratio"] >= latest["vol_ratio"]*0.35) & (s["vol_ratio"] <= latest["vol_ratio"]*2.25)]
    if session_match:
        s = s[s["india_session"] == latest["india_session"]]
    s = s[s.index <= len(df)-horizon-2]

    outcomes = [first_hit(df, idx, direction, tp, sl, horizon) for idx in s.index]
    outcomes = [x for x in outcomes if x]
    ser = pd.Series(outcomes)
    wins = int((ser == "Win").sum())
    losses = int((ser == "Loss").sum())
    no_hit = int((ser == "No hit").sum())
    ambiguous = int((ser == "Ambiguous").sum())
    decided = wins + losses
    prob = wins / decided * 100 if decided else 0
    return {"direction": direction, "matches": len(outcomes), "wins": wins, "losses": losses,
            "no_hit": no_hit, "ambiguous": ambiguous, "decided": decided, "probability": prob}


def score_side(direction, res, latest, mtf, structure, support, resistance, price, tp):
    score = 0
    reasons = []
    p, d = res["probability"], res["decided"]
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
        score += 10; reasons.append("Structure aligned")
    if latest["adx14"] >= 22:
        score += 6; reasons.append("Trend strength acceptable")
    if (direction == "LONG" and price > latest["vwap"]) or (direction == "SHORT" and price < latest["vwap"]):
        score += 6; reasons.append("VWAP aligned")
    if direction == "LONG" and (resistance-price) < tp:
        score -= 18; reasons.append("Resistance too close")
    if direction == "SHORT" and (price-support) < tp:
        score -= 18; reasons.append("Support too close")

    return max(0, min(100, score)), reasons


def decide(lr, sr, ls, ss, min_prob, min_samples, min_gap, min_score):
    lp = lr["probability"] if lr["decided"] >= min_samples else 0
    sp = sr["probability"] if sr["decided"] >= min_samples else 0
    if lr["decided"] < min_samples and sr["decided"] < min_samples:
        return "NO TRADE", "Low sample size"
    if lp >= min_prob and ls >= min_score and (lp-sp) >= min_gap:
        return "LONG", "Long probability and score passed"
    if sp >= min_prob and ss >= min_score and (sp-lp) >= min_gap:
        return "SHORT", "Short probability and score passed"
    return "NO TRADE", "No high-quality edge"


st.title("BTCUSDT 300-Point CoinDCX Only Predictor")
st.caption("Uses only CoinDCX public data: CoinDCX candles + CoinDCX futures live price. No Binance data.")

with st.sidebar:
    st.header("Automatic settings")
    years = st.slider("Historical years requested", 1, 5, 5)
    refresh = st.selectbox("Auto-refresh seconds", [0, 60, 120, 300], index=1)
    force_rebuild = st.checkbox("Force rebuild CoinDCX database", False)

    st.header("Trade system")
    tp_points = st.number_input("Target points", value=300, min_value=50, step=50)
    sl_points = st.number_input("Stop-loss points", value=250, min_value=50, step=50)
    horizon = st.slider("Max 15m candles to hold", 4, 96, 32)

    st.header("Matching")
    price_tol = st.slider("Similar price zone ±%", 0.25, 8.0, 3.5, 0.25)
    session_match = st.checkbox("India session matching", True)

    st.header("Signal strictness")
    min_samples = st.slider("Minimum decided samples", 30, 500, 60)
    min_prob = st.slider("Minimum probability %", 55, 80, 60)
    min_gap = st.slider("Minimum Long/Short gap %", 5, 35, 8)
    min_score = st.slider("Minimum quality score", 30, 90, 45)

if refresh and not load_cache().empty and not force_rebuild:
    st_autorefresh(interval=refresh*1000, key="refresh")

st.subheader("CoinDCX-only automatic database")
df15_raw = build_or_update_cache(years=years, interval="15m", force_rebuild=force_rebuild)

if df15_raw.empty or len(df15_raw) < 500:
    st.error("CoinDCX candle database is not ready yet. Keep app open or reduce requested years.")
    st.stop()

df15 = add_indicators(df15_raw)

try:
    price, price_src = fetch_live_price()
    if np.isnan(price):
        raise ValueError("No price")
except Exception:
    price = float(df15["close"].iloc[-1])
    price_src = "CoinDCX latest candle close"

dfs = {
    "15m": df15_raw,
    "1h": resample_ohlcv(df15_raw, "1h"),
    "4h": resample_ohlcv(df15_raw, "4h"),
    "1d": resample_ohlcv(df15_raw, "1D"),
}

latest = df15.iloc[-1]
mtf, votes = mtf_bias(dfs)
structure, support, resistance = structure_levels(df15, price)

long_r = analyze(df15, price, "LONG", tp_points, sl_points, horizon, price_tol, session_match)
short_r = analyze(df15, price, "SHORT", tp_points, sl_points, horizon, price_tol, session_match)
lscore, lreasons = score_side("LONG", long_r, latest, mtf, structure, support, resistance, price, tp_points)
sscore, sreasons = score_side("SHORT", short_r, latest, mtf, structure, support, resistance, price, tp_points)
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
c8.metric("VWAP", f"{latest['vwap']:,.0f}")

c9, c10 = st.columns(2)
c9.metric("Support", f"{support:,.0f}")
c10.metric("Resistance", f"{resistance:,.0f}")

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
    st.info("NO TRADE: The app did not find enough CoinDCX-based edge for a 300-point trade.")

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
st.write(f"CoinDCX data: 15m={len(df15_raw):,}, 1h={len(dfs['1h']):,}, 4h={len(dfs['4h']):,}, 1d={len(dfs['1d']):,}")
st.write(f"Data range: {df15_raw['datetime'].min()} to {df15_raw['datetime'].max()}")
st.info("This app intentionally uses CoinDCX only. If CoinDCX returns fewer historical candles than 5 years, the app will use whatever CoinDCX provides.")

st.subheader("Latest chart")
st.line_chart(df15.tail(300).set_index("datetime")[["close", "ema20", "ema50", "ema200", "vwap"]])

st.warning("Decision-support only. BTC futures are risky. This app cannot guarantee profit or accuracy.")
