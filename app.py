
import io
import os
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="BTC 300 Cloud Predictor V2", page_icon="₿", layout="wide")

COINDCX_TRADES = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/trades"
ARCHIVE_URL = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-{ym}.zip"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_NAME = "BTCUSDT_15m_cloud_cache_v2.csv"

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]


def cache_path():
    return os.path.join(DATA_DIR, CACHE_NAME)


def fetch_price():
    r = requests.get(COINDCX_TRADES, params={"pair": "B-BTC_USDT"}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return float(data[0]["price"]), "CoinDCX live"
    return np.nan, "No live price"


def load_cache():
    p = cache_path()
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def save_cache(df):
    df.to_csv(cache_path(), index=False)


def last_completed_month():
    now = datetime.now(timezone.utc)
    y, m = now.year, now.month - 1
    if m == 0:
        y -= 1
        m = 12
    return pd.Timestamp(year=y, month=m, day=1)


def month_list(years):
    end = last_completed_month()
    start = end - pd.DateOffset(months=years * 12 - 1)
    months = pd.date_range(start=start, end=end, freq="MS")
    return [x.strftime("%Y-%m") for x in months]


def read_month_15m(ym):
    url = ARCHIVE_URL.format(ym=ym)
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=45, headers=headers)

    if r.status_code == 404:
        return pd.DataFrame(), f"{ym}: not available"
    if r.status_code != 200:
        return pd.DataFrame(), f"{ym}: HTTP {r.status_code}"

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame(), f"{ym}: no CSV inside ZIP"

        raw = z.read(names[0])

        # Binance Vision futures archive is normally headerless.
        df = pd.read_csv(io.BytesIO(raw), header=None)
        if df.shape[1] >= 12:
            df = df.iloc[:, :12]
            df.columns = COLS
        else:
            # fallback for rare headered CSV
            df = pd.read_csv(io.BytesIO(raw))
            lower_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
            df = df.rename(columns=lower_map)
            if "open_time" not in df.columns:
                return pd.DataFrame(), f"{ym}: unrecognized columns {list(df.columns)[:5]}"

        # Strong numeric conversion without unsafe string replacement.
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades",
                  "taker_buy_volume", "taker_buy_quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

        df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]

        # Remove impossible rows
        df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
                (df["low"] <= df[["open", "close", "high"]].min(axis=1))]

        df["datetime"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

        # 15m monthly data should have roughly 2600-3000 rows. Accept partial latest listings too.
        return df, f"{ym}: loaded {len(df):,} candles, close {df['close'].iloc[-1]:,.0f}"

    except Exception as e:
        return pd.DataFrame(), f"{ym}: parse error {e}"


def cache_is_valid(df, years, live_price=None):
    if df.empty:
        return False, "cache missing"

    min_expected = int(years * 365 * 96 * 0.70)  # 70% tolerance
    if len(df) < min_expected:
        return False, f"cache too small: {len(df):,}, expected around {years*365*96:,}"

    latest_dt = pd.to_datetime(df["datetime"].max())
    age_days = (pd.Timestamp.now(tz=latest_dt.tz) - latest_dt).days
    if age_days > 75:
        return False, f"cache stale: latest candle {latest_dt}"

    if live_price and live_price > 0:
        hist_close = float(df["close"].iloc[-1])
        diff_pct = abs(hist_close - live_price) / live_price * 100
        if diff_pct > 35:
            return False, f"cache price mismatch: hist {hist_close:,.0f}, live {live_price:,.0f}"

    return True, "cache valid"


def build_or_load_15m(years, live_price=None, force_rebuild=False):
    status = st.empty()
    cached = load_cache()
    valid, reason = cache_is_valid(cached, years, live_price)

    if valid and not force_rebuild:
        status.success(f"15m database ready with {len(cached):,} candles | {reason}")
        return cached

    status.warning(f"Rebuilding database because: {reason}")

    months = month_list(years)
    frames = []
    logs = []

    prog = st.progress(0, text="Starting historical download...")

    for i, ym in enumerate(months, 1):
        dfm, msg = read_month_15m(ym)
        logs.append(msg)
        if not dfm.empty:
            frames.append(dfm)

        count = sum(len(x) for x in frames)
        prog.progress(i / len(months), text=f"15m archive {i}/{len(months)}: {msg} | total {count:,}")
        status.info(f"Downloading {ym} | total candles {count:,}")

        if frames and i % 6 == 0:
            tmp = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
            save_cache(tmp)

        time.sleep(0.04)

    if not frames:
        st.error("No archive candles loaded.")
        st.write(logs[-20:])
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    save_cache(out)
    valid2, reason2 = cache_is_valid(out, years, live_price)

    if valid2:
        status.success(f"15m database ready with {len(out):,} candles | {reason2}")
    else:
        status.error(f"Database built but still invalid: {reason2}")
        st.write("Recent archive logs:")
        st.write(logs[-20:])

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
    out["quote_volume"] = d["quote_volume"].resample(rule).sum()
    out["trades"] = d["trades"].resample(rule).sum()
    out = out.dropna().reset_index()
    out["open_time"] = (pd.to_datetime(out["datetime"]).dt.tz_convert("UTC").astype("int64") // 10**6)
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
        x = d.iloc[-1]
        score = 0
        score += 1 if x["ema20"] > x["ema50"] else -1
        score += 1 if x["close"] > x["ema200"] else -1
        score += 1 if x["close"] > x["vwap"] else -1
        votes.append((tf, "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"))
    bull = sum(1 for _, v in votes if v == "Bullish")
    bear = sum(1 for _, v in votes if v == "Bearish")
    return ("Bullish" if bull > bear else "Bearish" if bear > bull else "Mixed"), votes


def structure_levels(df):
    d = df.tail(240).reset_index(drop=True)
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


st.title("BTCUSDT 300-Point Cloud Predictor V2")
st.caption("Fixed cloud version: validates cache, rebuilds stale data, uses Binance archive data, no same Wi-Fi needed.")

try:
    live_price, live_src = fetch_price()
except Exception:
    live_price, live_src = np.nan, "Live price unavailable"

with st.sidebar:
    st.header("Automatic settings")
    years = st.slider("Historical years", 1, 5, 5)
    refresh = st.selectbox("Auto-refresh seconds", [0, 60, 120, 300], index=1)
    force_rebuild = st.checkbox("Force rebuild database", False)

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

st.subheader("Automatic database")
df15_raw = build_or_load_15m(years, live_price if not np.isnan(live_price) else None, force_rebuild=force_rebuild)

if df15_raw.empty:
    st.error("Could not build candle database.")
    st.stop()

valid, valid_reason = cache_is_valid(df15_raw, years, live_price if not np.isnan(live_price) else None)
if not valid:
    st.error(f"Database invalid: {valid_reason}")
    st.stop()

st.success(f"15m database ready with {len(df15_raw):,} candles. Latest archive close: {df15_raw['close'].iloc[-1]:,.0f}")

df15 = add_indicators(df15_raw)
dfs = {
    "15m": df15_raw,
    "1h": resample_ohlcv(df15_raw, "1h"),
    "4h": resample_ohlcv(df15_raw, "4h"),
    "1d": resample_ohlcv(df15_raw, "1D"),
}

price = live_price if not np.isnan(live_price) else float(df15["close"].iloc[-1])
price_src = live_src if not np.isnan(live_price) else "Latest archive close"

latest = df15.iloc[-1]
mtf, votes = mtf_bias(dfs)
structure, support, resistance = structure_levels(df15)

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
st.write(f"Data: 15m={len(df15_raw):,}, 1h={len(dfs['1h']):,}, 4h={len(dfs['4h']):,}, 1d={len(dfs['1d']):,}")
st.subheader("Latest chart")
st.line_chart(df15.tail(300).set_index("datetime")[["close", "ema20", "ema50", "ema200", "vwap"]])

st.warning("Decision-support only. BTC futures are risky. This app cannot guarantee profit or accuracy.")
