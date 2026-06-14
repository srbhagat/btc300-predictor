
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
DAILY_ARCHIVE_URL = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/BTCUSDT-15m-{date}.zip"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_NAME = "BTCUSDT_15m_final_cloud_cache.csv"

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



def parse_archive_csv(raw, label):
    """Shared parser for Binance Vision monthly/daily zip CSVs."""
    try:
        df = pd.read_csv(io.BytesIO(raw), header=None)
        if df.shape[1] >= 12:
            df = df.iloc[:, :12]
            df.columns = COLS
        else:
            df = pd.read_csv(io.BytesIO(raw))
            lower_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
            df = df.rename(columns=lower_map)
            if "open_time" not in df.columns:
                return pd.DataFrame(), f"{label}: unrecognized csv"

        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades",
                  "taker_buy_volume", "taker_buy_quote_volume"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["close_time"] = pd.to_numeric(df.get("close_time", np.nan), errors="coerce")

        df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        df = df[(df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
                (df["low"] <= df[["open", "close", "high"]].min(axis=1))]
        df["datetime"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True).dt.tz_convert("Asia/Kolkata")
        return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True), f"{label}: loaded {len(df):,}"
    except Exception as e:
        return pd.DataFrame(), f"{label}: parse error {e}"


def read_daily_15m(date_str):
    url = DAILY_ARCHIVE_URL.format(date=date_str)
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, timeout=30, headers=headers)
    if r.status_code == 404:
        return pd.DataFrame(), f"{date_str}: daily not available"
    if r.status_code != 200:
        return pd.DataFrame(), f"{date_str}: HTTP {r.status_code}"
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame(), f"{date_str}: no csv"
        return parse_archive_csv(z.read(names[0]), date_str)
    except Exception as e:
        return pd.DataFrame(), f"{date_str}: zip error {e}"


def recent_daily_dates_from_monthly_end():
    """
    Monthly archive usually ends at previous completed month.
    Daily archive gives current month completed days.
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    start = (last_completed_month() + pd.offsets.MonthBegin(1)).tz_localize("UTC")
    # Use yesterday UTC; today's daily ZIP may not be complete/available.
    end = (now_utc - pd.Timedelta(days=1)).normalize()
    if start > end:
        return []
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start, end=end, freq="D")]


def cache_is_valid(df, years, live_price=None):
    if df.empty:
        return False, "cache missing"

    min_expected = int(years * 365 * 96 * 0.70)  # 70% tolerance
    if len(df) < min_expected:
        return False, f"cache too small: {len(df):,}, expected around {years*365*96:,}"

    latest_dt = pd.to_datetime(df["datetime"].max())
    age_days = (pd.Timestamp.now(tz=latest_dt.tz) - latest_dt).days
    if age_days > 20:
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

    # Add current-month daily files so live price and support/resistance are not stale.
    daily_dates = recent_daily_dates_from_monthly_end()
    if daily_dates:
        st.info(f"Adding current-month daily archive candles: {daily_dates[0]} to {daily_dates[-1]}")
    for j, day in enumerate(daily_dates, 1):
        dfd, msg = read_daily_15m(day)
        logs.append(msg)
        if not dfd.empty:
            frames.append(dfd)
        prog.progress(1.0, text=f"Daily archive {j}/{len(daily_dates)}: {msg}")
        time.sleep(0.03)

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


def structure_levels(df, live_price=None, latest_atr=None):
    """
    FINAL SAFE SUPPORT/RESISTANCE LOGIC

    Rule 1:
    If archive latest close is close to CoinDCX live price,
    use recent swing highs/lows.

    Rule 2:
    If archive latest close is far from CoinDCX live price,
    DO NOT use stale historical swing levels.
    Use ATR-based live zone and label it clearly as estimated.

    This prevents wrong values like support 73k when live price is 64k.
    """
    d = df.tail(600).reset_index(drop=True)
    archive_latest_close = float(d["close"].iloc[-1])
    atr = float(latest_atr) if latest_atr and latest_atr > 0 else max(250.0, archive_latest_close * 0.004)

    live_price = float(live_price) if live_price and live_price > 0 else archive_latest_close
    archive_live_gap_pct = abs(archive_latest_close - live_price) / live_price * 100

    # Structure is still calculated from latest available archive candles.
    highs, lows = [], []
    for i in range(3, len(d) - 3):
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

    # If archive and live are too far apart, historical levels are stale.
    if archive_live_gap_pct > 3.0:
        support = live_price - max(300.0, atr * 2.0)
        resistance = live_price + max(300.0, atr * 2.0)
        level_mode = f"ATR live zone; archive/live gap {archive_live_gap_pct:.1f}%"
        return structure, float(support), float(resistance), level_mode

    # Archive is close enough to live; use real swing levels around live price.
    swing_lows = sorted([x for x in lows if x < live_price], reverse=True)
    swing_highs = sorted([x for x in highs if x > live_price])

    support = float(swing_lows[0]) if swing_lows else live_price - max(300.0, atr * 2.0)
    resistance = float(swing_highs[0]) if swing_highs else live_price + max(300.0, atr * 2.0)

    # Final sanity: if a level is absurdly far, replace with ATR live zone.
    if abs(support - live_price) / live_price > 0.08:
        support = live_price - max(300.0, atr * 2.0)
    if abs(resistance - live_price) / live_price > 0.08:
        resistance = live_price + max(300.0, atr * 2.0)

    return structure, float(support), float(resistance), "Recent swing levels"

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


def recency_weight(dt):
    """
    Recent data should matter more for short-term BTC futures.
    <= 90 days: 1.00
    <= 365 days: 0.70
    <= 3 years: 0.35
    Older: 0.15
    """
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    d = pd.to_datetime(dt)
    if d.tzinfo is None:
        d = d.tz_localize("Asia/Kolkata")
    age_days = (now - d).days
    if age_days <= 90:
        return 1.00
    if age_days <= 365:
        return 0.70
    if age_days <= 365 * 3:
        return 0.35
    return 0.15


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

    outcomes = []
    weights = []
    for idx in s.index:
        outcome = first_hit(df, idx, direction, tp, sl, horizon)
        if outcome:
            outcomes.append(outcome)
            weights.append(recency_weight(df.loc[idx, "datetime"]))

    if not outcomes:
        return {"direction": direction, "matches": 0, "wins": 0, "losses": 0,
                "no_hit": 0, "ambiguous": 0, "decided": 0, "probability": 0.0,
                "weighted_probability": 0.0, "weighted_decided": 0.0}

    ser = pd.Series(outcomes)
    w = pd.Series(weights)

    wins = int((ser == "Win").sum())
    losses = int((ser == "Loss").sum())
    no_hit = int((ser == "No hit").sum())
    ambiguous = int((ser == "Ambiguous").sum())
    decided = wins + losses
    raw_prob = wins / decided * 100 if decided else 0.0

    win_w = float(w[ser == "Win"].sum())
    loss_w = float(w[ser == "Loss"].sum())
    weighted_decided = win_w + loss_w
    weighted_prob = win_w / weighted_decided * 100 if weighted_decided else 0.0

    return {"direction": direction, "matches": len(outcomes), "wins": wins, "losses": losses,
            "no_hit": no_hit, "ambiguous": ambiguous, "decided": decided,
            "probability": raw_prob, "weighted_probability": weighted_prob,
            "weighted_decided": weighted_decided}


def advanced_market_context(df, live_price):
    """
    Current-market only features. Fast and safe.
    - BOS: Break of Structure
    - CHOCH: Change of Character
    - Liquidity sweep
    - Volume spike
    - High-volume zone around recent price buckets
    """
    d = df.tail(480).reset_index(drop=True)
    if len(d) < 80:
        return {
            "bos": "None", "choch": "None", "sweep": "None",
            "volume_spike": "No", "volume_zone": "Unknown",
            "hv_support": np.nan, "hv_resistance": np.nan
        }

    # Swing highs/lows
    swing_highs = []
    swing_lows = []
    for i in range(3, len(d)-3):
        if d.loc[i, "high"] == d.loc[i-3:i+3, "high"].max():
            swing_highs.append((i, float(d.loc[i, "high"])))
        if d.loc[i, "low"] == d.loc[i-3:i+3, "low"].min():
            swing_lows.append((i, float(d.loc[i, "low"])))

    close = float(d["close"].iloc[-1])
    prev_high = max([x[1] for x in swing_highs[-8:]], default=float(d["high"].tail(80).max()))
    prev_low = min([x[1] for x in swing_lows[-8:]], default=float(d["low"].tail(80).min()))

    bos = "None"
    if close > prev_high:
        bos = "Bullish BOS"
    elif close < prev_low:
        bos = "Bearish BOS"

    # CHOCH based on prior structure then opposite break
    choch = "None"
    prior_structure = "Mixed"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1] > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1] < swing_lows[-2][1]
        if hh and hl:
            prior_structure = "Bullish"
        elif lh and ll:
            prior_structure = "Bearish"

    if prior_structure == "Bullish" and close < prev_low:
        choch = "Bearish CHOCH"
    elif prior_structure == "Bearish" and close > prev_high:
        choch = "Bullish CHOCH"

    # Liquidity sweep: current candle takes old high/low but closes back inside
    last = d.iloc[-1]
    prev_80_high = float(d["high"].iloc[:-1].tail(80).max())
    prev_80_low = float(d["low"].iloc[:-1].tail(80).min())
    sweep = "None"
    if last["high"] > prev_80_high and last["close"] < prev_80_high:
        sweep = "Bearish sweep"
    elif last["low"] < prev_80_low and last["close"] > prev_80_low:
        sweep = "Bullish sweep"

    # Volume spike
    vol_avg = float(d["volume"].tail(50).mean())
    vol_now = float(d["volume"].iloc[-1])
    volume_spike = "Yes" if vol_avg > 0 and vol_now >= vol_avg * 1.75 else "No"

    # Simple volume profile zone: bucket recent closes and sum volume by price bucket
    recent = d.tail(320).copy()
    atr = float(recent["atr14"].iloc[-1]) if "atr14" in recent.columns and pd.notna(recent["atr14"].iloc[-1]) else max(100.0, live_price * 0.002)
    bucket = max(50, round(atr / 2 / 50) * 50)
    recent["price_bucket"] = (recent["close"] / bucket).round() * bucket
    vp = recent.groupby("price_bucket")["volume"].sum().sort_values(ascending=False)

    hv_support = np.nan
    hv_resistance = np.nan
    if not vp.empty:
        buckets = list(vp.index.astype(float))
        below = sorted([b for b in buckets if b < live_price], reverse=True)
        above = sorted([b for b in buckets if b > live_price])
        hv_support = below[0] if below else np.nan
        hv_resistance = above[0] if above else np.nan

    # Is live price inside or near a high-volume zone?
    volume_zone = "Neutral"
    if not np.isnan(hv_support) and abs(live_price - hv_support) <= bucket:
        volume_zone = "Near HV support"
    if not np.isnan(hv_resistance) and abs(live_price - hv_resistance) <= bucket:
        volume_zone = "Near HV resistance"

    return {
        "bos": bos,
        "choch": choch,
        "sweep": sweep,
        "volume_spike": volume_spike,
        "volume_zone": volume_zone,
        "hv_support": hv_support,
        "hv_resistance": hv_resistance
    }


def score_side(direction, res, latest, mtf, structure, support, resistance, price, tp, ctx):
    score = 0
    reasons = []
    p, d = res.get("weighted_probability", res["probability"]), res["decided"]

    # Historical edge
    if p >= 72:
        score += 38; reasons.append("Excellent weighted historical probability")
    elif p >= 68:
        score += 32; reasons.append("Very strong weighted historical probability")
    elif p >= 64:
        score += 26; reasons.append("Strong weighted historical probability")
    elif p >= 60:
        score += 18; reasons.append("Acceptable weighted historical probability")
    elif p >= 55:
        score += 8; reasons.append("Mild historical edge")

    # Sample size
    if d >= 300:
        score += 18; reasons.append("Large sample size")
    elif d >= 150:
        score += 12; reasons.append("Good sample size")
    elif d >= 60:
        score += 7; reasons.append("Minimum sample size")

    # Trend / structure alignment
    if (direction == "LONG" and mtf == "Bullish") or (direction == "SHORT" and mtf == "Bearish"):
        score += 12; reasons.append("MTF aligned")

    if (direction == "LONG" and "Bullish" in structure) or (direction == "SHORT" and "Bearish" in structure):
        score += 10; reasons.append("Swing structure aligned")

    # BOS / CHOCH
    if direction == "LONG" and ctx.get("bos") == "Bullish BOS":
        score += 10; reasons.append("Bullish BOS")
    if direction == "SHORT" and ctx.get("bos") == "Bearish BOS":
        score += 10; reasons.append("Bearish BOS")

    if direction == "LONG" and ctx.get("choch") == "Bullish CHOCH":
        score += 10; reasons.append("Bullish CHOCH")
    if direction == "SHORT" and ctx.get("choch") == "Bearish CHOCH":
        score += 10; reasons.append("Bearish CHOCH")

    # Liquidity sweep
    if direction == "LONG" and ctx.get("sweep") == "Bullish sweep":
        score += 12; reasons.append("Bullish liquidity sweep")
    if direction == "SHORT" and ctx.get("sweep") == "Bearish sweep":
        score += 12; reasons.append("Bearish liquidity sweep")

    # Trend strength + VWAP
    if latest["adx14"] >= 22:
        score += 6; reasons.append("ADX trend strength acceptable")
    if (direction == "LONG" and price > latest["vwap"]) or (direction == "SHORT" and price < latest["vwap"]):
        score += 6; reasons.append("VWAP aligned")

    # Volume spike
    if ctx.get("volume_spike") == "Yes":
        score += 5; reasons.append("Volume expansion present")

    # Volume profile style zones
    if direction == "LONG" and ctx.get("volume_zone") == "Near HV support":
        score += 8; reasons.append("Near high-volume support")
    if direction == "SHORT" and ctx.get("volume_zone") == "Near HV resistance":
        score += 8; reasons.append("Near high-volume resistance")

    # Fixed 300 TP proximity filters
    if direction == "LONG" and (resistance - price) < tp:
        score -= 20; reasons.append("Resistance too close for fixed 300 TP")
    if direction == "SHORT" and (price - support) < tp:
        score -= 20; reasons.append("Support too close for fixed 300 TP")

    # Avoid opposite strong signal
    if direction == "LONG" and ctx.get("sweep") == "Bearish sweep":
        score -= 10; reasons.append("Opposite bearish sweep")
    if direction == "SHORT" and ctx.get("sweep") == "Bullish sweep":
        score -= 10; reasons.append("Opposite bullish sweep")

    return max(0, min(100, score)), reasons

def decide(lr, sr, ls, ss, min_prob, min_samples, min_gap, min_score):
    lp = lr.get("weighted_probability", lr["probability"]) if lr["decided"] >= min_samples else 0
    sp = sr.get("weighted_probability", sr["probability"]) if sr["decided"] >= min_samples else 0
    if lr["decided"] < min_samples and sr["decided"] < min_samples:
        return "NO TRADE", "Low sample size"
    if lp >= min_prob and ls >= min_score and (lp-sp) >= min_gap:
        return "LONG", "Long weighted probability and score passed"
    if sp >= min_prob and ss >= min_score and (sp-lp) >= min_gap:
        return "SHORT", "Short weighted probability and score passed"
    return "NO TRADE", "No high-quality edge"


st.title("BTCUSDT 300-Point Final Weighted Pro Predictor")
st.caption("Final weighted pro version: 300-point fixed target + BOS/CHOCH + liquidity sweep + volume-zone confirmation.")

try:
    live_price, live_src = fetch_price()
except Exception:
    live_price, live_src = np.nan, "Live price unavailable"

with st.sidebar:
    st.header("Automatic settings")
    years = st.slider("Historical years", 1, 5, 3)
    refresh = st.selectbox("Auto-refresh seconds", [0, 60, 120, 300], index=1)
    force_rebuild = st.checkbox("Force rebuild database", False)

    st.header("Trade system")
    tp_points = st.number_input("Target points", value=300, min_value=300, max_value=300, step=50, disabled=True)
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
if not np.isnan(live_price):
    gap = abs(float(df15_raw['close'].iloc[-1]) - live_price) / live_price * 100
    if gap > 3:
        st.warning(f"Archive latest close differs from CoinDCX live price by {gap:.1f}%. Support/resistance fallback will use live price area.")

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
structure, support, resistance, level_mode = structure_levels(df15, price, latest['atr14'])

archive_gap_pct = abs(float(df15_raw["close"].iloc[-1]) - price) / price * 100
if archive_gap_pct > 3:
    st.warning(
        f"Historical archive latest close differs from CoinDCX live by {archive_gap_pct:.1f}%. "
        "Support/resistance is shown as an ATR live zone. Signal engine still uses historical pattern matching, "
        "but trade should be treated as lower-confidence until archive catches up."
    )

ctx = advanced_market_context(df15, price)

long_r = analyze(df15, price, "LONG", tp_points, sl_points, horizon, price_tol, session_match)
short_r = analyze(df15, price, "SHORT", tp_points, sl_points, horizon, price_tol, session_match)
lscore, lreasons = score_side("LONG", long_r, latest, mtf, structure, support, resistance, price, tp_points, ctx)
sscore, sreasons = score_side("SHORT", short_r, latest, mtf, structure, support, resistance, price, tp_points, ctx)
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

c9, c10, c11 = st.columns(3)
c9.metric("Support", f"{support:,.0f}")
c10.metric("Resistance", f"{resistance:,.0f}")
c11.metric("Level mode", level_mode)

m1, m2, m3, m4 = st.columns(4)
m1.metric("BOS", ctx.get("bos", "None"))
m2.metric("CHOCH", ctx.get("choch", "None"))
m3.metric("Liquidity sweep", ctx.get("sweep", "None"))
m4.metric("Volume zone", ctx.get("volume_zone", "Neutral"))

st.subheader("Final signal")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Signal", signal)
s2.metric("Long weighted probability", f"{long_r.get('weighted_probability', long_r['probability']):.1f}%")
s3.metric("Short weighted probability", f"{short_r.get('weighted_probability', short_r['probability']):.1f}%")
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
st.caption("Weighted probability gives higher importance to recent market behavior: last 90 days > last 1 year > older history.")

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

st.warning("Decision-support only. BTC futures are risky. This app rejects weak trades and cannot guarantee profit or accuracy.")
