"""
backtest_rate_bias.py
=====================
中央銀行政策金利バイアス検証バックテスト

【概要】
  中央銀行の利上げ/利下げ発表後、次の1時間足確定からロットサイズに
  方向バイアスを加える効果を検証する。

【ロジック】
  - 利上げ（ベース通貨）→ ロングに有利 → ロング: ×1.1、ショート: ×0.9
  - 利下げ（ベース通貨）→ ショートに有利 → ロング: ×0.9、ショート: ×1.1
  - 利上げ（クォート通貨）→ ショートに有利（逆）
  - 据え置き → バイアスなし（×1.0）

【バイアスの持続】
  次の同中銀の発表まで維持（例: FOMCの利上げ → 次のFOMCまでロングバイアス）

【対象】
  採用7銘柄 × {バイアスなし, バイアスあり} 比較
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
RR_RATIO      = 2.5
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000
INIT_CASH     = 1_000_000  # 100万円

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
E0_WINDOW_MIN   = 2
ADX_MIN         = 20
STREAK_MIN      = 4

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 中央銀行政策金利データ（2025/1〜2026/3） ──────────────────────
# format: (date_str, action, rate_after, change_bp)
# action: "hike" / "cut" / "hold"

# Federal Reserve (FOMC) — 2025: 据え置き→Q3-Q4に3回利下げ(-75bp)
FED_DECISIONS = [
    ("2025-01-29", "hold",  4.375, 0),
    ("2025-03-19", "hold",  4.375, 0),
    ("2025-05-07", "hold",  4.375, 0),
    ("2025-06-18", "hold",  4.375, 0),
    ("2025-07-30", "hold",  4.375, 0),
    ("2025-09-17", "cut",   4.125, -25),
    ("2025-10-29", "cut",   3.875, -25),
    ("2025-12-10", "cut",   3.625, -25),
    ("2026-01-28", "hold",  3.625, 0),
]

# Bank of Japan (BOJ) — 2025: 1月+12月に利上げ（世界的利下げ局面で逆行）
BOJ_DECISIONS = [
    ("2025-01-24", "hike",  0.50, +25),
    ("2025-03-19", "hold",  0.50, 0),
    ("2025-05-01", "hold",  0.50, 0),
    ("2025-06-18", "hold",  0.50, 0),
    ("2025-07-31", "hold",  0.50, 0),
    ("2025-09-19", "hold",  0.50, 0),
    ("2025-10-30", "hold",  0.50, 0),
    ("2025-12-19", "hike",  0.75, +25),
    ("2026-01-23", "hold",  0.75, 0),
]

# European Central Bank (ECB) — 2025: 3回利下げ(-75bp), Q3以降据え置き
ECB_DECISIONS = [
    ("2025-01-30", "hold",  2.75, 0),
    ("2025-03-06", "cut",   2.50, -25),
    ("2025-04-17", "cut",   2.25, -25),
    ("2025-06-05", "cut",   2.00, -25),
    ("2025-07-24", "hold",  2.00, 0),
    ("2025-09-11", "hold",  2.00, 0),
    ("2025-10-30", "hold",  2.00, 0),
    ("2025-12-18", "hold",  2.00, 0),
    ("2026-02-05", "hold",  2.00, 0),
]

# Bank of England (BOE) — 2025: 4回利下げ(-100bp)
BOE_DECISIONS = [
    ("2025-02-06", "cut",   4.50, -25),
    ("2025-03-20", "hold",  4.50, 0),
    ("2025-05-08", "cut",   4.25, -25),
    ("2025-06-19", "hold",  4.25, 0),
    ("2025-08-07", "cut",   4.00, -25),
    ("2025-09-18", "hold",  4.00, 0),
    ("2025-11-06", "hold",  4.00, 0),
    ("2025-12-19", "cut",   3.75, -25),
    ("2026-02-05", "hold",  3.75, 0),
]

# Reserve Bank of Australia (RBA) — 2025: 3回利下げ(-75bp), 2026/2に利上げ
RBA_DECISIONS = [
    ("2025-02-05", "cut",   4.10, -25),
    ("2025-04-01", "hold",  4.10, 0),
    ("2025-05-07", "cut",   3.85, -25),
    ("2025-06-03", "hold",  3.85, 0),
    ("2025-07-01", "hold",  3.85, 0),
    ("2025-08-05", "cut",   3.60, -25),
    ("2025-09-02", "hold",  3.60, 0),
    ("2025-10-07", "hold",  3.60, 0),
    ("2025-11-04", "hold",  3.60, 0),
    ("2025-12-03", "hold",  3.60, 0),
    ("2026-02-03", "hike",  3.85, +25),
]

# Reserve Bank of New Zealand (RBNZ) — 2025: 6回利下げ(-175bp, 50bp刻みあり)
RBNZ_DECISIONS = [
    ("2025-02-19", "cut",   3.75, -50),
    ("2025-04-09", "cut",   3.50, -25),
    ("2025-05-28", "cut",   3.25, -25),
    ("2025-07-09", "hold",  3.25, 0),
    ("2025-08-06", "cut",   3.00, -25),
    ("2025-10-08", "cut",   2.50, -50),
    ("2025-11-26", "cut",   2.25, -25),
    ("2026-02-04", "hold",  2.25, 0),
]

# Bank of Canada (BOC) — 2025: 5回利下げ(-125bp)
BOC_DECISIONS = [
    ("2025-01-29", "cut",   3.25, -25),
    ("2025-03-12", "cut",   3.00, -25),
    ("2025-04-16", "cut",   2.75, -25),
    ("2025-06-04", "hold",  2.75, 0),
    ("2025-07-30", "hold",  2.75, 0),
    ("2025-09-17", "cut",   2.50, -25),
    ("2025-10-29", "cut",   2.25, -25),
    ("2025-12-10", "hold",  2.25, 0),
    ("2026-01-28", "hold",  2.25, 0),
]

# ── 通貨ペアと中央銀行のマッピング ─────────────────────────────────
# (base_bank_decisions, quote_bank_decisions)
# base通貨の利上げ → ロングバイアス（ペア価格上昇）
# quote通貨の利上げ → ショートバイアス（ペア価格下落）
PAIR_BANKS = {
    "USDJPY": (FED_DECISIONS, BOJ_DECISIONS),   # USD/JPY: Fed=base, BOJ=quote
    "EURUSD": (ECB_DECISIONS, FED_DECISIONS),   # EUR/USD: ECB=base, Fed=quote
    "GBPUSD": (BOE_DECISIONS, FED_DECISIONS),   # GBP/USD: BOE=base, Fed=quote
    "AUDUSD": (RBA_DECISIONS, FED_DECISIONS),   # AUD/USD: RBA=base, Fed=quote
    "NZDUSD": (RBNZ_DECISIONS, FED_DECISIONS),  # NZD/USD: RBNZ=base, Fed=quote
    "USDCAD": (FED_DECISIONS, BOC_DECISIONS),   # USD/CAD: Fed=base, BOC=quote
    "XAUUSD": ([], FED_DECISIONS),              # XAU/USD: Gold=base(なし), Fed=quote
    # XAUUSD: Fed利上げ→ドル高→ゴールド安→ショートバイアス
}

# ── バイアス計算 ──────────────────────────────────────────────────
def build_rate_bias_series(pair, start_ts, end_ts):
    """
    指定ペアの政策金利バイアスを時系列で構築する。

    Returns
    -------
    list of (effective_from_ts, long_mult, short_mult)
        effective_from_ts: バイアスが有効になるタイムスタンプ（発表日の次の1H確定後）
        long_mult: ロングロットに掛ける倍率
        short_mult: ショートロットに掛ける倍率
    """
    if pair not in PAIR_BANKS:
        return []

    base_decisions, quote_decisions = PAIR_BANKS[pair]
    events = []

    # base通貨: 利上げ→ロングバイアス, 利下げ→ショートバイアス
    for date_str, action, rate, change_bp in base_decisions:
        ts = pd.Timestamp(date_str, tz="UTC")
        # 発表は通常午後 → 次の1H足確定は翌日の最初の1H足
        # 安全策: 発表日の翌日 00:00 UTC から適用
        effective = ts + pd.Timedelta(hours=24)
        if action == "hike":
            events.append((effective, +1, "base_hike"))
        elif action == "cut":
            events.append((effective, -1, "base_cut"))
        else:
            events.append((effective, 0, "base_hold"))

    # quote通貨: 利上げ→ショートバイアス, 利下げ→ロングバイアス（逆方向）
    for date_str, action, rate, change_bp in quote_decisions:
        ts = pd.Timestamp(date_str, tz="UTC")
        effective = ts + pd.Timedelta(hours=24)
        if action == "hike":
            events.append((effective, -1, "quote_hike"))
        elif action == "cut":
            events.append((effective, +1, "quote_cut"))
        else:
            events.append((effective, 0, "quote_hold"))

    # 時系列順にソート
    events.sort(key=lambda x: x[0])
    return events


def get_lot_multiplier(entry_time, events, direction, bias_pct=0.10):
    """
    エントリー時点でのロット倍率を返す。

    Parameters
    ----------
    entry_time : pd.Timestamp
        エントリー時刻
    events : list
        build_rate_bias_series() の出力
    direction : int
        1=ロング, -1=ショート
    bias_pct : float
        バイアス幅（デフォルト10%）

    Returns
    -------
    float
        ロット倍率（1.0 ± bias_pct）
    """
    if not events:
        return 1.0

    # 直近のbase/quoteイベントを個別に追跡
    net_bias = 0  # +1=ロングバイアス, -1=ショートバイアス

    # 最新のbase/quoteそれぞれの最新バイアスを取得
    latest_base_bias = 0
    latest_quote_bias = 0

    for eff_ts, bias_dir, event_type in events:
        if eff_ts > entry_time:
            break
        if event_type.startswith("base"):
            latest_base_bias = bias_dir
        else:
            latest_quote_bias = bias_dir

    net_bias = latest_base_bias + latest_quote_bias

    # net_bias > 0 → ロング有利、< 0 → ショート有利
    if net_bias > 0:
        # ロングバイアス
        if direction == 1:
            return 1.0 + bias_pct * abs(net_bias)
        else:
            return 1.0 - bias_pct * abs(net_bias)
    elif net_bias < 0:
        # ショートバイアス
        if direction == -1:
            return 1.0 + bias_pct * abs(net_bias)
        else:
            return 1.0 - bias_pct * abs(net_bias)
    else:
        return 1.0


# ── 採用7銘柄 ────────────────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "EURUSD",  "logic": "C", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "GBPUSD",  "logic": "A", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "USDCAD",  "logic": "A", "risk_pct": 0.02, "tol": 0.30},
    {"sym": "NZDUSD",  "logic": "A", "risk_pct": 0.02, "tol": 0.20},
    {"sym": "XAUUSD",  "logic": "A", "risk_pct": 0.02, "tol": 0.20},
    {"sym": "AUDUSD",  "logic": "B", "risk_pct": 0.02, "tol": 0.30},
]

LOGIC_NAMES = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}

# ── データロード（backtest_lot_cap.pyと同一） ────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def load_all(sym):
    sym_l = sym.lower()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p):
            d1m = load_csv(p); break
    else:
        return None, None
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_4h.csv")]:
        if os.path.exists(p):
            return d1m, load_csv(p)
    p_is  = os.path.join(DATA_DIR, f"{sym_l}_is_4h.csv")
    p_oos = os.path.join(DATA_DIR, f"{sym_l}_oos_4h.csv")
    if os.path.exists(p_is) and os.path.exists(p_oos):
        d4h = pd.concat([load_csv(p_is), load_csv(p_oos)])
        return d1m, d4h[~d4h.index.duplicated(keep="first")].sort_index()
    d4h = d1m.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    return d1m, d4h

# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    h = df["high"]; l = df["low"]
    pdm = h.diff().clip(lower=0); mdm = (-l.diff()).clip(lower=0)
    pdm[pdm < mdm] = 0.0; mdm[mdm < pdm] = 0.0
    atr = calc_atr(df, 1).ewm(alpha=1/n, adjust=False).mean()
    dip = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dim = 100 * mdm.ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)

def build_4h(df4h, need_1d=False):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]   = calc_adx(df, 14)
    d1 = None
    if need_1d:
        d1 = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df):
    r = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    r["atr"]   = calc_atr(r, 14)
    r["ema20"] = r["close"].ewm(span=20, adjust=False).mean()
    return r

# ── エントリー ────────────────────────────────────────────────────
def pick_e0(t, sp, direction, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E0_WINDOW_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

def pick_e1(t, direction, sp, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=E1_MAX_WAIT_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        o = m1c["opens"][i]; c = m1c["closes"][i]
        if direction == 1 and c <= o: continue
        if direction == -1 and c >= o: continue
        ni = i + 1
        if ni >= len(idx): return None, None
        return idx[ni], m1c["opens"][ni] + (sp if direction == 1 else -sp)
    return None, None

def pick_e2(t, direction, sp, atr_d, m1c):
    idx = m1c["idx"]
    s = idx.searchsorted(t, side="left")
    e = idx.searchsorted(t + pd.Timedelta(minutes=max(2, E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        rng = m1c["highs"][i] - m1c["lows"][i]
        av  = atr_d.get(idx[i], np.nan)
        if not np.isnan(av) and rng > av * E2_SPIKE_ATR: continue
        return idx[i], m1c["opens"][i] + (sp if direction == 1 else -sp)
    return None, None

# ── フィルター ────────────────────────────────────────────────────
def chk_kmid(b, d): return (d == 1 and b["close"] > b["open"]) or (d == -1 and b["close"] < b["open"])
def chk_klow(b): return (min(b["open"], b["close"]) - b["low"]) / b["open"] < KLOW_THR if b["open"] > 0 else False
def chk_ema(b): return not pd.isna(b["atr"]) and b["atr"] > 0 and abs(b["close"] - b["ema20"]) >= b["atr"] * A1_EMA_DIST_MIN

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, tol_factor=0.30):
    d4h, d1d = build_4h(d4h_full, need_1d=(logic == "A"))
    d1h = build_1h(d1m)
    signals = []; used = set()

    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue

        h4b = d4h[d4h.index < hct]
        if len(h4b) < max(2, STREAK_MIN): continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        if logic == "A":
            if d1d is None: continue
            d1b = d1d[d1d.index.normalize() < hct.normalize()]
            if not len(d1b) or d1b.iloc[-1]["trend1d"] != trend: continue
        elif logic == "B":
            if h4l.get("adx", 0) < ADX_MIN: continue
            if not all(t == trend for t in h4b["trend"].iloc[-STREAK_MIN:].values): continue
            body = abs(h4l["close"] - h4l["open"])
            rng  = h4l["high"] - h4l["low"]
            if rng > 0 and body / rng < 0.3: continue

        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if logic != "C" and not chk_ema(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol_factor: continue

        if logic == "C":
            if d == 1 and p1["close"] <= p1["open"]: continue
            if d == -1 and p1["close"] >= p1["open"]: continue

        if logic == "A":   et, ep = pick_e2(hct, d, spread, atr_d, m1c)
        elif logic == "C": et, ep = pick_e0(hct, spread, d, m1c)
        else:              et, ep = pick_e1(hct, d, spread, m1c)

        if et is None or et in used: continue
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * RR_RATIO, "risk": risk})
            used.add(et)

    return sorted(signals, key=lambda x: x["time"])

# ── シミュレーション（金利バイアス対応版） ────────────────────────
def _exit(highs, lows, ep, sl, tp, risk, d):
    half = ep + d * risk * HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss", False
            if h  >= tp: return tp, "win",  False
            if h  >= half:
                for j in range(i+1, lim):
                    if lows[j]  <= ep: return ep, "win", True
                    if highs[j] >= tp: return tp, "win", True
                return None, None, True
        else:
            if h  >= sl: return sl, "loss", False
            if lo <= tp: return tp, "win",  False
            if lo <= half:
                for j in range(i+1, lim):
                    if highs[j] >= ep: return ep, "win", True
                    if lows[j]  <= tp: return tp, "win", True
                return None, None, True
    return None, None, False

def simulate(signals, d1m, sym, risk_pct, rate_events=None, bias_pct=0.10):
    """
    バックテスト実行（金利バイアス対応版）。

    Parameters
    ----------
    rate_events : list or None
        build_rate_bias_series() の出力。Noneの場合はバイアスなし。
    bias_pct : float
        バイアス幅（デフォルト10%）
    """
    if not signals:
        return [], INIT_CASH, 0.0, {}

    rm = RiskManager(sym, risk_pct=risk_pct)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    bias_stats = {"biased_long": 0, "biased_short": 0, "neutral": 0}

    for sig in signals:
        rm.risk_pct = risk_pct
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)

        # 金利バイアス適用
        if rate_events:
            mult = get_lot_multiplier(sig["time"], rate_events, sig["dir"], bias_pct)
            lot *= mult
            if mult > 1.0:
                if sig["dir"] == 1: bias_stats["biased_long"] += 1
                else: bias_stats["biased_short"] += 1
            elif mult < 1.0:
                if sig["dir"] == 1: bias_stats["biased_long"] += 1
                else: bias_stats["biased_short"] += 1
            else:
                bias_stats["neutral"] += 1
        else:
            bias_stats["neutral"] += 1

        sp = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        xp, result, half_done = _exit(m1h[sp:], m1l[sp:],
                                       sig["ep"], sig["sl"], sig["tp"],
                                       sig["risk"], sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            equity  += half_pnl; rem = lot * 0.5
        else:
            rem = lot

        pnl    = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total  = half_pnl + pnl
        trades.append({"result": result, "pnl": total, "dir": sig["dir"],
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, bias_stats

# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades):
    if len(trades) < 5: return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    gw   = wins.sum(); gl = abs(loss.sum())
    pf   = gw / gl if gl > 0 else float("inf")

    monthly = df.groupby("month")["pnl"].sum()
    plus_m  = (monthly > 0).sum()

    eq = INIT_CASH
    monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq if eq > 0 else 0
        monthly_ret.append(ret)
        eq += monthly[m]
    mr     = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0

    avg_w  = wins.mean() if len(wins) > 0 else 0
    avg_l  = abs(loss.mean()) if len(loss) > 0 else 1
    kelly  = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0

    # 方向別WR
    long_trades  = df[df["dir"] == 1]
    short_trades = df[df["dir"] == -1]
    long_wr  = (long_trades["pnl"] > 0).mean() if len(long_trades) > 0 else 0
    short_wr = (short_trades["pnl"] > 0).mean() if len(short_trades) > 0 else 0

    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "final_eq": eq,
            "long_n": len(long_trades), "long_wr": long_wr,
            "short_n": len(short_trades), "short_wr": short_wr}

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*120)
    print("  中央銀行政策金利バイアス 検証バックテスト")
    print("  利上げ → ロング+10% | 利下げ → ショート+10% | 発表翌日の1H確定から適用")
    print("="*120)

    # バイアス幅の比較: 10%, 15%, 20%
    BIAS_LEVELS = [0.10, 0.15, 0.20]

    all_results = []

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        lname = LOGIC_NAMES[logic]
        tol   = tgt["tol"]

        print(f"\n{'─'*110}")
        print(f"  {sym}  Logic-{logic}:{lname}  tol={tol}")

        d1m, d4h = load_all(sym)
        if d1m is None:
            print("  データ未発見 → スキップ"); continue

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs":  d1m["high"].values, "lows": d1m["low"].values}

        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, tol_factor=tol)
        print(f"  シグナル数: {len(sigs)}")
        if len(sigs) < 5:
            print("  シグナル不足 → スキップ"); continue

        # 金利バイアスイベント構築
        rate_events = build_rate_bias_series(
            sym,
            d1m.index[0] if len(d1m) > 0 else pd.Timestamp("2025-01-01", tz="UTC"),
            d1m.index[-1] if len(d1m) > 0 else pd.Timestamp("2026-03-15", tz="UTC"),
        )
        n_events = len([e for e in rate_events if e[1] != 0])
        print(f"  金利イベント数: {n_events} (変更のみ)")
        print(f"{'─'*110}")

        # ベースライン（バイアスなし）
        tr_base, eq_base, mdd_base, _ = simulate(sigs, d1m, sym, tgt["risk_pct"])
        st_base = calc_stats(tr_base)
        if not st_base:
            print("  統計不足"); continue

        pf_base = min(st_base["pf"], 99.99)
        print(f"\n  {'モード':12} | {'n':>5} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'MDD':>7} "
              f"{'月+':>5} | {'Long':>5} {'L-WR':>6} {'Short':>5} {'S-WR':>6} | "
              f"{'最終資産':>14} {'差分':>8}")
        print(f"  {'-'*106}")

        print(f"  {'バイアスなし':12} | "
              f"{st_base['n']:>5} {st_base['wr']*100:>5.1f}% {pf_base:>6.2f} "
              f"{st_base['sharpe']:>7.2f} {mdd_base:>6.1f}% "
              f"{st_base['plus_m']:>2}/{st_base['total_m']:<2} | "
              f"{st_base['long_n']:>5} {st_base['long_wr']*100:>5.1f}% "
              f"{st_base['short_n']:>5} {st_base['short_wr']*100:>5.1f}% | "
              f"{eq_base:>14,.0f} {'基準':>8}")

        for bias_pct in BIAS_LEVELS:
            label = f"±{int(bias_pct*100)}%バイアス"
            tr_b, eq_b, mdd_b, bstats = simulate(
                sigs, d1m, sym, tgt["risk_pct"], rate_events, bias_pct)
            st_b = calc_stats(tr_b)
            if not st_b: continue

            pf_b    = min(st_b["pf"], 99.99)
            pf_diff = pf_b - pf_base
            eq_diff = (eq_b - eq_base) / eq_base * 100

            print(f"  {label:12} | "
                  f"{st_b['n']:>5} {st_b['wr']*100:>5.1f}% {pf_b:>6.2f} "
                  f"{st_b['sharpe']:>7.2f} {mdd_b:>6.1f}% "
                  f"{st_b['plus_m']:>2}/{st_b['total_m']:<2} | "
                  f"{st_b['long_n']:>5} {st_b['long_wr']*100:>5.1f}% "
                  f"{st_b['short_n']:>5} {st_b['short_wr']*100:>5.1f}% | "
                  f"{eq_b:>14,.0f} {eq_diff:>+7.1f}%")

            all_results.append({
                "sym": sym, "logic": logic, "bias_pct": bias_pct,
                "base_pf": pf_base, "base_sharpe": st_base["sharpe"],
                "base_eq": eq_base, "base_mdd": mdd_base,
                "bias_pf": pf_b, "bias_sharpe": st_b["sharpe"],
                "bias_eq": eq_b, "bias_mdd": mdd_b,
                "pf_diff": pf_diff, "eq_diff_pct": eq_diff,
                "long_n": st_b["long_n"], "long_wr": st_b["long_wr"],
                "short_n": st_b["short_n"], "short_wr": st_b["short_wr"],
            })

    # ── 総合まとめ ────────────────────────────────────────────────
    print("\n" + "="*120)
    print("  ■ 金利バイアス効果まとめ（±10%）")
    print("="*120)

    df_res = pd.DataFrame(all_results)
    if df_res.empty:
        print("  結果なし"); return

    b10 = df_res[df_res["bias_pct"] == 0.10]
    if not b10.empty:
        print(f"\n  {'銘柄':8} | {'PF(基準)':>8} {'PF(+10%)':>10} {'差分':>8} | "
              f"{'Sharpe基準':>10} {'Sharpe+10%':>12} | {'資産差':>8} | {'判定':>6}")
        print(f"  {'-'*90}")
        for _, r in b10.iterrows():
            verdict = "✅改善" if r["pf_diff"] > 0 and r["eq_diff_pct"] > 0 else \
                      "⚠️微差" if abs(r["pf_diff"]) < 0.05 else "❌悪化"
            print(f"  {r['sym']:8} | {r['base_pf']:>8.2f} {r['bias_pf']:>10.2f} "
                  f"{r['pf_diff']:>+8.2f} | {r['base_sharpe']:>10.2f} "
                  f"{r['bias_sharpe']:>12.2f} | {r['eq_diff_pct']:>+7.1f}% | {verdict:>6}")

        avg_diff = b10["pf_diff"].mean()
        avg_eq   = b10["eq_diff_pct"].mean()
        print(f"\n  平均: PF差 {avg_diff:+.3f} | 資産差 {avg_eq:+.1f}%")

    # ── CSV保存 ───────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "backtest_rate_bias.csv")
    df_res.to_csv(out, index=False)
    print(f"\n  結果保存: {out}")

if __name__ == "__main__":
    main()
