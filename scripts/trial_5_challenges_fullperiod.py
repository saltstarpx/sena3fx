#!/usr/bin/env python3
"""
YAGAMI改 5試練 — 全期間バックテスト版
======================================
IS+OOS全期間でバックテストエンジンを再実行し、
全トレードログを生成して5試練を実施する。

Trial 1: ホームラン依存度
Trial 2: USD集中リスク
Trial 3: USDCADの真実
Trial 4: 現実的MDD推定
Trial 5: フィルターアブレーション（全期間）
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
RR_RATIO      = 2.5
RR_RATIO_V80  = 3.0
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
E0_WINDOW_MIN   = 2
ADX_MIN         = 20
STREAK_MIN      = 4
BODY_RATIO_MIN  = 0.3

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 採用銘柄×確定ロジック（CLAUDE.mdの最終確定）──────────────────
APPROVED = [
    {"sym": "USDCAD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "XAUUSD", "logic": "A",   "rr": 2.5, "tol": 0.20},
    {"sym": "EURUSD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "AUDUSD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "GBPUSD", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "USDCHF", "logic": "V80", "rr": 3.0, "tol": 0.30},
    {"sym": "NZDUSD", "logic": "A",   "rr": 3.0, "tol": 0.20},
    {"sym": "USDJPY", "logic": "C",   "rr": 3.0, "tol": 0.30},
]

# ── データロード ──────────────────────────────────────────────────
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
def chk_body_ratio(b, min_ratio=BODY_RATIO_MIN):
    rng = b["high"] - b["low"]
    if rng <= 0: return False
    return abs(b["close"] - b["open"]) / rng >= min_ratio

# ── シグナル生成（Logic-A/B/C）──────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, rr=RR_RATIO, tol=A3_DEFAULT_TOL):
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

        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if logic != "C" and not chk_ema(h4l): continue
        if logic == "B" and not chk_body_ratio(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue

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
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)

    return sorted(signals, key=lambda x: x["time"])

# ── シグナル生成（v80）──────────────────────────────────────────
def generate_signals_v80(d1m, d4h_full, spread, m1c, rr=RR_RATIO_V80, tol=A3_DEFAULT_TOL):
    d4h, _ = build_4h(d4h_full, need_1d=False)
    d1h = build_1h(d1m)
    signals = []; used = set()

    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue

        h4b = d4h[d4h.index < hct]
        if len(h4b) < 2: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        if not chk_kmid(h4l, trend): continue
        if not chk_klow(h4l): continue
        if not chk_body_ratio(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue

        et, ep = pick_e0(hct, spread, d, m1c)
        if et is None or et in used: continue

        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)

    return sorted(signals, key=lambda x: x["time"])

# ── アブレーション v80（試練⑤用）────────────────────────────────
def generate_signals_ablation_v80(d1m, d4h_full, spread, m1c,
                                   rr=RR_RATIO_V80, tol=A3_DEFAULT_TOL,
                                   use_kmid=True, use_klow=True, use_body=True):
    d4h, _ = build_4h(d4h_full, need_1d=False)
    d1h = build_1h(d1m)
    signals = []; used = set()

    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue
        h4b = d4h[d4h.index < hct]
        if len(h4b) < 2: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        if use_kmid and not chk_kmid(h4l, trend): continue
        if use_klow and not chk_klow(h4l): continue
        if use_body and not chk_body_ratio(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue

        et, ep = pick_e0(hct, spread, d, m1c)
        if et is None or et in used: continue
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)
    return sorted(signals, key=lambda x: x["time"])

# ── アブレーション Logic-A（試練⑤用）──────────────────────────────
def generate_signals_ablation_a(d1m, d4h_full, spread, atr_d, m1c,
                                 rr=RR_RATIO_V80, tol=A3_DEFAULT_TOL,
                                 use_kmid=True, use_klow=True,
                                 use_1d_trend=True, use_ema_dist=True):
    d4h, d1d = build_4h(d4h_full, need_1d=use_1d_trend)
    d1h = build_1h(d1m)
    signals = []; used = set()

    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue
        h4b = d4h[d4h.index < hct]
        if len(h4b) < 2: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]

        if use_1d_trend:
            if d1d is None: continue
            d1b = d1d[d1d.index.normalize() < hct.normalize()]
            if not len(d1b) or d1b.iloc[-1]["trend1d"] != trend: continue
        if use_kmid and not chk_kmid(h4l, trend): continue
        if use_klow and not chk_klow(h4l): continue
        if use_ema_dist and not chk_ema(h4l): continue

        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue

        et, ep = pick_e2(hct, d, spread, atr_d, m1c)
        if et is None or et in used: continue
        raw = ep - spread if d == 1 else ep + spread
        sl  = (min(v1, v2) - atr1h * 0.15) if d == 1 else (max(v1, v2) + atr1h * 0.15)
        risk = (raw - sl) if d == 1 else (sl - raw)
        if 0 < risk <= h4atr * 2:
            signals.append({"time": et, "dir": d, "ep": ep, "sl": sl,
                            "tp": raw + d * risk * rr, "risk": risk})
            used.add(et)
    return sorted(signals, key=lambda x: x["time"])

# ── シミュレーション（詳細トレードログ付き）──────────────────────
def _exit(highs, lows, ep, sl, tp, risk, d):
    half = ep + d * risk * HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss", False, i
            if h  >= tp: return tp, "win",  False, i
            if h  >= half:
                for j in range(i+1, lim):
                    if lows[j]  <= ep: return ep, "win", True, j
                    if highs[j] >= tp: return tp, "win", True, j
                return None, None, True, lim
        else:
            if h  >= sl: return sl, "loss", False, i
            if lo <= tp: return tp, "win",  False, i
            if lo <= half:
                for j in range(i+1, lim):
                    if highs[j] >= ep: return ep, "win", True, j
                    if lows[j]  <= tp: return tp, "win", True, j
                return None, None, True, lim
    return None, None, False, lim

def simulate_detailed(signals, d1m, sym):
    """Returns detailed trade list with entry/exit times, equity"""
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        rm.risk_pct = 0.02
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        xp, result, half_done, exit_offset = _exit(m1h[sp:], m1l[sp:],
                                                     sig["ep"], sig["sl"], sig["tp"],
                                                     sig["risk"], sig["dir"])
        if result is None: continue

        exit_idx = min(sp + exit_offset, len(m1t) - 1)
        exit_time = m1t[exit_idx]

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
        trades.append({
            "sym": sym,
            "result": result, "pnl": total, "equity": equity,
            "entry_time": sig["time"], "exit_time": exit_time,
            "month": sig["time"].strftime("%Y-%m"),
            "date": sig["time"].strftime("%Y-%m-%d"),
        })
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd

def simulate_simple(signals, d1m, sym):
    """Simple simulate returning stats only (for ablation)"""
    if not signals: return {"n": 0, "wr": 0, "pf": 0, "pnl": 0}
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; wins_pnl = 0; loss_pnl = 0; n_win = 0; n_total = 0

    for sig in signals:
        rm.risk_pct = 0.02
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        xp, result, half_done, _ = _exit(m1h[sp:], m1l[sp:],
                                          sig["ep"], sig["sl"], sig["tp"],
                                          sig["risk"], sig["dir"])
        if result is None: continue
        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            equity += half_pnl; rem = lot * 0.5
        else:
            rem = lot
        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total = half_pnl + pnl
        n_total += 1
        if total > 0:
            wins_pnl += total; n_win += 1
        else:
            loss_pnl += total

    wr = n_win / n_total if n_total > 0 else 0
    pf_val = wins_pnl / abs(loss_pnl) if loss_pnl < 0 else float("inf")
    return {"n": n_total, "wr": wr, "pf": pf_val, "pnl": wins_pnl + loss_pnl}

# ── helpers ──────────────────────────────────────────────────────
def pf(pnl_series):
    gw = pnl_series[pnl_series > 0].sum()
    gl = abs(pnl_series[pnl_series < 0].sum())
    return gw / gl if gl > 0 else float("inf")

def calc_mdd_details(equity_series):
    eq = equity_series.reset_index(drop=True)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    trough_idx = dd.idxmin()
    mdd_pct = dd.iloc[trough_idx] * 100
    peak_val = peak.iloc[trough_idx]
    start_candidates = eq.iloc[:trough_idx+1]
    start_idx = start_candidates[start_candidates == peak_val].index[-1]
    post = eq.iloc[trough_idx:]
    recovery = post[post >= peak_val]
    recovery_idx = recovery.index[0] if len(recovery) > 0 else None
    return mdd_pct, start_idx, trough_idx, recovery_idx

def print_sep(title, f):
    line = "=" * 80
    f.write(f"\n{line}\n  {title}\n{line}\n\n")

# ═══════════════════════════════════════════════════════════════════════
#  Step 1: 全期間バックテスト実行 → ポートフォリオトレードログ生成
# ═══════════════════════════════════════════════════════════════════════
def generate_full_period_trades():
    print("=" * 80)
    print("  全期間バックテスト実行（8銘柄）")
    print("=" * 80)

    all_trades = []
    data_cache = {}

    for tgt in APPROVED:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        rr    = tgt["rr"]
        tol   = tgt["tol"]

        print(f"  {sym:>8} Logic-{logic} RR={rr} tol={tol} ...", end=" ", flush=True)
        t0 = time.time()

        if sym not in data_cache:
            d1m, d4h = load_all(sym)
            if d1m is None:
                print("データなし"); continue
            data_cache[sym] = (d1m, d4h)

        d1m, d4h = data_cache[sym]
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs": d1m["high"].values, "lows": d1m["low"].values}

        if logic == "V80":
            sigs = generate_signals_v80(d1m, d4h, spread, m1c, rr=rr, tol=tol)
        else:
            atr_d = calc_atr(d1m, 10).to_dict()
            sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, rr=rr, tol=tol)

        trades, final_eq, mdd = simulate_detailed(sigs, d1m, sym)
        all_trades.extend(trades)
        elapsed = time.time() - t0
        print(f"{len(trades):>5}トレード  PF={pf(pd.Series([t['pnl'] for t in trades])):.2f}  "
              f"MDD={mdd:.1f}%  ({elapsed:.0f}s)")

    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["date"] = pd.to_datetime(df["date"])
    df["entry_hour"] = df["entry_time"].dt.hour
    df["h4_bucket"] = (df["entry_time"].dt.hour // 4) * 4
    df["h4_key"] = df["entry_time"].dt.strftime("%Y-%m-%d") + "_" + df["h4_bucket"].astype(str).str.zfill(2)

    csv_path = os.path.join(OUT_DIR, "backtest_portfolio_fullperiod.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  全期間トレードログ: {csv_path}")
    print(f"  合計: {len(df)}トレード, 期間: {df['entry_time'].min()} 〜 {df['entry_time'].max()}")

    return df, data_cache

# ═══════════════════════════════════════════════════════════════════════
#  Step 2: 5試練実行
# ═══════════════════════════════════════════════════════════════════════
def run_trials(df, data_cache):
    OUT_PATH = os.path.join(OUT_DIR, "trial_5_challenges_fullperiod_report.txt")
    out = open(OUT_PATH, "w", encoding="utf-8")

    TOTAL_TRADES = len(df)
    TOTAL_PNL = df["pnl"].sum()
    WINS = df[df["result"] == "win"]
    OVERALL_PF = pf(df["pnl"])
    OVERALL_WR = len(WINS) / TOTAL_TRADES

    out.write(f"全期間バックテスト概要\n")
    out.write(f"  トレード数: {TOTAL_TRADES}\n")
    out.write(f"  期間: {df['entry_time'].min()} 〜 {df['entry_time'].max()}\n")
    out.write(f"  PF: {OVERALL_PF:.2f}  WR: {OVERALL_WR:.1%}  総損益: {TOTAL_PNL:,.0f}\n")

    # ── 試練① ────────────────────────────────────────────────────
    print_sep("試練① ホームラン依存度の実態を暴け", out)

    df_sorted = df.sort_values("pnl", ascending=False).reset_index(drop=True)
    n_top10 = int(np.ceil(TOTAL_TRADES * 0.10))
    n_top5  = int(np.ceil(TOTAL_TRADES * 0.05))
    n_top20 = int(np.ceil(TOTAL_TRADES * 0.20))

    top10 = df_sorted.head(n_top10)
    bot90 = df_sorted.iloc[n_top10:]
    top5  = df_sorted.head(n_top5)
    bot95 = df_sorted.iloc[n_top5:]
    top20 = df_sorted.head(n_top20)
    bot80 = df_sorted.iloc[n_top20:]

    out.write(f"全トレード数: {TOTAL_TRADES}  上位10%={n_top10}  上位5%={n_top5}  上位20%={n_top20}\n\n")

    scenarios = [("A: 全トレード", df), ("B: 上位10%除外", bot90),
                 ("C: 上位5%除外", bot95), ("D: 上位20%除外", bot80)]

    out.write(f"{'シナリオ':<22} {'n':>6} {'総損益':>16} {'PF':>8} {'WR':>8}\n")
    out.write("-" * 64 + "\n")
    for name, data in scenarios:
        t_pnl = data["pnl"].sum()
        t_pf = pf(data["pnl"])
        t_wr = len(data[data["result"] == "win"]) / len(data) if len(data) > 0 else 0
        out.write(f"{name:<22} {len(data):>6} {t_pnl:>16,.0f} {t_pf:>8.2f} {t_wr:>7.1%}\n")

    out.write(f"\n上位10%利益占有率: {top10['pnl'].sum()/TOTAL_PNL:.1%}\n")
    out.write(f"上位20%利益占有率: {top20['pnl'].sum()/TOTAL_PNL:.1%}\n")
    out.write(f"上位5%利益占有率:  {top5['pnl'].sum()/TOTAL_PNL:.1%}\n")

    # 月次比較 A vs B
    out.write(f"\n--- 月次損益比較（全トレード vs 上位10%除外）---\n")
    monthly_a = df.groupby("month")["pnl"].sum()
    monthly_b = bot90.groupby("month")["pnl"].sum()
    out.write(f"{'月':>10} {'A: 全':>14} {'B: 10%除外':>14} {'マイナス?':>10}\n")
    out.write("-" * 52 + "\n")
    minus_months = 0
    for m in sorted(monthly_a.index):
        a_val = monthly_a.get(m, 0)
        b_val = monthly_b.get(m, 0)
        mark = "★" if b_val < 0 else ""
        if b_val < 0: minus_months += 1
        out.write(f"{m:>10} {a_val:>14,.0f} {b_val:>14,.0f} {mark:>10}\n")
    out.write(f"\nマイナス月: {minus_months} / {len(monthly_a)} ヶ月\n")

    # デシル分析
    out.write(f"\n--- 利益デシル分析（勝ちトレードのみ）---\n")
    wins_sorted = df[df["pnl"] > 0].sort_values("pnl", ascending=False)
    total_win_pnl = wins_sorted["pnl"].sum()
    n_wins = len(wins_sorted)
    for i in range(10):
        s = int(n_wins * i / 10); e = int(n_wins * (i+1) / 10)
        sl_pnl = wins_sorted.iloc[s:e]["pnl"].sum()
        out.write(f"  {i*10+1:>3}〜{(i+1)*10:>3}%: {sl_pnl:>14,.0f} ({sl_pnl/total_win_pnl*100:>5.1f}%)\n")

    pf_no_hr = pf(bot90["pnl"])
    verdict_1 = "✅ PASS" if pf_no_hr >= 1.5 else "❌ FAIL"
    out.write(f"\n【判定】ホームラン除外PF = {pf_no_hr:.2f} (基準: ≥1.5) → {verdict_1}\n")

    # ── 試練② ────────────────────────────────────────────────────
    print_sep("試練② USD集中リスクの定量化", out)

    h4_counts = df.groupby("h4_key")["sym"].nunique()
    out.write("--- 同一4Hバケットでの同時エントリー ---\n")
    for n_sym in range(2, 9):
        cnt = (h4_counts >= n_sym).sum()
        pct = cnt / len(h4_counts) * 100
        out.write(f"  {n_sym}銘柄以上同時: {cnt:>4}回 ({pct:>5.1f}%)\n")

    # 連敗
    out.write(f"\n--- 最大連敗分析 ---\n")
    df_time = df.sort_values("exit_time").reset_index(drop=True)
    max_streak = 0; cur_streak = 0; streak_pnl = 0; streak_start = None
    best_streaks = []
    for i, row in df_time.iterrows():
        if row["result"] == "loss":
            if cur_streak == 0: streak_start = row["entry_time"]
            cur_streak += 1; streak_pnl += row["pnl"]
            if cur_streak > max_streak: max_streak = cur_streak
        else:
            if cur_streak >= 5:
                best_streaks.append({
                    "n": cur_streak, "pnl": streak_pnl, "start": streak_start,
                    "end": df_time.iloc[i-1]["exit_time"],
                    "syms": ", ".join(df_time.iloc[max(0,i-cur_streak):i]["sym"].unique()),
                })
            cur_streak = 0; streak_pnl = 0
    best_streaks.sort(key=lambda x: x["n"], reverse=True)
    out.write(f"最大連敗数: {max_streak}\n\n")
    out.write(f"{'#':>3} {'連敗':>4} {'損失額':>14} {'銘柄':>30} {'期間'}\n")
    out.write("-" * 90 + "\n")
    for i, s in enumerate(best_streaks[:10]):
        out.write(f"{i+1:>3} {s['n']:>4} {s['pnl']:>14,.0f} {s['syms']:>30} "
                  f"{str(s['start'])[:16]}〜{str(s['end'])[:16]}\n")

    # 同日損失
    out.write(f"\n--- 同日損失 最悪10日 ---\n")
    daily_losses = df[df["result"] == "loss"].groupby("date").agg(
        cnt=("pnl", "count"), total=("pnl", "sum"),
        syms=("sym", lambda x: ", ".join(sorted(x.unique()))),
        n_sym=("sym", "nunique"),
    )
    worst_days = daily_losses.nlargest(10, "cnt")
    out.write(f"{'日付':>12} {'敗':>4} {'銘柄数':>4} {'損失額':>14} {'銘柄'}\n")
    out.write("-" * 80 + "\n")
    for dt, row in worst_days.iterrows():
        out.write(f"{str(dt)[:10]:>12} {row['cnt']:>4} {row['n_sym']:>4} {row['total']:>14,.0f} {row['syms']}\n")

    # 相関
    out.write(f"\n--- 銘柄ペア勝敗相関（日次）---\n")
    pivot = df.pivot_table(index="date", columns="sym", values="pnl", aggfunc="sum")
    pivot_bin = (pivot > 0).astype(int)
    valid = [c for c in pivot_bin.columns if pivot_bin[c].notna().sum() >= 20]
    corr = pivot_bin[valid].corr()
    out.write(corr.round(2).to_string() + "\n")
    out.write(f"\n高相関ペア（>0.3）:\n")
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            c = corr.iloc[i, j]
            if abs(c) > 0.3:
                out.write(f"  {valid[i]:>8} - {valid[j]:<8}: {c:.3f}\n")

    pct_4plus = (h4_counts >= 4).sum() / len(h4_counts) * 100
    verdict_2 = "✅ PASS" if pct_4plus < 10 else "❌ FAIL"
    out.write(f"\n【判定】4銘柄以上同時 = {pct_4plus:.1f}% (基準: <10%) → {verdict_2}\n")

    # ── 試練③ ────────────────────────────────────────────────────
    print_sep("試練③ USDCADの3.32は本物か（全期間）", out)

    cad = df[df["sym"] == "USDCAD"].copy()
    non_cad = df[df["sym"] != "USDCAD"].copy()
    cad_pf = pf(cad["pnl"])
    cad_wr = len(cad[cad["result"] == "win"]) / len(cad)
    cad_pnl = cad["pnl"].sum()

    out.write(f"USDCAD全期間: {len(cad)}トレード, PF={cad_pf:.2f}, WR={cad_wr:.1%}, 損益={cad_pnl:,.0f}\n\n")

    # 前半/後半
    cad_s = cad.sort_values("entry_time")
    mid = len(cad_s) // 2
    cad_1 = cad_s.iloc[:mid]; cad_2 = cad_s.iloc[mid:]
    out.write(f"--- 前半 vs 後半 ---\n")
    for label, data in [("前半", cad_1), ("後半", cad_2)]:
        p = pf(data["pnl"]); w = len(data[data["result"]=="win"])/len(data)
        out.write(f"  {label}: {len(data)}トレード, PF={p:.2f}, WR={w:.1%}\n")

    # 月次
    out.write(f"\n--- USDCAD月次 ---\n")
    cad_monthly = cad.groupby("month").agg(n=("pnl","count"), pnl=("pnl","sum"),
                                            wr=("result", lambda x: (x=="win").mean()))
    cad_monthly["pf"] = cad.groupby("month").apply(lambda g: pf(g["pnl"]))
    out.write(f"{'月':>10} {'n':>5} {'PF':>8} {'WR':>8} {'損益':>14}\n")
    out.write("-" * 50 + "\n")
    for m, row in cad_monthly.iterrows():
        out.write(f"{m:>10} {row['n']:>5.0f} {row['pf']:>8.2f} {row['wr']:>7.1%} {row['pnl']:>14,.0f}\n")

    best_m = cad_monthly["pnl"].idxmax()
    cad_no_best = cad[cad["month"] != best_m]
    pf_no_best = pf(cad_no_best["pnl"])
    out.write(f"\n最高月（{best_m}）除外: PF={pf_no_best:.2f}\n")

    pf_with = pf(df["pnl"]); pf_without = pf(non_cad["pnl"])
    out.write(f"\nポートフォリオ: USDCAD込みPF={pf_with:.2f}, 除外PF={pf_without:.2f}\n")
    out.write(f"USDCAD利益占有率: {cad_pnl/TOTAL_PNL:.1%}\n")

    verdict_3a = "✅ PASS" if cad_pf >= 2.0 else "❌ FAIL"
    verdict_3b = "✅ PASS" if pf_no_best >= 2.0 else "❌ FAIL"
    out.write(f"\n【判定】PF={cad_pf:.2f} (≥2.0) → {verdict_3a}  最高月除外PF={pf_no_best:.2f} (≥2.0) → {verdict_3b}\n")

    # ── 試練④ ────────────────────────────────────────────────────
    print_sep("試練④ 本当に耐えられるMDDを計算せよ（全期間）", out)

    eq = df.sort_values("exit_time")["equity"].reset_index(drop=True)
    mdd_pct, start_i, trough_i, recovery_i = calc_mdd_details(eq)
    bt_mdd = abs(mdd_pct)

    out.write(f"--- バックテストMDD ---\n")
    out.write(f"  MDD: {bt_mdd:.2f}%\n")
    out.write(f"  MDD開始→底: {trough_i - start_i} トレード\n")
    if recovery_i:
        out.write(f"  底→回復: {recovery_i - trough_i} トレード\n")
    else:
        out.write(f"  底→回復: 未回復\n")

    out.write(f"\n--- 銘柄別MDD ---\n")
    for sym in sorted(df["sym"].unique()):
        sym_eq = df[df["sym"]==sym].sort_values("exit_time")["equity"]
        sym_mdd_pct, _, _, _ = calc_mdd_details(sym_eq)
        out.write(f"  {sym:>8}: {abs(sym_mdd_pct):>6.2f}%\n")

    corrected = bt_mdd * 1.5 / 0.85
    out.write(f"\n--- 補正後MDD ---\n")
    out.write(f"  バックテスト: {bt_mdd:.2f}%\n")
    out.write(f"  スリッページ(÷0.85)+心理(×1.5): {corrected:.2f}%\n")

    out.write(f"\n--- 証拠金別 ---\n")
    for cap in [2_000_000, 5_000_000, 10_000_000]:
        bt_loss = cap * bt_mdd / 100
        cr_loss = cap * corrected / 100
        out.write(f"  ¥{cap:>10,}: BT ¥{bt_loss:>10,.0f}  補正 ¥{cr_loss:>10,.0f}\n")

    avg_win = df[df["pnl"]>0]["pnl"].mean()
    avg_loss = df[df["pnl"]<0]["pnl"].mean()
    exp = OVERALL_WR * avg_win + (1-OVERALL_WR) * avg_loss
    trades_per_month = TOTAL_TRADES / df["month"].nunique()
    cap = 10_000_000
    t_recover_bt = abs(cap * bt_mdd / 100 / exp) if exp > 0 else float("inf")
    t_recover_cr = abs(cap * corrected / 100 / exp) if exp > 0 else float("inf")

    out.write(f"\n--- 回復シミュレーション（証拠金1000万円）---\n")
    out.write(f"  期待値/トレード: {exp:,.0f}\n")
    out.write(f"  月間トレード: {trades_per_month:.0f}\n")
    out.write(f"  BT MDD回復: {t_recover_bt:.0f}トレード = {t_recover_bt/trades_per_month:.1f}ヶ月\n")
    out.write(f"  補正MDD回復: {t_recover_cr:.0f}トレード = {t_recover_cr/trades_per_month:.1f}ヶ月\n")

    daily_pnl = df.groupby("date")["pnl"].sum()
    out.write(f"\n--- 日次損益 ---\n")
    out.write(f"  平均: {daily_pnl.mean():>12,.0f}  σ: {daily_pnl.std():>12,.0f}\n")
    out.write(f"  最大: {daily_pnl.max():>12,.0f}  最小: {daily_pnl.min():>12,.0f}\n")
    out.write(f"  マイナス日: {(daily_pnl<0).sum()}/{len(daily_pnl)} ({(daily_pnl<0).mean():.1%})\n")

    verdict_4 = "✅ PASS" if corrected < 40 and t_recover_cr/trades_per_month < 3 else "❌ FAIL"
    out.write(f"\n【判定】補正MDD={corrected:.1f}% (<40%), 回復={t_recover_cr/trades_per_month:.1f}ヶ月 (<3) → {verdict_4}\n")

    # ── 試練⑤ ────────────────────────────────────────────────────
    print_sep("試練⑤ フィルターアブレーション（全期間）", out)

    # v80 ablation
    v80_patterns = [
        ("A: 全フィルター",         True,  True,  True),
        ("B: KMID除外",            False, True,  True),
        ("C: KLOW除外",            True,  False, True),
        ("D: Body除外",            True,  True,  False),
        ("E: KMID+KLOW除外",       False, False, True),
        ("H: 全除外（4HEMA20のみ）",  False, False, False),
    ]

    v80_syms = [t for t in APPROVED if t["logic"] == "V80"]

    out.write("■ v80ロジック — フィルター個別除外（全期間）\n")
    out.write(f"{'銘柄':>8} {'パターン':>22} {'KMID':>5} {'KLOW':>5} {'Body':>5} | {'n':>5} {'WR':>6} {'PF':>6} {'損益':>14}\n")
    out.write("-" * 85 + "\n")

    v80_results = {}
    for tgt in v80_syms:
        sym = tgt["sym"]; tol = tgt["tol"]; rr = tgt["rr"]
        d1m, d4h = data_cache[sym]
        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        m1c = {"idx": d1m.index, "opens": d1m["open"].values,
               "closes": d1m["close"].values,
               "highs": d1m["high"].values, "lows": d1m["low"].values}

        for pname, uk, ul, ub in v80_patterns:
            sigs = generate_signals_ablation_v80(d1m, d4h, spread, m1c, rr=rr, tol=tol,
                                                  use_kmid=uk, use_klow=ul, use_body=ub)
            st = simulate_simple(sigs, d1m, sym)
            v80_results[(sym, pname)] = st
            km = "✓" if uk else "✗"; kl = "✓" if ul else "✗"; bd = "✓" if ub else "✗"
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            out.write(f"{sym:>8} {pname:>22} {km:>5} {kl:>5} {bd:>5} | "
                      f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['pnl']:>14,.0f}\n")
        out.write("\n")

    # v80 summary
    out.write("■ v80 寄与度サマリー\n")
    out.write(f"{'銘柄':>8} {'KMID':>10} {'KLOW':>10} {'Body':>10}\n")
    out.write("-" * 42 + "\n")
    all_kmid = []; all_klow = []; all_body = []
    for tgt in v80_syms:
        sym = tgt["sym"]
        base = v80_results.get((sym, "A: 全フィルター"), {})
        no_k = v80_results.get((sym, "B: KMID除外"), {})
        no_l = v80_results.get((sym, "C: KLOW除外"), {})
        no_b = v80_results.get((sym, "D: Body除外"), {})
        if all(x.get("n",0)>0 for x in [base,no_k,no_l,no_b]):
            dk = base["pf"]-no_k["pf"]; dl = base["pf"]-no_l["pf"]; db = base["pf"]-no_b["pf"]
            all_kmid.append(dk); all_klow.append(dl); all_body.append(db)
            out.write(f"{sym:>8} {dk:>+10.2f} {dl:>+10.2f} {db:>+10.2f}\n")
    if all_kmid:
        out.write(f"{'平均':>8} {np.mean(all_kmid):>+10.2f} {np.mean(all_klow):>+10.2f} {np.mean(all_body):>+10.2f}\n")

    # Logic-A ablation
    a_patterns = [
        ("A: 全フィルター",          True,  True,  True,  True),
        ("B: KMID除外",             False, True,  True,  True),
        ("C: KLOW除外",             True,  False, True,  True),
        ("D: 日足EMA除外",          True,  True,  False, True),
        ("E: EMA距離除外",           True,  True,  True,  False),
        ("H: 全除外（4HEMA20のみ）",  False, False, False, False),
    ]
    a_syms = [t for t in APPROVED if t["logic"] == "A"]

    out.write("\n■ Logic-A — フィルター個別除外（全期間）\n")
    out.write(f"{'銘柄':>8} {'パターン':>26} {'KMID':>5} {'KLOW':>5} {'1dEMA':>5} {'dist':>5} | {'n':>5} {'WR':>6} {'PF':>6} {'損益':>14}\n")
    out.write("-" * 95 + "\n")

    a_results = {}
    for tgt in a_syms:
        sym = tgt["sym"]; tol = tgt["tol"]; rr = tgt["rr"]
        d1m, d4h = data_cache[sym]
        cfg = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        m1c = {"idx": d1m.index, "opens": d1m["open"].values,
               "closes": d1m["close"].values,
               "highs": d1m["high"].values, "lows": d1m["low"].values}
        atr_d = calc_atr(d1m, 10).to_dict()

        for pname, uk, ul, ud, ue in a_patterns:
            sigs = generate_signals_ablation_a(d1m, d4h, spread, atr_d, m1c, rr=rr, tol=tol,
                                                use_kmid=uk, use_klow=ul, use_1d_trend=ud, use_ema_dist=ue)
            st = simulate_simple(sigs, d1m, sym)
            a_results[(sym, pname)] = st
            km="✓" if uk else "✗"; kl="✓" if ul else "✗"; d1="✓" if ud else "✗"; ed="✓" if ue else "✗"
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            out.write(f"{sym:>8} {pname:>26} {km:>5} {kl:>5} {d1:>5} {ed:>5} | "
                      f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['pnl']:>14,.0f}\n")
        out.write("\n")

    # Logic-A summary
    out.write("■ Logic-A 寄与度サマリー\n")
    out.write(f"{'銘柄':>8} {'KMID':>10} {'KLOW':>10} {'日足EMA':>10} {'EMA距離':>10}\n")
    out.write("-" * 52 + "\n")
    a_kmid = []; a_klow = []
    for tgt in a_syms:
        sym = tgt["sym"]
        base = a_results.get((sym, "A: 全フィルター"), {})
        no_k = a_results.get((sym, "B: KMID除外"), {})
        no_l = a_results.get((sym, "C: KLOW除外"), {})
        no_d = a_results.get((sym, "D: 日足EMA除外"), {})
        no_e = a_results.get((sym, "E: EMA距離除外"), {})
        if all(x.get("n",0)>0 for x in [base,no_k,no_l,no_d,no_e]):
            dk=base["pf"]-no_k["pf"]; dl=base["pf"]-no_l["pf"]
            dd=base["pf"]-no_d["pf"]; de=base["pf"]-no_e["pf"]
            a_kmid.append(dk); a_klow.append(dl)
            out.write(f"{sym:>8} {dk:>+10.2f} {dl:>+10.2f} {dd:>+10.2f} {de:>+10.2f}\n")

    # 総合判定
    total_kmid = all_kmid + a_kmid
    total_klow = all_klow + a_klow
    avg_km = np.mean(total_kmid) if total_kmid else 0
    avg_kl = np.mean(total_klow) if total_klow else 0
    vk = "✅ PASS" if avg_km >= 0.3 else "❌ FAIL"
    vl = "✅ PASS" if avg_kl >= 0.3 else "❌ FAIL"
    out.write(f"\n  KMID平均寄与: {avg_km:+.2f} (≥+0.3) → {vk}\n")
    out.write(f"  KLOW平均寄与: {avg_kl:+.2f} (≥+0.3) → {vl}\n")

    # ── 総合判定 ──────────────────────────────────────────────────
    print_sep("5試練 総合判定（全期間）", out)

    verdicts = {
        "① ホームラン依存": verdict_1,
        "② USD集中リスク": verdict_2,
        "③ USDCAD真贋": verdict_3a,
        "④ MDD耐性": verdict_4,
        "⑤ KMID寄与": vk,
        "⑤ KLOW寄与": vl,
    }
    out.write(f"{'試練':>18} {'結果':>10}\n")
    out.write("-" * 32 + "\n")
    for name, v in verdicts.items():
        out.write(f"{name:>18} {v:>10}\n")

    pass_count = sum(1 for v in verdicts.values() if "PASS" in v)
    out.write(f"\n合格: {pass_count}/6\n")
    failed = [k for k, v in verdicts.items() if "FAIL" in v]
    if failed:
        out.write(f"不合格: {', '.join(failed)}\n")

    out.close()
    print(f"\nレポート出力先: {OUT_PATH}")
    return OUT_PATH

# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    df, data_cache = generate_full_period_trades()
    report_path = run_trials(df, data_cache)
    elapsed = time.time() - t0
    print(f"\n全処理完了: {elapsed:.0f}秒")

    # print report
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())

if __name__ == "__main__":
    main()
