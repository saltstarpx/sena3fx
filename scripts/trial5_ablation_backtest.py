#!/usr/bin/env python3
"""
試練⑤ フィルターアブレーション分析
====================================
KMID/KLOW/日足EMA/EMA距離/4Hボディ比率の各フィルターを1つずつ外して
PF・勝率・トレード数への寄与度を定量化する。

パターン:
  A: 全フィルター（現行v80: KMID+KLOW+Body）
  B: KMID除外
  C: KLOW除外
  D: Body除外
  E: KMID+KLOW両方除外
  F: Logic-A全フィルター（日足EMA+KMID+KLOW+EMA距離）
  G: Logic-A - 日足EMA除外
  H: Logic-A - EMA距離除外
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数（backtest_final_optimized.py と同一）──────────────────────
INIT_CASH     = 1_000_000
RR_RATIO      = 3.0
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000
KLOW_THR      = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E0_WINDOW_MIN   = 2
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
BODY_RATIO_MIN  = 0.3

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")

# ── 銘柄リスト（v80 + Logic-A の両方テスト）──────────────────────
SYMBOLS_V80 = [
    {"sym": "EURUSD",  "tol": 0.30},
    {"sym": "GBPUSD",  "tol": 0.30},
    {"sym": "AUDUSD",  "tol": 0.30},
    {"sym": "USDCAD",  "tol": 0.30},
    {"sym": "USDCHF",  "tol": 0.30},
]

SYMBOLS_LOGIC_A = [
    {"sym": "XAUUSD",  "tol": 0.20},
    {"sym": "NZDUSD",  "tol": 0.20},
    {"sym": "GBPUSD",  "tol": 0.30},
    {"sym": "USDCAD",  "tol": 0.30},
]

# ── データロード（backtest_final_optimized.py と同一）──────────────
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

def split_is_oos(d1m):
    n = int(len(d1m) * 0.4)
    ts = d1m.index[n]
    return d1m[d1m.index < ts].copy(), d1m[d1m.index >= ts].copy(), ts

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

# ── アブレーション対応シグナル生成（v80ベース）──────────────────────
def generate_signals_ablation_v80(d1m, d4h_full, spread, m1c,
                                   rr=RR_RATIO, tol=A3_DEFAULT_TOL,
                                   use_kmid=True, use_klow=True, use_body=True):
    """v80ベースでKMID/KLOW/Bodyを個別にON/OFF可能"""
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

# ── アブレーション対応シグナル生成（Logic-Aベース）──────────────────
def generate_signals_ablation_a(d1m, d4h_full, spread, atr_d, m1c,
                                 rr=RR_RATIO, tol=A3_DEFAULT_TOL,
                                 use_kmid=True, use_klow=True,
                                 use_1d_trend=True, use_ema_dist=True):
    """Logic-Aベースで各フィルターを個別にON/OFF可能"""
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

        # 日足EMA方向一致
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

# ── シミュレーション ──────────────────────────────────────────────
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

def simulate(signals, d1m, sym):
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        rm.risk_pct = 0.02
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
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
        trades.append({"result": result, "pnl": total,
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd

def calc_stats(trades):
    if len(trades) < 5: return {"n": 0, "wr": 0, "pf": 0, "pnl": 0}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr = len(wins) / n
    gw = wins.sum(); gl = abs(loss.sum())
    pf_val = gw / gl if gl > 0 else float("inf")
    return {"n": n, "wr": wr, "pf": pf_val, "pnl": df["pnl"].sum()}

# ── メイン ───────────────────────────────────────────────────────
def main():
    out_path = os.path.join(OUT_DIR, "trial5_ablation_report.txt")
    f = open(out_path, "w", encoding="utf-8")

    def write(s=""):
        print(s)
        f.write(s + "\n")

    write("=" * 90)
    write("  試練⑤ フィルターアブレーション分析")
    write("=" * 90)

    # ── Part 1: v80ベース（KMID / KLOW / Body の寄与）──────────────
    write("\n■ Part 1: v80ロジック — フィルター個別除外（OOS期間）")
    write(f"{'銘柄':>8} {'パターン':>22} {'KMID':>5} {'KLOW':>5} {'Body':>5} | {'n':>5} {'WR':>6} {'PF':>6} {'損益':>14}")
    write("-" * 85)

    v80_patterns = [
        ("A: 全フィルター",        True,  True,  True),
        ("B: KMID除外",           False, True,  True),
        ("C: KLOW除外",           True,  False, True),
        ("D: Body除外",           True,  True,  False),
        ("E: KMID+KLOW除外",      False, False, True),
        ("F: KMID+Body除外",      False, True,  False),
        ("G: KLOW+Body除外",      True,  False, False),
        ("H: 全除外（4HEMA20のみ）", False, False, False),
    ]

    v80_results = {}
    data_cache = {}

    for tgt in SYMBOLS_V80:
        sym = tgt["sym"]
        tol = tgt["tol"]

        if sym not in data_cache:
            d1m_full, d4h_full = load_all(sym)
            if d1m_full is None:
                write(f"  {sym}: データなし")
                continue
            _, oos_d, _ = split_is_oos(d1m_full)
            cfg = SYMBOL_CONFIG[sym]
            spread = cfg["spread"] * cfg["pip"]
            m1c = {"idx": oos_d.index, "opens": oos_d["open"].values,
                   "closes": oos_d["close"].values,
                   "highs": oos_d["high"].values, "lows": oos_d["low"].values}
            data_cache[sym] = (oos_d, d4h_full, spread, m1c)

        oos_d, d4h_full, spread, m1c = data_cache[sym]

        for pname, use_kmid, use_klow, use_body in v80_patterns:
            sigs = generate_signals_ablation_v80(
                oos_d, d4h_full, spread, m1c,
                rr=RR_RATIO, tol=tol,
                use_kmid=use_kmid, use_klow=use_klow, use_body=use_body
            )
            trades, _, _ = simulate(sigs, oos_d, sym)
            st = calc_stats(trades)
            k = (sym, pname)
            v80_results[k] = st

            km = "✓" if use_kmid else "✗"
            kl = "✓" if use_klow else "✗"
            bd = "✓" if use_body else "✗"
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            write(f"{sym:>8} {pname:>22} {km:>5} {kl:>5} {bd:>5} | "
                  f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['pnl']:>14,.0f}")

        write("")  # blank line between symbols

    # ── v80 寄与度サマリー ──────────────────────────────────────────
    write("\n■ v80 フィルター寄与度サマリー（全フィルターPF - 除外PF）")
    write(f"{'銘柄':>8} {'KMID寄与':>10} {'KLOW寄与':>10} {'Body寄与':>10}")
    write("-" * 42)

    kmid_contribs = []
    klow_contribs = []
    body_contribs = []

    for tgt in SYMBOLS_V80:
        sym = tgt["sym"]
        base = v80_results.get((sym, "A: 全フィルター"), {})
        no_kmid = v80_results.get((sym, "B: KMID除外"), {})
        no_klow = v80_results.get((sym, "C: KLOW除外"), {})
        no_body = v80_results.get((sym, "D: Body除外"), {})

        if base and no_kmid and no_klow and no_body:
            kmid_d = base["pf"] - no_kmid["pf"]
            klow_d = base["pf"] - no_klow["pf"]
            body_d = base["pf"] - no_body["pf"]
            kmid_contribs.append(kmid_d)
            klow_contribs.append(klow_d)
            body_contribs.append(body_d)
            write(f"{sym:>8} {kmid_d:>+10.2f} {klow_d:>+10.2f} {body_d:>+10.2f}")

    if kmid_contribs:
        write(f"{'平均':>8} {np.mean(kmid_contribs):>+10.2f} {np.mean(klow_contribs):>+10.2f} {np.mean(body_contribs):>+10.2f}")

    # ── Part 2: Logic-Aベース（日足EMA / EMA距離 / KMID / KLOW）───
    write("\n\n■ Part 2: Logic-Aロジック — フィルター個別除外（OOS期間）")
    write(f"{'銘柄':>8} {'パターン':>26} {'KMID':>5} {'KLOW':>5} {'1dEMA':>5} {'dist':>5} | {'n':>5} {'WR':>6} {'PF':>6} {'損益':>14}")
    write("-" * 95)

    a_patterns = [
        ("A: 全フィルター",          True,  True,  True,  True),
        ("B: KMID除外",             False, True,  True,  True),
        ("C: KLOW除外",             True,  False, True,  True),
        ("D: 日足EMA除外",          True,  True,  False, True),
        ("E: EMA距離除外",           True,  True,  True,  False),
        ("F: KMID+KLOW除外",        False, False, True,  True),
        ("G: 日足EMA+EMA距離除外",    True,  True,  False, False),
        ("H: 全除外（4HEMA20のみ）",  False, False, False, False),
    ]

    a_results = {}

    for tgt in SYMBOLS_LOGIC_A:
        sym = tgt["sym"]
        tol = tgt["tol"]

        if sym not in data_cache:
            d1m_full, d4h_full = load_all(sym)
            if d1m_full is None:
                write(f"  {sym}: データなし")
                continue
            _, oos_d, _ = split_is_oos(d1m_full)
            cfg = SYMBOL_CONFIG[sym]
            spread = cfg["spread"] * cfg["pip"]
            m1c = {"idx": oos_d.index, "opens": oos_d["open"].values,
                   "closes": oos_d["close"].values,
                   "highs": oos_d["high"].values, "lows": oos_d["low"].values}
            data_cache[sym] = (oos_d, d4h_full, spread, m1c)

        oos_d, d4h_full, spread, m1c = data_cache[sym]
        atr_d = calc_atr(oos_d, 10).to_dict()

        for pname, use_kmid, use_klow, use_1d, use_dist in a_patterns:
            sigs = generate_signals_ablation_a(
                oos_d, d4h_full, spread, atr_d, m1c,
                rr=RR_RATIO, tol=tol,
                use_kmid=use_kmid, use_klow=use_klow,
                use_1d_trend=use_1d, use_ema_dist=use_dist
            )
            trades, _, _ = simulate(sigs, oos_d, sym)
            st = calc_stats(trades)
            k = (sym, pname)
            a_results[k] = st

            km = "✓" if use_kmid else "✗"
            kl = "✓" if use_klow else "✗"
            d1 = "✓" if use_1d else "✗"
            ed = "✓" if use_dist else "✗"
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            write(f"{sym:>8} {pname:>26} {km:>5} {kl:>5} {d1:>5} {ed:>5} | "
                  f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['pnl']:>14,.0f}")

        write("")

    # ── Logic-A 寄与度サマリー ──────────────────────────────────────
    write("\n■ Logic-A フィルター寄与度サマリー（全フィルターPF - 除外PF）")
    write(f"{'銘柄':>8} {'KMID寄与':>10} {'KLOW寄与':>10} {'日足EMA':>10} {'EMA距離':>10}")
    write("-" * 52)

    for tgt in SYMBOLS_LOGIC_A:
        sym = tgt["sym"]
        base = a_results.get((sym, "A: 全フィルター"), {})
        no_km = a_results.get((sym, "B: KMID除外"), {})
        no_kl = a_results.get((sym, "C: KLOW除外"), {})
        no_d1 = a_results.get((sym, "D: 日足EMA除外"), {})
        no_ed = a_results.get((sym, "E: EMA距離除外"), {})

        if all(x.get("n", 0) > 0 for x in [base, no_km, no_kl, no_d1, no_ed]):
            write(f"{sym:>8} {base['pf']-no_km['pf']:>+10.2f} {base['pf']-no_kl['pf']:>+10.2f} "
                  f"{base['pf']-no_d1['pf']:>+10.2f} {base['pf']-no_ed['pf']:>+10.2f}")

    # ── 判定 ──────────────────────────────────────────────────────
    write("\n" + "=" * 90)
    write("  試練⑤ 判定")
    write("=" * 90)

    # KMID average contribution across all tests
    all_kmid = []
    for tgt in SYMBOLS_V80:
        sym = tgt["sym"]
        base = v80_results.get((sym, "A: 全フィルター"), {})
        no_km = v80_results.get((sym, "B: KMID除外"), {})
        if base.get("n", 0) > 0 and no_km.get("n", 0) > 0:
            all_kmid.append(base["pf"] - no_km["pf"])
    for tgt in SYMBOLS_LOGIC_A:
        sym = tgt["sym"]
        base = a_results.get((sym, "A: 全フィルター"), {})
        no_km = a_results.get((sym, "B: KMID除外"), {})
        if base.get("n", 0) > 0 and no_km.get("n", 0) > 0:
            all_kmid.append(base["pf"] - no_km["pf"])

    all_klow = []
    for tgt in SYMBOLS_V80:
        sym = tgt["sym"]
        base = v80_results.get((sym, "A: 全フィルター"), {})
        no_kl = v80_results.get((sym, "C: KLOW除外"), {})
        if base.get("n", 0) > 0 and no_kl.get("n", 0) > 0:
            all_klow.append(base["pf"] - no_kl["pf"])
    for tgt in SYMBOLS_LOGIC_A:
        sym = tgt["sym"]
        base = a_results.get((sym, "A: 全フィルター"), {})
        no_kl = a_results.get((sym, "C: KLOW除外"), {})
        if base.get("n", 0) > 0 and no_kl.get("n", 0) > 0:
            all_klow.append(base["pf"] - no_kl["pf"])

    avg_kmid = np.mean(all_kmid) if all_kmid else 0
    avg_klow = np.mean(all_klow) if all_klow else 0

    verdict_kmid = "✅ PASS" if avg_kmid >= 0.3 else "❌ FAIL"
    verdict_klow = "✅ PASS" if avg_klow >= 0.3 else "❌ FAIL"

    write(f"  KMID平均寄与: {avg_kmid:+.2f} (基準: ≥+0.3) → {verdict_kmid}")
    write(f"  KLOW平均寄与: {avg_klow:+.2f} (基準: ≥+0.3) → {verdict_klow}")

    if avg_kmid >= 0.3 and avg_klow >= 0.3:
        write("\n  → KMID・KLOWともにPFに+0.3以上寄与。フィルターは有効。")
    elif avg_kmid >= 0.3:
        write(f"\n  → KMIDは有効（+{avg_kmid:.2f}）。KLOWは寄与不十分（{avg_klow:+.2f}）→ 削除検討。")
    elif avg_klow >= 0.3:
        write(f"\n  → KLOWは有効（+{avg_klow:.2f}）。KMIDは寄与不十分（{avg_kmid:+.2f}）→ 削除検討。")
    else:
        write(f"\n  → KMID（{avg_kmid:+.2f}）もKLOW（{avg_klow:+.2f}）も寄与不十分。フィルター構成を再検討。")

    f.close()
    print(f"\nレポート出力先: {out_path}")

if __name__ == "__main__":
    main()
