#!/usr/bin/env python3
"""
KLOW除去 全期間バックテスト比較
================================
現行（KLOW有り）vs KLOW除去版 を全8銘柄で全期間バックテスト。
銘柄別 + ポートフォリオレベルでPF/トレード数/MDD/月次を比較。
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数（trial_5_challenges_fullperiod.py と同一）──────────────
INIT_CASH     = 1_000_000
RR_RATIO      = 2.5
RR_RATIO_V80  = 3.0
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000
EQUITY_CAP    = 100_000_000

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

# ── シグナル生成（use_klow パラメータ付き）───────────────────────
def generate_signals_v80(d1m, d4h_full, spread, m1c, rr=RR_RATIO_V80, tol=A3_DEFAULT_TOL, use_klow=True):
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
        if use_klow and not chk_klow(h4l): continue
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

def generate_signals_a(d1m, d4h_full, spread, atr_d, m1c, rr=RR_RATIO, tol=A3_DEFAULT_TOL, use_klow=True):
    d4h, d1d = build_4h(d4h_full, need_1d=True)
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
        if d1d is None: continue
        d1b = d1d[d1d.index.normalize() < hct.normalize()]
        if not len(d1b) or d1b.iloc[-1]["trend1d"] != trend: continue
        if not chk_kmid(h4l, trend): continue
        if use_klow and not chk_klow(h4l): continue
        if not chk_ema(h4l): continue
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

def generate_signals_c(d1m, d4h_full, spread, m1c, rr=RR_RATIO_V80, tol=A3_DEFAULT_TOL, use_klow=True):
    d4h, _ = build_4h(d4h_full, need_1d=False)
    d1h = build_1h(d1m)
    signals = []; used = set()
    for i in range(2, len(d1h)):
        hct = d1h.index[i]
        p1  = d1h.iloc[i-1]; p2 = d1h.iloc[i-2]
        atr1h = d1h.iloc[i]["atr"]
        if pd.isna(atr1h) or atr1h <= 0: continue
        h4b = d4h[d4h.index < hct]
        if len(h4b) < STREAK_MIN: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l.get("atr", np.nan)): continue
        trend = h4l["trend"]; h4atr = h4l["atr"]
        if not chk_kmid(h4l, trend): continue
        if use_klow and not chk_klow(h4l): continue
        d = trend
        v1 = p2["low"]  if d == 1 else p2["high"]
        v2 = p1["low"]  if d == 1 else p1["high"]
        if abs(v1 - v2) > atr1h * tol: continue
        if d == 1 and p1["close"] <= p1["open"]: continue
        if d == -1 and p1["close"] >= p1["open"]: continue
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

# ── シミュレーション ──────────────────────────────────────────────
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

def simulate(signals, d1m, sym):
    if not signals: return [], INIT_CASH
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []
    for sig in signals:
        rm.risk_pct = 0.02
        sizing_eq = min(equity, EQUITY_CAP)
        lot = rm.calc_lot(sizing_eq, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
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
            equity += half_pnl; rem = lot * 0.5
        else:
            rem = lot
        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total = half_pnl + pnl
        trades.append({
            "sym": sym, "result": result, "pnl": total, "equity": equity,
            "entry_time": sig["time"], "exit_time": exit_time,
            "month": sig["time"].strftime("%Y-%m"),
        })
    return trades, equity

def pf(pnl):
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    return gw / gl if gl > 0 else float("inf")

def calc_mdd(equity_series):
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak
    return abs(dd.min()) * 100

# ═══════════════════════════════════════════════════════════════════
#  メイン
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 90)
    print("  KLOW除去 全期間バックテスト比較")
    print("  現行（KLOW有り） vs KLOW除去版")
    print("=" * 90)

    data_cache = {}
    results = []  # (sym, variant, trades_list)

    for tgt in APPROVED:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        rr    = tgt["rr"]
        tol   = tgt["tol"]

        if sym not in data_cache:
            d1m, d4h = load_all(sym)
            if d1m is None:
                print(f"  {sym}: データなし"); continue
            data_cache[sym] = (d1m, d4h)

        d1m, d4h = data_cache[sym]
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs": d1m["high"].values, "lows": d1m["low"].values}

        for use_klow, variant_name in [(True, "現行(KLOW有)"), (False, "KLOW除去")]:
            t0 = time.time()
            if logic == "V80":
                sigs = generate_signals_v80(d1m, d4h, spread, m1c, rr=rr, tol=tol, use_klow=use_klow)
            elif logic == "A":
                atr_d = calc_atr(d1m, 10).to_dict()
                sigs = generate_signals_a(d1m, d4h, spread, atr_d, m1c, rr=rr, tol=tol, use_klow=use_klow)
            elif logic == "C":
                sigs = generate_signals_c(d1m, d4h, spread, m1c, rr=rr, tol=tol, use_klow=use_klow)
            else:
                continue

            trades, final_eq = simulate(sigs, d1m, sym)
            elapsed = time.time() - t0
            results.append({"sym": sym, "logic": logic, "variant": variant_name,
                            "trades": trades, "final_eq": final_eq})
            pnl_s = pd.Series([t["pnl"] for t in trades])
            n = len(trades)
            wr = len([t for t in trades if t["result"] == "win"]) / n if n > 0 else 0
            pf_val = pf(pnl_s) if n > 0 else 0
            print(f"  {sym:>8} {variant_name:<12} {n:>5}件  "
                  f"PF={pf_val:.2f}  WR={wr:.1%}  ({elapsed:.0f}s)")

    # ── 銘柄別比較表 ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  銘柄別比較表")
    print("=" * 90)

    print(f"\n{'銘柄':>8} {'ロジック':>6}  "
          f"{'── 現行(KLOW有) ──':>30}  {'── KLOW除去 ──':>30}  {'差分':>10}")
    print(f"{'':>16}  {'n':>5} {'PF':>6} {'WR':>7} {'PnL':>14}  "
          f"{'n':>5} {'PF':>6} {'WR':>7} {'PnL':>14}  {'ΔPF':>6}")
    print("-" * 110)

    all_trades_current = []
    all_trades_no_klow = []

    for tgt in APPROVED:
        sym = tgt["sym"]
        r_cur = [r for r in results if r["sym"] == sym and r["variant"] == "現行(KLOW有)"]
        r_nok = [r for r in results if r["sym"] == sym and r["variant"] == "KLOW除去"]
        if not r_cur or not r_nok: continue

        t_cur = r_cur[0]["trades"]; t_nok = r_nok[0]["trades"]
        all_trades_current.extend(t_cur)
        all_trades_no_klow.extend(t_nok)

        pnl_cur = pd.Series([t["pnl"] for t in t_cur])
        pnl_nok = pd.Series([t["pnl"] for t in t_nok])

        n_cur = len(t_cur); n_nok = len(t_nok)
        pf_cur = pf(pnl_cur) if n_cur > 0 else 0
        pf_nok = pf(pnl_nok) if n_nok > 0 else 0
        wr_cur = len([t for t in t_cur if t["result"]=="win"]) / n_cur if n_cur > 0 else 0
        wr_nok = len([t for t in t_nok if t["result"]=="win"]) / n_nok if n_nok > 0 else 0
        pnl_sum_cur = pnl_cur.sum() if n_cur > 0 else 0
        pnl_sum_nok = pnl_nok.sum() if n_nok > 0 else 0
        delta_pf = pf_nok - pf_cur

        print(f"{sym:>8} {tgt['logic']:>6}  "
              f"{n_cur:>5} {pf_cur:>6.2f} {wr_cur:>6.1%} {pnl_sum_cur:>14,.0f}  "
              f"{n_nok:>5} {pf_nok:>6.2f} {wr_nok:>6.1%} {pnl_sum_nok:>14,.0f}  "
              f"{delta_pf:>+6.2f}")

    # ── ポートフォリオレベル比較 ──────────────────────────────────
    print("\n" + "=" * 90)
    print("  ポートフォリオレベル比較")
    print("=" * 90)

    for label, trades_list in [("現行(KLOW有)", all_trades_current),
                                ("KLOW除去", all_trades_no_klow)]:
        df = pd.DataFrame(trades_list)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"]  = pd.to_datetime(df["exit_time"])

        n = len(df)
        pnl_total = df["pnl"].sum()
        pf_val = pf(df["pnl"])
        wr = len(df[df["result"]=="win"]) / n

        portfolio_eq = INIT_CASH + df.sort_values("exit_time")["pnl"].cumsum()
        mdd_val = calc_mdd(portfolio_eq)

        monthly = df.groupby("month")["pnl"].sum()
        months_minus = (monthly < 0).sum()
        total_months = len(monthly)

        print(f"\n  {label}:")
        print(f"    トレード数:  {n:>6}")
        print(f"    PF:          {pf_val:>6.2f}")
        print(f"    WR:          {wr:>6.1%}")
        print(f"    総PnL:       {pnl_total:>14,.0f}")
        print(f"    MDD:         {mdd_val:>6.1f}%")
        print(f"    マイナス月:  {months_minus}/{total_months}")

    # ── 月次比較 ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  月次PnL比較")
    print("=" * 90)

    df_cur = pd.DataFrame(all_trades_current)
    df_nok = pd.DataFrame(all_trades_no_klow)

    monthly_cur = df_cur.groupby("month")["pnl"].sum()
    monthly_nok = df_nok.groupby("month")["pnl"].sum()

    all_months = sorted(set(monthly_cur.index) | set(monthly_nok.index))
    print(f"{'月':>8} {'現行':>16} {'KLOW除去':>16} {'差分':>16} {'改善?':>6}")
    print("-" * 70)
    for m in all_months:
        c = monthly_cur.get(m, 0)
        n = monthly_nok.get(m, 0)
        d = n - c
        mark = "✓" if d > 0 else ""
        print(f"{m:>8} {c:>16,.0f} {n:>16,.0f} {d:>+16,.0f} {mark:>6}")

    print("-" * 70)
    print(f"{'合計':>8} {monthly_cur.sum():>16,.0f} {monthly_nok.sum():>16,.0f} "
          f"{monthly_nok.sum()-monthly_cur.sum():>+16,.0f}")

    # ── 追加トレードの質 ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  KLOW除去で追加されるトレードの質")
    print("=" * 90)

    n_added = len(all_trades_no_klow) - len(all_trades_current)
    print(f"  追加トレード数: {n_added}")

    # 追加トレードを特定（KLOW除去にはあるが現行にはないentry_time）
    cur_times = set(t["entry_time"] for t in all_trades_current)
    added = [t for t in all_trades_no_klow if t["entry_time"] not in cur_times]
    if added:
        added_df = pd.DataFrame(added)
        added_wins = len(added_df[added_df["result"]=="win"])
        added_wr = added_wins / len(added_df)
        added_pf = pf(added_df["pnl"])
        added_pnl = added_df["pnl"].sum()
        print(f"  追加トレード: {len(added_df)}件")
        print(f"  勝率: {added_wr:.1%}")
        print(f"  PF:   {added_pf:.2f}")
        print(f"  PnL:  {added_pnl:>+14,.0f}")

        # 銘柄別内訳
        print(f"\n  銘柄別内訳:")
        for sym, grp in added_df.groupby("sym"):
            sw = len(grp[grp["result"]=="win"])
            print(f"    {sym:>8}: {len(grp):>4}件  勝率={sw/len(grp):.1%}  "
                  f"PnL={grp['pnl'].sum():>+14,.0f}")

    # ── 結論 ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  結論")
    print("=" * 90)

    pf_cur_total = pf(df_cur["pnl"])
    pf_nok_total = pf(df_nok["pnl"])
    n_cur_total = len(df_cur)
    n_nok_total = len(df_nok)
    mdd_cur = calc_mdd(INIT_CASH + df_cur.sort_values("exit_time")["pnl"].cumsum())
    mdd_nok = calc_mdd(INIT_CASH + df_nok.sort_values("exit_time")["pnl"].cumsum())

    print(f"\n  現行:     PF={pf_cur_total:.2f}  n={n_cur_total}  MDD={mdd_cur:.1f}%")
    print(f"  KLOW除去: PF={pf_nok_total:.2f}  n={n_nok_total}  MDD={mdd_nok:.1f}%")
    print(f"  差分:     ΔPF={pf_nok_total-pf_cur_total:+.2f}  "
          f"Δn={n_nok_total-n_cur_total:+d}  ΔMDD={mdd_nok-mdd_cur:+.1f}pp")

    if pf_nok_total < pf_cur_total:
        print(f"\n  ⚠ KLOW除去でPF低下。削除は推奨しない。")
    elif pf_nok_total >= pf_cur_total and mdd_nok <= mdd_cur * 1.1:
        print(f"\n  ✅ KLOW除去でPF維持/改善かつMDD悪化なし。削除可。")
    else:
        print(f"\n  △ KLOW除去でPF改善だがMDD悪化。慎重に判断。")

if __name__ == "__main__":
    main()
