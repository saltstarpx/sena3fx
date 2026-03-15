"""
backtest_rr_comparison.py
==========================
RR倍率 × 半利確有無 バックテスト比較

【検証バリアント】
  1. 現行:     半利確あり(1R→50%) + TP=2.5R
  2. 全利 2.0R: 半利確なし + TP=2.0R
  3. 全利 2.5R: 半利確なし + TP=2.5R
  4. 全利 3.0R: 半利確なし + TP=3.0R
  5. 半利+2.0R: 半利確あり(1R→50%) + TP=2.0R
  6. 半利+3.0R: 半利確あり(1R→50%) + TP=3.0R

目的: 半利確の価値 + 最適RR倍率を定量評価
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
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

TARGETS = [
    {"sym": "GBPUSD",  "logic": "A", "tol": 0.30},
    {"sym": "EURUSD",  "logic": "C", "tol": 0.30},
    {"sym": "USDCAD",  "logic": "A", "tol": 0.30},
    {"sym": "NZDUSD",  "logic": "A", "tol": 0.20},
    {"sym": "XAUUSD",  "logic": "A", "tol": 0.20},
    {"sym": "AUDUSD",  "logic": "B", "tol": 0.30},
    {"sym": "USDJPY",  "logic": "C", "tol": 0.30},
]

# ── データロード（backtest_final_optimized.pyと同一） ─────────────
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
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, tol=0.30, rr=2.5):
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

# ══════════════════════════════════════════════════════════════════
# EXIT関数
# ══════════════════════════════════════════════════════════════════

def _exit_with_half(highs, lows, ep, sl, tp, risk, d):
    """半利確あり: 1R→50%決済(SL→BE) → TP/BE"""
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


def _exit_no_half(highs, lows, ep, sl, tp, d):
    """半利確なし: 100%ポジションでTP/SL"""
    lim = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if d == 1:
            if lo <= sl: return sl, "loss"
            if h  >= tp: return tp, "win"
        else:
            if h  >= sl: return sl, "loss"
            if lo <= tp: return tp, "win"
    return None, None

# ══════════════════════════════════════════════════════════════════
# シミュレーション
# ══════════════════════════════════════════════════════════════════

def simulate_half(signals, d1m, sym):
    """半利確あり"""
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        xp, result, half_done = _exit_with_half(
            m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
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
        trades.append({"result": result, "pnl": half_pnl + pnl,
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)
    return trades, equity, mdd


def simulate_nohalf(signals, d1m, sym):
    """半利確なし（100%全利確）"""
    if not signals: return [], INIT_CASH, 0.0
    rm = RiskManager(sym, risk_pct=0.02)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        xp, result = _exit_no_half(
            m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["dir"])
        if result is None: continue
        pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, lot, USDJPY_RATE, sig["ep"])
        equity += pnl
        trades.append({"result": result, "pnl": pnl,
                       "month": sig["time"].strftime("%Y-%m")})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)
    return trades, equity, mdd

# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades, init=INIT_CASH):
    if len(trades) < 10: return {}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr = len(wins) / n
    gw = wins.sum(); gl = abs(loss.sum())
    pf = gw / gl if gl > 0 else float("inf")
    total_pnl = df["pnl"].sum()
    monthly = df.groupby("month")["pnl"].sum()
    plus_m = (monthly > 0).sum()
    eq = init; monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq if eq > 0 else 0
        monthly_ret.append(ret); eq += monthly[m]
    mr = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0.0
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(loss.mean()) if len(loss) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 and avg_w > 0 else 0
    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe, "kelly": kelly,
            "plus_m": plus_m, "total_m": len(monthly), "total_pnl": total_pnl,
            "final_eq": eq}


# ══════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*130)
    print("  RR倍率 × 半利確有無 バックテスト比較")
    print("  半利確あり: 1R→50%決済(SL→BE) → TP / 半利確なし: 100%でTP/SL")
    print("="*130)

    # テストするバリアント: (label, rr, use_half)
    VARIANTS = [
        ("現行 半利+2.5R",  2.5, True),
        ("半利+2.0R",       2.0, True),
        ("半利+3.0R",       3.0, True),
        ("全利 2.0R",       2.0, False),
        ("全利 2.5R",       2.5, False),
        ("全利 3.0R",       3.0, False),
        ("全利 1.5R",       1.5, False),
    ]

    all_results = []

    for tgt in TARGETS:
        sym = tgt["sym"]
        print(f"\n  {sym} ...", end=" ", flush=True)

        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("データなし"); continue

        is_d, oos_d, split_ts = split_is_oos(d1m_full)
        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(oos_d, 10).to_dict()
        m1c    = {"idx": oos_d.index, "opens": oos_d["open"].values,
                  "closes": oos_d["close"].values,
                  "highs":  oos_d["high"].values, "lows": oos_d["low"].values}

        # IS用
        atr_d_is = calc_atr(is_d, 10).to_dict()
        m1c_is   = {"idx": is_d.index, "opens": is_d["open"].values,
                    "closes": is_d["close"].values,
                    "highs":  is_d["high"].values, "lows": is_d["low"].values}

        sym_results = {"sym": sym}

        for label, rr, use_half in VARIANTS:
            # OOS
            sigs_oos = generate_signals(oos_d, d4h_full, spread, tgt["logic"],
                                        atr_d, m1c, tol=tgt["tol"], rr=rr)
            if use_half:
                t, eq, mdd = simulate_half(sigs_oos, oos_d, sym)
            else:
                t, eq, mdd = simulate_nohalf(sigs_oos, oos_d, sym)
            st = calc_stats(t)
            if st: st["mdd"] = mdd

            # IS
            sigs_is = generate_signals(is_d, d4h_full, spread, tgt["logic"],
                                       atr_d_is, m1c_is, tol=tgt["tol"], rr=rr)
            if use_half:
                t_is, _, mdd_is = simulate_half(sigs_is, is_d, sym)
            else:
                t_is, _, mdd_is = simulate_nohalf(sigs_is, is_d, sym)
            st_is = calc_stats(t_is)

            sym_results[label] = {
                "oos": st,
                "is_pf": st_is.get("pf", 0) if st_is else 0,
            }

        all_results.append(sym_results)
        print("完了")

    # ── 結果テーブル ──────────────────────────────────────────────
    v_labels = [v[0] for v in VARIANTS]

    print("\n" + "="*140)
    print("  ■ OOS期間 比較結果")
    print(f"  {'銘柄':8} {'バリアント':18} | {'n':>4} {'WR':>6} {'PF':>6} {'Sharpe':>7} "
          f"{'Kelly':>6} {'MDD':>7} {'月+':>5} {'総PnL':>12} {'IS PF':>7} {'OOS/IS':>7}")
    print("-"*140)

    csv_rows = []
    for r in all_results:
        sym = r["sym"]
        for vl in v_labels:
            data = r.get(vl, {})
            st = data.get("oos", {})
            if not st:
                continue
            is_pf = data.get("is_pf", 0)
            oos_is = st["pf"] / is_pf if is_pf > 0 and st["pf"] < 99 else 0
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            marker = " ★" if vl == "現行 半利+2.5R" else ""
            print(f"  {sym:8} {vl:18} | "
                  f"{st['n']:>4} {st['wr']*100:>5.1f}% {pf_s:>6} {st['sharpe']:>7.2f} "
                  f"{st['kelly']:>6.3f} {st['mdd']:>6.1f}% "
                  f"{st['plus_m']:>2}/{st['total_m']:<2} "
                  f"{st['total_pnl']:>11,.0f} {is_pf:>7.2f} {oos_is:>7.2f}{marker}")
            csv_rows.append({
                "sym": sym, "variant": vl, "n": st["n"],
                "wr": round(st["wr"]*100, 1), "pf": round(st["pf"], 2),
                "sharpe": round(st["sharpe"], 2), "kelly": round(st["kelly"], 3),
                "mdd": round(st["mdd"], 1),
                "plus_m": st["plus_m"], "total_m": st["total_m"],
                "total_pnl": round(st["total_pnl"]),
                "is_pf": round(is_pf, 2), "oos_is_ratio": round(oos_is, 2),
            })
        print()

    # ── 全銘柄平均 ────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 全銘柄平均比較（OOS）")
    print(f"  {'バリアント':18} | {'avg PF':>8} {'avg Sharpe':>11} {'avg MDD':>9} "
          f"{'avg Kelly':>10} {'合計PnL':>14} {'avg WR':>8}")
    print("-"*90)

    for vl in v_labels:
        pfs = []; shs = []; mdds = []; kls = []; pnls = []; wrs = []
        for r in all_results:
            st = r.get(vl, {}).get("oos", {})
            if st and st.get("pf", 0) < 99:
                pfs.append(st["pf"]); shs.append(st["sharpe"])
                mdds.append(st["mdd"]); kls.append(st["kelly"])
                pnls.append(st["total_pnl"]); wrs.append(st["wr"]*100)
        if pfs:
            marker = " ★" if vl == "現行 半利+2.5R" else ""
            print(f"  {vl:18} | {np.mean(pfs):>8.2f} {np.mean(shs):>11.2f} "
                  f"{np.mean(mdds):>8.1f}% {np.mean(kls):>10.3f} "
                  f"{sum(pnls):>13,.0f} {np.mean(wrs):>7.1f}%{marker}")

    # ── 現行比 改善率 ─────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 現行（半利+2.5R）比 変化率（OOS）")
    print(f"  {'銘柄':8} | {'半利+2.0R':>10} {'半利+3.0R':>10} "
          f"{'全利1.5R':>10} {'全利2.0R':>10} {'全利2.5R':>10} {'全利3.0R':>10}")
    print("-"*85)

    compare_vs = ["半利+2.0R", "半利+3.0R", "全利 1.5R", "全利 2.0R", "全利 2.5R", "全利 3.0R"]

    for metric, label in [("pf", "PF"), ("total_pnl", "総PnL"), ("mdd", "MDD")]:
        print(f"\n  {'':8} [{label}]")
        for r in all_results:
            sym = r["sym"]
            base_st = r.get("現行 半利+2.5R", {}).get("oos", {})
            base = base_st.get(metric, 0) if base_st else 0
            if base == 0: continue
            parts = []
            for cv in compare_vs:
                st = r.get(cv, {}).get("oos", {})
                val = st.get(metric, 0) if st else 0
                diff = ((val - base) / abs(base) * 100)
                parts.append(f"{diff:>+9.1f}%")
            print(f"  {sym:8} | {''.join(parts)}")

    # CSV保存
    out_path = os.path.join(OUT_DIR, "backtest_rr_comparison.csv")
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)
    print(f"\n  結果保存: {out_path}")


if __name__ == "__main__":
    main()
