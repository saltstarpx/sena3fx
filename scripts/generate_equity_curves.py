"""
generate_equity_curves.py
=========================
全7本番銘柄の現行構成でバックテストを実行し、
エクイティカーブ（累積PnL推移）をPNGグラフとして出力する。

銘柄別構成:
  GBPUSD:  v80 (KMID+KLOW+Body, E0, RR=3.0)
  EURUSD:  v80 (KMID+KLOW+Body, E0, RR=3.0)
  USDCAD:  v80 (KMID+KLOW+Body, E0, RR=3.0)
  NZDUSD:  v79A (use_1d_trend, tol=0.20, RR=2.5)
  XAUUSD:  v79A (use_1d_trend, tol=0.20, RR=2.5)
  AUDUSD:  v79BC (ADX+Streak+Body, RR=2.5)
  USDJPY:  v77 Logic-C (E0, RR=2.5)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 1_000_000
RISK_PCT      = 0.02
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
E0_WINDOW_MIN   = 2
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3
E1_MAX_WAIT_MIN = 5
ADX_MIN         = 20
STREAK_MIN      = 4
BODY_RATIO_MIN  = 0.3

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 銘柄×構成定義
TARGETS = [
    {"sym": "GBPUSD",  "logic": "v80",  "tol": 0.30, "rr": 3.0,
     "kwargs": {"use_kmid": True, "use_klow": True, "use_body_ratio": True}},
    {"sym": "EURUSD",  "logic": "v80",  "tol": 0.30, "rr": 3.0,
     "kwargs": {"use_kmid": True, "use_klow": True, "use_body_ratio": True}},
    {"sym": "USDCAD",  "logic": "v80",  "tol": 0.30, "rr": 3.0,
     "kwargs": {"use_kmid": True, "use_klow": True, "use_body_ratio": True}},
    {"sym": "NZDUSD",  "logic": "A",    "tol": 0.20, "rr": 2.5, "kwargs": {}},
    {"sym": "XAUUSD",  "logic": "A",    "tol": 0.20, "rr": 2.5, "kwargs": {}},
    {"sym": "AUDUSD",  "logic": "B",    "tol": 0.30, "rr": 2.5, "kwargs": {}},
    {"sym": "USDJPY",  "logic": "C",    "tol": 0.30, "rr": 2.5, "kwargs": {}},
]

# ── データロード ─────────────────────────────────────────────────
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
    d1m = None
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p):
            d1m = load_csv(p); break

    # USDJPY: 1mがなければ15mで代用
    if d1m is None:
        for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_15m.csv"),
                  os.path.join(DATA_DIR, f"{sym_l}_15m.csv"),
                  os.path.join(DATA_DIR, f"{sym_l}_is_15m.csv")]:
            if os.path.exists(p):
                d1m = load_csv(p); break
    if d1m is None:
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
def chk_kmid(b, d):
    return (d == 1 and b["close"] > b["open"]) or (d == -1 and b["close"] < b["open"])

def chk_klow(b):
    return (min(b["open"], b["close"]) - b["low"]) / b["open"] < KLOW_THR if b["open"] > 0 else False

def chk_ema(b):
    return not pd.isna(b["atr"]) and b["atr"] > 0 and abs(b["close"] - b["ema20"]) >= b["atr"] * A1_EMA_DIST_MIN

def chk_body_ratio(b, min_ratio=BODY_RATIO_MIN):
    rng = b["high"] - b["low"]
    if rng <= 0: return False
    return abs(b["close"] - b["open"]) / rng >= min_ratio

# ── シグナル生成: 現行 Logic A/B/C ──────────────────────────────
def generate_signals_current(d1m, d4h_full, spread, logic, atr_d, m1c, tol=0.30, rr=2.5):
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

# ── シグナル生成: v80 ──────────────────────────────────────────
def generate_signals_v80(d1m, d4h_full, spread, m1c, tol=0.30, rr=2.5,
                          use_kmid=True, use_klow=False, use_body_ratio=False):
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
        if use_body_ratio and not chk_body_ratio(h4l): continue
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

# ── EXIT + シミュレーション（エクイティカーブ記録付き） ────────
def _exit_with_half(highs, lows, ep, sl, tp, risk, d):
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

def simulate_with_curve(signals, d1m, sym):
    """シミュレーション実行 + エクイティカーブ（時刻, equity）を返す"""
    if not signals:
        return [], INIT_CASH, 0.0, []

    rm = RiskManager(sym, risk_pct=RISK_PCT)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0
    curve = [(d1m.index[0], INIT_CASH)]  # initial point

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
                       "time": sig["time"], "month": sig["time"].strftime("%Y-%m")})
        curve.append((sig["time"], equity))
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, curve

# ── メイン ─────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("  Equity Curve Generator - 7 Production Symbols")
    print("  INIT_CASH = {:,.0f} JPY / risk = {:.0%}".format(INIT_CASH, RISK_PCT))
    print("="*80)

    all_curves = {}  # sym -> [(time, equity), ...]
    all_stats  = {}  # sym -> {trades, final_eq, mdd}

    for tgt in TARGETS:
        sym = tgt["sym"]
        logic = tgt["logic"]
        print(f"\n  Processing {sym} ({logic}) ...", end=" ", flush=True)

        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("No data"); continue

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m_full, 10).to_dict()
        m1c    = {"idx": d1m_full.index, "opens": d1m_full["open"].values,
                  "closes": d1m_full["close"].values,
                  "highs":  d1m_full["high"].values, "lows": d1m_full["low"].values}

        if logic == "v80":
            sigs = generate_signals_v80(
                d1m_full, d4h_full, spread, m1c,
                tol=tgt["tol"], rr=tgt["rr"], **tgt["kwargs"])
        else:
            sigs = generate_signals_current(
                d1m_full, d4h_full, spread, logic,
                atr_d, m1c, tol=tgt["tol"], rr=tgt["rr"])

        trades, final_eq, mdd, curve = simulate_with_curve(sigs, d1m_full, sym)
        all_curves[sym] = curve
        n_trades = len(trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        all_stats[sym] = {"n": n_trades, "wr": wr, "final_eq": final_eq,
                          "mdd": mdd, "total_pnl": total_pnl, "logic": logic,
                          "rr": tgt["rr"]}
        print(f"done ({n_trades} trades, WR={wr:.1f}%, PnL={total_pnl:+,.0f}, MDD={mdd:.1f}%)")

    # ── ポートフォリオ合成 ───────────────────────────────────────
    # 各銘柄のcurveを日次にリサンプルして合算
    sym_daily = {}
    min_date = None; max_date = None
    for sym, curve in all_curves.items():
        if len(curve) < 2: continue
        df = pd.DataFrame(curve, columns=["time", "equity"])
        df = df.set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        # forward-fill to daily
        daily = df.resample("1D").last().ffill()
        sym_daily[sym] = daily
        sd = daily.index[0]; ed = daily.index[-1]
        if min_date is None or sd < min_date: min_date = sd
        if max_date is None or ed > max_date: max_date = ed

    # Align all to same date range
    date_range = pd.date_range(min_date, max_date, freq="1D")
    portfolio = pd.DataFrame(index=date_range)
    for sym, daily in sym_daily.items():
        s = daily.reindex(date_range).ffill().bfill()
        portfolio[sym] = s["equity"]

    # Portfolio = sum of individual PnLs (each starting from INIT_CASH)
    # Total portfolio equity = sum of all symbol equities
    portfolio["Portfolio"] = portfolio.sum(axis=1)
    portfolio_init = INIT_CASH * len(sym_daily)
    portfolio_final = portfolio["Portfolio"].iloc[-1] if len(portfolio) > 0 else portfolio_init

    # ── サマリー出力 ───────────────────────────────────────────
    print("\n" + "="*80)
    print("  Summary")
    print(f"  {'Symbol':8} {'Logic':6} {'RR':>4} {'Trades':>7} {'WR':>6} "
          f"{'Final Eq':>14} {'PnL':>14} {'MDD':>6}")
    print("  " + "-"*75)
    for tgt in TARGETS:
        sym = tgt["sym"]
        if sym not in all_stats: continue
        st = all_stats[sym]
        print(f"  {sym:8} {st['logic']:6} {st['rr']:>4.1f} {st['n']:>7} {st['wr']:>5.1f}% "
              f"{st['final_eq']:>13,.0f} {st['total_pnl']:>+13,.0f} {st['mdd']:>5.1f}%")
    print("  " + "-"*75)
    print(f"  {'Portfolio':8} {'':6} {'':>4} {'':>7} {'':>6} "
          f"{portfolio_final:>13,.0f} {portfolio_final - portfolio_init:>+13,.0f}")

    # ── グラフ生成 ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("\n  matplotlib not installed - skipping graph generation")
        return

    # Try Japanese font
    try:
        from matplotlib import font_manager
        jp_fonts = [f for f in font_manager.findSystemFonts()
                    if any(n in f.lower() for n in ["noto", "ipag", "takao", "migmix", "mplus"])]
        if jp_fonts:
            prop = font_manager.FontProperties(fname=jp_fonts[0])
            plt.rcParams["font.family"] = prop.get_name()
    except Exception:
        pass

    colors = {sym: SYMBOL_CONFIG[sym]["color"] for sym in all_curves}
    portfolio_color = "#1e3a5f"

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), height_ratios=[2, 2, 1.2],
                              gridspec_kw={"hspace": 0.30})
    ax_log, ax_norm, ax_port = axes

    # --- Panel 1: Individual equity curves (LOG scale) ---
    for sym in [t["sym"] for t in TARGETS]:
        if sym not in sym_daily: continue
        daily = sym_daily[sym]
        label = f"{sym} ({all_stats[sym]['logic']}, {all_stats[sym]['rr']:.1f}R)"
        ax_log.plot(daily.index, daily["equity"].values,
                    label=label,
                    color=colors.get(sym, "#888888"), linewidth=1.3, alpha=0.85)

    ax_log.axhline(y=INIT_CASH, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("Equity (JPY, log scale)", fontsize=11)
    ax_log.set_title(
        f"YAGAMI Production Equity Curves (7 Symbols) - Log Scale\n"
        f"Initial: {INIT_CASH:,.0f} JPY per symbol | Risk: {RISK_PCT:.0%} fixed | Compound growth",
        fontsize=13, fontweight="bold")
    ax_log.legend(loc="upper left", fontsize=9, ncol=2)
    ax_log.grid(True, alpha=0.3, which="both")
    ax_log.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_log.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # --- Panel 2: Normalized return (%) ---
    for sym in [t["sym"] for t in TARGETS]:
        if sym not in sym_daily: continue
        daily = sym_daily[sym]
        pct_return = (daily["equity"].values / INIT_CASH - 1) * 100
        label = f"{sym} (final: {daily['equity'].iloc[-1]/INIT_CASH:.0f}x)"
        ax_norm.plot(daily.index, pct_return,
                     label=label,
                     color=colors.get(sym, "#888888"), linewidth=1.3, alpha=0.85)

    ax_norm.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_norm.set_yscale("symlog", linthresh=100)
    ax_norm.set_ylabel("Return (%, symlog scale)", fontsize=11)
    ax_norm.set_title("Cumulative Return per Symbol (%)", fontsize=12, fontweight="bold")
    ax_norm.legend(loc="upper left", fontsize=8, ncol=2)
    ax_norm.grid(True, alpha=0.3)
    ax_norm.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_norm.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # --- Panel 3: Portfolio equity curve (LOG scale) ---
    if len(portfolio) > 0:
        ax_port.plot(portfolio.index, portfolio["Portfolio"].values,
                     label=f"Portfolio (7 symbols)",
                     color=portfolio_color, linewidth=2.0)
        ax_port.axhline(y=portfolio_init, color="gray", linestyle="--",
                         linewidth=0.8, alpha=0.5)
        ax_port.set_yscale("log")

        # Final value annotation
        def fmt_yen(v):
            if v >= 1e12: return f"{v/1e12:.1f}T"
            if v >= 1e8:  return f"{v/1e8:.1f}B"
            if v >= 1e4:  return f"{v/1e4:.0f}M"
            return f"{v:,.0f}"

        final_text = (f"Final: {fmt_yen(portfolio_final)} JPY\n"
                      f"({portfolio_final/portfolio_init:.0f}x return)")
        ax_port.annotate(final_text,
                         xy=(portfolio.index[-1], portfolio["Portfolio"].iloc[-1]),
                         xytext=(-200, -30), textcoords="offset points",
                         fontsize=10, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=portfolio_color),
                         color=portfolio_color)

    ax_port.set_ylabel("Portfolio Equity (JPY, log)", fontsize=11)
    ax_port.set_xlabel("Date", fontsize=11)
    ax_port.set_title(
        f"Combined Portfolio | Init: {portfolio_init:,.0f} JPY | "
        f"Final: {fmt_yen(portfolio_final)} JPY",
        fontsize=12, fontweight="bold")
    ax_port.legend(loc="upper left", fontsize=10)
    ax_port.grid(True, alpha=0.3, which="both")
    ax_port.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_port.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.autofmt_xdate(rotation=30)

    out_path = os.path.join(OUT_DIR, "equity_curves_v80_production.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Graph saved: {out_path}")


if __name__ == "__main__":
    main()
