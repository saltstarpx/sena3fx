"""
backtest_portfolio_680k_compound.py
====================================
採用6銘柄ポートフォリオ — 初期資金68万円 複利運用バックテスト
リスク2%版 と リスク1%版 を同時実行し比較
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 680_000
RR_RATIO      = 2.5
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

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "label": "USDJPY (Logic-C)"},
    {"sym": "GBPUSD",  "logic": "A", "label": "GBPUSD (Logic-A)"},
    {"sym": "USDCAD",  "logic": "A", "label": "USDCAD (Logic-A)"},
    {"sym": "NZDUSD",  "logic": "A", "label": "NZDUSD (Logic-A)"},
    {"sym": "AUDUSD",  "logic": "B", "label": "AUDUSD (Logic-B)"},
    {"sym": "XAUUSD",  "logic": "A", "label": "XAUUSD (Logic-A)"},
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

def load_1m(sym):
    sym_l = sym.lower()
    for p in [os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv"),
              os.path.join(DATA_DIR, f"{sym_l}_1m.csv")]:
        if os.path.exists(p):
            df = load_csv(p)
            if len(df) < 10: continue
            return df
    return None

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
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, sym):
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
        if abs(v1 - v2) > atr1h * A3_DEFAULT_TOL: continue

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
                            "tp": raw + d * risk * RR_RATIO, "risk": risk,
                            "sym": sym, "spread": spread})
            used.add(et)

    return signals

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

# ── 複利シミュレーション ──────────────────────────────────────────
def simulate_compound(all_signals, m1_data, risk_pct, equity_cap=None):
    """全銘柄のシグナルを時系列で統合し、現在エクイティ基準でロット計算
    equity_cap: ロット計算に使うエクイティの上限（Noneなら無制限）
    """
    equity = INIT_CASH
    peak   = INIT_CASH
    mdd    = 0.0
    trades = []
    eq_history = [{"time": all_signals[0]["time"] - pd.Timedelta(days=1), "equity": equity}]

    rm_cache = {}
    for sig in all_signals:
        sym = sig["sym"]
        if sym not in rm_cache:
            rm_cache[sym] = RiskManager(sym, risk_pct=risk_pct)
        rm = rm_cache[sym]
        rm.risk_pct = risk_pct

        d1m = m1_data[sym]
        m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values

        # ロット計算用エクイティ（キャップ適用）
        lot_equity = min(equity, equity_cap) if equity_cap else equity
        lot = rm.calc_lot(lot_equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0: continue

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
            equity += half_pnl
            rem = lot * 0.5
        else:
            rem = lot

        pnl    = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total   = half_pnl + pnl

        trades.append({"time": sig["time"], "sym": sym, "result": result,
                        "pnl": total, "month": sig["time"].strftime("%Y-%m"),
                        "equity": equity})
        eq_history.append({"time": sig["time"], "equity": equity})

        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, eq_history, mdd

# ── 統計計算 ──────────────────────────────────────────────────────
def calc_stats(trades):
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")

    monthly = df.groupby("month")["pnl"].sum()
    plus_m = (monthly > 0).sum()

    eq_t = INIT_CASH
    monthly_ret = []
    for m in monthly.index:
        ret = monthly[m] / eq_t if eq_t > 0 else 0
        monthly_ret.append(ret)
        eq_t += monthly[m]
    mr = np.array(monthly_ret)
    sharpe = (mr.mean() / mr.std()) * np.sqrt(12) if len(mr) >= 2 and mr.std() > 0 else 0

    return {"n": n, "wr": wr, "pf": pf, "sharpe": sharpe,
            "plus_m": plus_m, "total_m": len(monthly),
            "final_eq": trades[-1]["equity"],
            "total_pnl": trades[-1]["equity"] - INIT_CASH,
            "monthly": monthly}

def fmt_yen(v):
    """日本円を万/億/兆で読みやすく表示"""
    av = abs(v)
    sign = "+" if v > 0 else ("-" if v < 0 else "")
    if av >= 1e12:
        return f"{sign}¥{v/1e12:,.1f}兆"
    elif av >= 1e8:
        return f"{sign}¥{v/1e8:,.1f}億"
    elif av >= 1e4:
        return f"{sign}¥{v/1e4:,.0f}万"
    else:
        return f"¥{v:,.0f}"

# ── メイン ───────────────────────────────────────────────────────
def main():
    print(f"\n{'='*90}")
    print(f"  YAGAMI改 複利ポートフォリオBT — ¥{INIT_CASH:,.0f} スタート")
    print(f"  リスク2% vs リスク1% 比較")
    print(f"{'='*90}")

    # 全銘柄のシグナルを収集
    all_signals = []
    m1_data = {}

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        print(f"  {tgt['label']} ... ", end="", flush=True)

        d1m = load_1m(sym)
        if d1m is None:
            print("データ未発見"); continue

        m1_data[sym] = d1m

        d4h = d1m.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs":  d1m["high"].values, "lows": d1m["low"].values}

        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, sym)
        all_signals.extend(sigs)
        print(f"{len(sigs)}シグナル")

    all_signals.sort(key=lambda x: x["time"])
    print(f"\n  合計シグナル: {len(all_signals)}")

    # 2%と1%で実行
    results = {}
    for risk_pct, label in [(0.02, "2%"), (0.01, "1%")]:
        print(f"\n  複利シミュレーション（リスク{label}）... ", end="", flush=True)
        trades, eq_hist, mdd = simulate_compound(all_signals, m1_data, risk_pct)
        st = calc_stats(trades)
        st["mdd"] = mdd
        st["eq_hist"] = eq_hist
        st["trades"] = trades
        results[label] = st
        print(f"完了 → {fmt_yen(st['final_eq'])}")

    # ── 比較表示 ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ■ 複利運用 最終結果比較")
    print(f"{'='*90}")
    print(f"  {'':20} {'リスク2%':>24} {'リスク1%':>24}")
    print(f"  {'-'*68}")
    r2 = results["2%"]; r1 = results["1%"]
    fe2 = r2["final_eq"]; fe1 = r1["final_eq"]
    tp2 = r2["total_pnl"]; tp1 = r1["total_pnl"]
    print(f"  {'初期資金':20} {fmt_yen(INIT_CASH):>24} {fmt_yen(INIT_CASH):>24}")
    print(f"  {'最終資産':20} {fmt_yen(fe2):>24} {fmt_yen(fe1):>24}")
    print(f"  {'総損益':20} {fmt_yen(tp2):>24} {fmt_yen(tp1):>24}")
    print(f"  {'リターン':20} {tp2/INIT_CASH*100:>23.1f}% {tp1/INIT_CASH*100:>23.1f}%")
    print(f"  {'トレード数':20} {r2['n']:>24} {r1['n']:>24}")
    print(f"  {'勝率':20} {r2['wr']*100:>23.1f}% {r1['wr']*100:>23.1f}%")
    print(f"  {'PF':20} {r2['pf']:>24.2f} {r1['pf']:>24.2f}")
    print(f"  {'月次シャープ':20} {r2['sharpe']:>24.2f} {r1['sharpe']:>24.2f}")
    print(f"  {'最大DD':20} {r2['mdd']:>23.1f}% {r1['mdd']:>23.1f}%")
    pm2 = f"{r2['plus_m']}/{r2['total_m']}"; pm1 = f"{r1['plus_m']}/{r1['total_m']}"
    print(f"  {'プラス月':20} {pm2:>24} {pm1:>24}")

    # ── 月末残高推移 ──────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  ■ 月末残高推移（複利）")
    print(f"  {'月':10} {'リスク2% 残高':>16} {'月次損益':>16} {'リスク1% 残高':>16} {'月次損益':>16}")
    print(f"  {'-'*78}")

    months_2 = r2["monthly"]
    months_1 = r1["monthly"]
    all_months = sorted(set(months_2.index) | set(months_1.index))

    eq2 = INIT_CASH; eq1 = INIT_CASH
    for m in all_months:
        p2 = months_2.get(m, 0)
        p1 = months_1.get(m, 0)
        eq2 += p2
        eq1 += p1
        print(f"  {m:10} {fmt_yen(eq2):>16} {fmt_yen(p2):>16} {fmt_yen(eq1):>16} {fmt_yen(p1):>16}")

    # ── 銘柄別内訳 ────────────────────────────────────────────────
    for label in ["2%", "1%"]:
        r = results[label]
        df_t = pd.DataFrame(r["trades"])
        print(f"\n  ■ 銘柄別内訳（リスク{label}）")
        print(f"  {'銘柄':10} {'トレード':>8} {'勝率':>8} {'損益':>18}")
        print(f"  {'-'*48}")
        for sym_name in [t["sym"] for t in TARGETS]:
            sub = df_t[df_t["sym"] == sym_name]
            if sub.empty: continue
            sw = len(sub[sub["pnl"] > 0])
            print(f"  {sym_name:10} {len(sub):>8} {sw/len(sub)*100:>7.1f}% {fmt_yen(sub['pnl'].sum()):>18}")

    # ── エクイティカーブ描画 ──────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import LogLocator, FuncFormatter

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    colors = {"2%": "#E53935", "1%": "#2196F3"}

    ax = axes[0]
    for label in ["2%", "1%"]:
        r = results[label]
        eq_df = pd.DataFrame(r["eq_hist"])
        times = eq_df["time"].values
        eqs   = eq_df["equity"].values
        final = r["final_eq"]
        ax.plot(times, eqs, color=colors[label], linewidth=1.5,
                label=f"Risk {label}: {fmt_yen(INIT_CASH)} -> {fmt_yen(final)}")

    ax.axhline(y=INIT_CASH, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yscale("log")
    ax.set_title(f"YAGAMI Kai Compound Portfolio (Log Scale) | Start: {fmt_yen(INIT_CASH)}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (JPY) - Log Scale")

    def log_yen_fmt(x, _):
        if x >= 1e12: return f"{x/1e12:.0f}T"
        if x >= 1e8:  return f"{x/1e8:.0f}B"
        if x >= 1e6:  return f"{x/1e6:.0f}M"
        if x >= 1e4:  return f"{x/1e4:.0f}W"
        return f"{x:,.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(log_yen_fmt))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    # DD比較（下段）
    ax2 = axes[1]
    for label in ["2%", "1%"]:
        r = results[label]
        eq_df = pd.DataFrame(r["eq_hist"])
        eqs = eq_df["equity"].values
        times = eq_df["time"].values
        peak_arr = np.maximum.accumulate(eqs)
        dd_pct = (peak_arr - eqs) / peak_arr * 100
        ax2.fill_between(times, 0, dd_pct, alpha=0.3, color=colors[label],
                          label=f"DD {label} (max {r['mdd']:.1f}%)")
        ax2.plot(times, dd_pct, color=colors[label], linewidth=0.8)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(axis="x", rotation=45)
    ax2.invert_yaxis()
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "equity_curve_680k_compound.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  エクイティカーブ保存: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
