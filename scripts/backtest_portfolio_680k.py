"""
backtest_portfolio_680k.py
==========================
採用6銘柄ポートフォリオ — 初期資金68万円バックテスト
エクイティカーブ（折れ線グラフ）を results/equity_curve_680k.png に出力
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 定数 ─────────────────────────────────────────────────────────
INIT_CASH     = 680_000
RR_RATIO      = 2.5
HALF_R        = 1.0
USDJPY_RATE   = 150.0
MAX_LOOKAHEAD = 20_000
RISK_PCT      = 0.02   # 固定2%

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

# ── 採用6銘柄 ────────────────────────────────────────────────────
TARGETS = [
    {"sym": "USDJPY",  "logic": "C", "label": "USDJPY (Logic-C)"},
    {"sym": "GBPUSD",  "logic": "A", "label": "GBPUSD (Logic-A)", "ema_dist_min": 1.5},
    {"sym": "USDCAD",  "logic": "A", "label": "USDCAD (Logic-A)"},
    {"sym": "NZDUSD",  "logic": "A", "label": "NZDUSD (Logic-A)"},
    {"sym": "AUDUSD",  "logic": "B", "label": "AUDUSD (Logic-B)", "ema_dist_min": 1.5},
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
            # LFSポインタチェック
            if len(df) < 10:
                continue
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
def generate_signals(d1m, d4h_full, spread, logic, atr_d, m1c, ema_dist_min=A1_EMA_DIST_MIN):
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
        # EMA距離フィルター（per-symbol ema_dist_min対応）
        if logic != "C" and not pd.isna(h4l["atr"]) and h4l["atr"] > 0:
            ema_dist = abs(h4l["close"] - h4l["ema20"])
            if ema_dist < ema_dist_min * h4l["atr"]:
                continue

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
                            "tp": raw + d * risk * RR_RATIO, "risk": risk})
            used.add(et)

    return sorted(signals, key=lambda x: x["time"])

# ── トレード判定 ──────────────────────────────────────────────────
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

# ── メイン ───────────────────────────────────────────────────────
def main():
    print(f"\n{'='*80}")
    print(f"  YAGAMI改 ポートフォリオバックテスト — 初期資金 ¥{INIT_CASH:,.0f}")
    print(f"  採用6銘柄 × 固定リスク2%")
    print(f"{'='*80}")

    # 全銘柄のシグナルを時系列で統合
    all_trades = []

    for tgt in TARGETS:
        sym   = tgt["sym"]
        logic = tgt["logic"]
        print(f"\n  {tgt['label']} ... ", end="", flush=True)

        d1m = load_1m(sym)
        if d1m is None:
            print("データ未発見"); continue

        # 4hをresampleで生成
        d4h = d1m.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])

        cfg    = SYMBOL_CONFIG[sym]
        spread = cfg["spread"] * cfg["pip"]
        atr_d  = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values,
                  "highs":  d1m["high"].values, "lows": d1m["low"].values}

        edm  = tgt.get("ema_dist_min", A1_EMA_DIST_MIN)
        sigs = generate_signals(d1m, d4h, spread, logic, atr_d, m1c, ema_dist_min=edm)
        print(f"{len(sigs)}シグナル", end="", flush=True)

        # シミュレーション（個別トレード記録）
        rm = RiskManager(sym, risk_pct=RISK_PCT)
        m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values

        sym_trades = 0
        for sig in sigs:
            rm.risk_pct = RISK_PCT
            lot = rm.calc_lot(INIT_CASH, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
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
                rem = lot * 0.5
            else:
                rem = lot

            pnl   = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
            total = half_pnl + pnl

            all_trades.append({
                "time": sig["time"], "sym": sym, "result": result,
                "pnl": total, "month": sig["time"].strftime("%Y-%m")
            })
            sym_trades += 1

        print(f" → {sym_trades}トレード")

    if not all_trades:
        print("\nトレードなし"); return

    # 時系列ソート
    all_trades.sort(key=lambda x: x["time"])
    df = pd.DataFrame(all_trades)

    # エクイティカーブ計算
    equity = INIT_CASH
    eq_series = [{"time": df.iloc[0]["time"] - pd.Timedelta(days=1), "equity": equity}]
    for _, row in df.iterrows():
        equity += row["pnl"]
        eq_series.append({"time": row["time"], "equity": equity})

    eq_df = pd.DataFrame(eq_series)
    final_equity = equity
    total_pnl = final_equity - INIT_CASH

    # 統計
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    wr = len(wins) / n
    gw = wins["pnl"].sum(); gl = abs(losses["pnl"].sum())
    pf = gw / gl if gl > 0 else float("inf")

    peak = INIT_CASH; mdd = 0; eq = INIT_CASH
    for _, row in df.iterrows():
        eq += row["pnl"]
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak * 100)

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

    # 銘柄別内訳
    print(f"\n{'='*80}")
    print(f"  ■ 銘柄別トレード内訳")
    print(f"  {'銘柄':10} {'トレード':>8} {'勝率':>8} {'損益(¥)':>14}")
    print(f"  {'-'*44}")
    for sym_name in [t["sym"] for t in TARGETS]:
        sub = df[df["sym"] == sym_name]
        if sub.empty: continue
        sw = len(sub[sub["pnl"] > 0])
        print(f"  {sym_name:10} {len(sub):>8} {sw/len(sub)*100:>7.1f}% {sub['pnl'].sum():>+14,.0f}")

    print(f"\n{'='*80}")
    print(f"  ■ ポートフォリオ最終結果")
    print(f"{'='*80}")
    print(f"  初期資金        : ¥{INIT_CASH:>14,.0f}")
    print(f"  最終資産        : ¥{final_equity:>14,.0f}")
    print(f"  総損益          : ¥{total_pnl:>+14,.0f} ({total_pnl/INIT_CASH*100:+.1f}%)")
    print(f"  トレード数      : {n:>8}")
    print(f"  勝率            : {wr*100:>7.1f}%")
    print(f"  プロフィットF   : {pf:>8.2f}")
    print(f"  月次シャープ    : {sharpe:>8.2f}")
    print(f"  最大DD          : {mdd:>7.1f}%")
    print(f"  プラス月        : {plus_m}/{len(monthly)}")
    print(f"  期間            : {df.iloc[0]['time'].strftime('%Y-%m-%d')} 〜 {df.iloc[-1]['time'].strftime('%Y-%m-%d')}")

    # 月次損益
    print(f"\n  ■ 月次損益")
    eq_m = INIT_CASH
    for m in monthly.index:
        pnl_m = monthly[m]
        eq_m += pnl_m
        mark = "+" if pnl_m > 0 else " "
        print(f"    {m} : ¥{pnl_m:>+12,.0f}  (残高 ¥{eq_m:>12,.0f})")

    # ── エクイティカーブ描画 ──────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams

    rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(14, 6))

    times = eq_df["time"].values
    eqs   = eq_df["equity"].values

    ax.plot(times, eqs, color="#2196F3", linewidth=1.5, label="Portfolio Equity")
    ax.axhline(y=INIT_CASH, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label=f"Initial: ¥{INIT_CASH:,.0f}")

    # 最大DDの区間を塗る
    peak_arr = np.maximum.accumulate(eqs)
    dd_arr = (peak_arr - eqs) / peak_arr * 100
    ax.fill_between(times, eqs, peak_arr, alpha=0.15, color="red", label="Drawdown")

    ax.set_title(f"YAGAMI Kai Portfolio — ¥{INIT_CASH:,.0f} Start → ¥{final_equity:,.0f} (+¥{total_pnl:,.0f})",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (JPY)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # サブテキスト
    info = (f"PF={pf:.2f}  WR={wr*100:.1f}%  Sharpe={sharpe:.2f}  "
            f"MDD={mdd:.1f}%  Trades={n}  Months+={plus_m}/{len(monthly)}")
    ax.text(0.5, -0.18, info, transform=ax.transAxes, ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "equity_curve_680k.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  エクイティカーブ保存: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
