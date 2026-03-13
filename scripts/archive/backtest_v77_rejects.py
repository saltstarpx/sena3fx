"""
backtest_v77_rejects.py
========================
Goldロジックで採用基準未達の銘柄をv77ロジックで再検証

【v77ロジック】
- 4H EMA20トレンドフィルター（日足EMA20なし ← Goldとの違い）
- KMID: 直前4H足の実体方向一致
- KLOW: 直前4H足の下ヒゲ比率 < 0.15%
- EMA距離: 4H終値とEMA20の距離 ≥ ATR×1.0
- 1H 二番底・二番天井パターン（ATR×0.3以内）
- E2エントリー（スパイク除外、2-3分以内）
- SL/TP: RR=2.5、半利確あり

【対象銘柄】Goldロジック PF<2.0 の全銘柄
  FX:  EURUSD / AUDUSD / GBPUSD / NZDUSD / USDCAD / USDCHF
  JPY: USDJPY
  IDX: NAS100 / US30
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import AdaptiveRiskManager, SYMBOL_CONFIG

# ── 設定 ─────────────────────────────────────────────────────────
START        = "2025-01-01"
END          = "2026-02-28"
INIT_CASH    = 100.0
BASE_RISK    = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
USDJPY_RATE  = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015    # 下ヒゲ比率 < 0.15%
A1_EMA_DIST_MIN = 1.0       # EMA距離 ≥ ATR×1.0
A3_DEFAULT_TOL  = 0.30      # 二番底/天井 ATR×0.3以内
E2_SPIKE_ATR    = 2.0       # スパイク除外閾値
E2_WINDOW_MIN   = 3         # エントリーウィンドウ（分）

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Gold通過組（XAUUSD/SPX500）を除いた全銘柄
SYMBOLS = [
    {"name": "EURUSD", "cat": "FX",  "gold_pf": 1.59},
    {"name": "AUDUSD", "cat": "FX",  "gold_pf": 1.94},
    {"name": "GBPUSD", "cat": "FX",  "gold_pf": 1.80},
    {"name": "NZDUSD", "cat": "FX",  "gold_pf": 1.78},
    {"name": "USDCAD", "cat": "FX",  "gold_pf": 1.52},
    {"name": "USDCHF", "cat": "FX",  "gold_pf": 1.83},
    {"name": "USDJPY", "cat": "JPY", "gold_pf": 1.74},
    {"name": "NAS100", "cat": "IDX", "gold_pf": 1.18},
    {"name": "US30",   "cat": "IDX", "gold_pf": 1.52},
]

CAT_COLORS = {"FX": "#4C9BE8", "JPY": "#ef4444", "IDX": "#7ED321"}

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tc = next((c for c in ["timestamp", "datetime"] if c in df.columns), df.columns[0])
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def resample_ohlcv(df, rule):
    cols = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        cols["volume"] = "sum"
    return df.resample(rule).agg(cols).dropna(subset=["open", "close"])

def load_all(sym):
    def _f(tf):
        p = os.path.join(DATA_DIR, f"{sym}_{tf}.csv")
        return load_csv(p) if os.path.exists(p) else None
    d1m  = _f("1m")
    d4h_raw  = _f("4h");  d4h  = d4h_raw  if d4h_raw  is not None else (resample_ohlcv(d1m, "4h")   if d1m is not None else None)
    d15m_raw = _f("15m"); d15m = d15m_raw if d15m_raw is not None else (resample_ohlcv(d1m, "15min") if d1m is not None else None)
    return d1m, d15m, d4h

def slice_period(df, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def build_4h(df4h):
    """v77: 4H EMA20のみ（日足フィルターなし）"""
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def build_1h(df15m):
    cols = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df15m.columns:
        cols["volume"] = "sum"
    df = df15m.resample("1h").agg(cols).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

# ── フィルター ────────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction ==-1 and bar["close"] < bar["open"])

def check_klow(bar):
    o = bar["open"]
    return (min(bar["open"], bar["close"]) - bar["low"]) / o < KLOW_THR if o > 0 else False

def check_ema_dist(bar):
    d = abs(bar["close"] - bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a > 0 and d >= a * A1_EMA_DIST_MIN

def pick_e2(signal_time, direction, spread, atr_1m_d, m1c):
    idx = m1c["idx"]
    s   = idx.searchsorted(signal_time, side="left")
    e   = idx.searchsorted(signal_time + pd.Timedelta(minutes=max(2, E2_WINDOW_MIN)), side="left")
    for i in range(s, min(e, len(idx))):
        bar_range = m1c["highs"][i] - m1c["lows"][i]
        atr_val   = atr_1m_d.get(idx[i], np.nan)
        if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR:
            continue
        return idx[i], m1c["opens"][i] + (spread if direction == 1 else -spread)
    return None, None

# ── シグナル生成（v77: 4H EMA20のみ、日足フィルターなし） ────────
def generate_signals_v77(d1m, d15m, d4h_full, spread, atr_1m_d, m1c):
    d4h = build_4h(d4h_full)
    d1h = build_1h(d15m)

    signals = []; used = set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct  = h1_times[i]
        h1_p1  = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h <= 0: continue

        h4_before = d4h[d4h.index < h1_ct]
        if len(h4_before) < 2: continue
        h4_lat = h4_before.iloc[-1]
        if pd.isna(h4_lat.get("atr", np.nan)): continue
        trend  = h4_lat["trend"]
        h4_atr = h4_lat["atr"]

        # v77コアフィルター（日足フィルターなし）
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat):        continue
        if not check_ema_dist(h4_lat):    continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction == 1: v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:              v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol: continue

        et, ep = pick_e2(h1_ct, direction, spread, atr_1m_d, m1c)
        if et is None or et in used: continue

        raw = ep - spread if direction == 1 else ep + spread
        if direction == 1: sl = min(v1, v2) - atr_1h * 0.15; risk = raw - sl
        else:              sl = max(v1, v2) + atr_1h * 0.15; risk = sl - raw

        if 0 < risk <= h4_atr * 2:
            tp = raw + direction * risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep,
                            "sl": sl, "tp": tp, "risk": risk})
            used.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── 出口探索 ─────────────────────────────────────────────────────
def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half = ep + direction * risk * HALF_R
    lim  = min(len(highs), MAX_LOOKAHEAD)
    for i in range(lim):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            if lo <= sl: return i, sl, "loss", False
            if h  >= tp: return i, tp, "win",  False
            if h  >= half:
                be = ep
                for j in range(i+1, lim):
                    if lows[j]  <= be: return j, be, "win", True
                    if highs[j] >= tp: return j, tp, "win", True
                return -1, None, None, True
        else:
            if h  >= sl: return i, sl, "loss", False
            if lo <= tp: return i, tp, "win",  False
            if lo <= half:
                be = ep
                for j in range(i+1, lim):
                    if highs[j] >= be: return j, be, "win", True
                    if lows[j]  <= tp: return j, tp, "win", True
                return -1, None, None, True
    return -1, None, None, False

# ── シミュレーション（AdaptiveRiskManager） ───────────────────────
def simulate(signals, d1m, sym):
    if not signals: return [], INIT_CASH, 0, 0, []

    arm    = AdaptiveRiskManager(sym, base_risk_pct=BASE_RISK)
    m1t    = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; peak = INIT_CASH; mdd = 0.0
    trades = []; equity_curve = [{"time": d1m.index[0], "equity": equity}]
    arm.update_peak(equity)

    for sig in signals:
        lot, _, _ = arm.calc_lot_adaptive(equity, sig["risk"], sig["ep"], USDJPY_RATE)
        sp = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue

        ei, xp, result, half_done = _find_exit(
            m1h[sp:], m1l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
        if result is None: continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = arm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot*0.5, USDJPY_RATE, sig["ep"])
            equity += half_pnl; arm.update_peak(equity); rem = lot * 0.5
        else:
            rem = lot

        pnl    = arm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl; arm.update_peak(equity)
        total  = half_pnl + pnl
        trades.append({"time": sig["time"], "result": result, "pnl": total, "equity": equity})
        equity_curve.append({"time": sig["time"], "equity": equity})
        peak = max(peak, equity); mdd = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, peak, equity_curve

# ── 統計 ─────────────────────────────────────────────────────────
def calc_stats(trades):
    if not trades: return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["pnl"] > 0]["pnl"]; loss = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    gw   = wins.sum(); gl = abs(loss.sum())
    pf   = gw / gl if gl > 0 else float("inf")
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(loss.mean()) if len(loss) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 else 0
    df["month"]  = pd.to_datetime(df["time"]).dt.to_period("M")
    monthly      = df.groupby("month")["pnl"].sum()
    return {"n": n, "wr": wr, "pf": pf, "kelly": kelly,
            "plus_months": (monthly > 0).sum(), "total_months": len(monthly)}

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  v77ロジック再検証（Goldロジック戦力外銘柄 / AdaptiveRisk）")
    print(f"  期間: {START} 〜 {END}  |  初期資産: {INIT_CASH}  |  基本リスク: {BASE_RISK*100:.0f}%")
    print(f"  v77: 4H EMA20のみ（日足フィルターなし） + KMID + KLOW + EMA距離")
    print("="*72)

    all_results = []; all_curves = {}

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        print(f"\n  [{sym}] (Gold PF={sym_info['gold_pf']:.2f}) データ読み込み中...", end=" ", flush=True)

        d1m_full, d15m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("1mデータなし → スキップ"); continue

        d1m  = slice_period(d1m_full,  START, END)
        d15m = slice_period(d15m_full, START, END)
        d4h  = slice_period(d4h_full,  "2024-01-01", END)

        if len(d1m) == 0:
            print("期間内データなし → スキップ"); continue

        cfg    = SYMBOL_CONFIG.get(sym, {})
        spread = cfg.get("spread", 0) * cfg.get("pip", 0.0001)
        atr_1m = calc_atr(d1m, 10).to_dict()
        m1c    = {"idx": d1m.index, "opens": d1m["open"].values,
                  "closes": d1m["close"].values, "highs": d1m["high"].values,
                  "lows":  d1m["low"].values}

        print("シグナル生成中...", end=" ", flush=True)
        sigs = generate_signals_v77(d1m, d15m, d4h, spread, atr_1m, m1c)
        print(f"{len(sigs)}シグナル  シミュレーション中...", end=" ", flush=True)

        trades, final_eq, mdd, _, eq_curve = simulate(sigs, d1m, sym)
        st = calc_stats(trades)

        if st:
            st["mdd"] = mdd; st["final_eq"] = final_eq; st["multiple"] = final_eq / INIT_CASH
            arrow = "↑" if st["pf"] > sym_info["gold_pf"] else "↓"
            diff  = st["pf"] - sym_info["gold_pf"]
            print(f"完了  PF={st['pf']:.2f}({arrow}{diff:+.2f}vs Gold)  WR={st['wr']*100:.1f}%  "
                  f"MDD={mdd:.1f}%  {INIT_CASH:.0f}→{final_eq:.1f}({st['multiple']:.1f}x)")
        else:
            print("トレードなし")

        all_results.append({**sym_info, **st} if st else {**sym_info, "n": 0})
        all_curves[sym] = eq_curve

    # ── 結果テーブル ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("  ■ v77 vs Goldロジック 比較サマリー")
    print(f"  {'銘柄':8} {'N':>5} {'WR':>6} {'v77 PF':>7} {'Gold PF':>8} {'差':>6} "
          f"{'MDD':>6} {'Kelly':>7} {'月+':>5} {'倍率':>7} {'判定':>6}")
    print("-"*72)

    passed_syms = []
    for r in all_results:
        sym = r["name"]
        if r.get("n", 0) == 0:
            print(f"  {sym:8} N/A"); continue
        pf      = r["pf"]
        gpf     = r["gold_pf"]
        diff    = pf - gpf
        pf_s    = f"{pf:.2f}" if pf < 99 else "∞"
        arrow   = "↑" if diff > 0 else "↓"
        passed  = pf >= 2.0 and r["wr"] >= 0.65 and r["mdd"] <= 20.0 and r["kelly"] >= 0.45
        verdict = "✅採用" if passed else ("⚠️惜" if pf >= 1.8 else "❌却下")
        if passed: passed_syms.append(sym)
        print(f"  {sym:8} {r['n']:>5} {r['wr']*100:>5.1f}% {pf_s:>7} {gpf:>8.2f} "
              f"{arrow}{abs(diff):>4.2f} {r['mdd']:>5.1f}% {r['kelly']:>6.3f} "
              f"{r['plus_months']:>2}/{r['total_months']:<2} {r['multiple']:>6.1f}x {verdict}")

    print()
    if passed_syms:
        print(f"  ✅ v77採用候補: {', '.join(passed_syms)}")
    else:
        print("  ❌ v77でも採用基準（PF≥2.0/WR≥65%/MDD≤20%/Kelly≥0.45）を満たす銘柄なし")

    # カテゴリ別avg
    for cat in ["FX", "JPY", "IDX"]:
        pfs = [r["pf"] for r in all_results if r.get("cat") == cat and r.get("pf", 0) < 99 and r.get("n", 0) > 0]
        if pfs:
            print(f"  {cat} avg PF: {np.mean(pfs):.2f}")

    # ── グラフ ───────────────────────────────────────────────────
    valid = [r for r in all_results if r.get("n", 0) > 0]
    if not valid: return

    print("\n  グラフ生成中...")
    n_sym = len(valid)
    cols  = 3; rows = (n_sym + cols - 1) // cols
    fig   = plt.figure(figsize=(18, 4 * rows + 1))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"v77ロジック再検証 — Goldロジック戦力外銘柄\n"
        f"期間: {START} 〜 {END}  |  初期資産: {INIT_CASH}  |  基本リスク: {BASE_RISK*100:.0f}%  "
        f"|  v77: 4H EMA20 + KMID + KLOW（日足フィルターなし）",
        color="white", fontsize=11, y=0.99)

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.6, wspace=0.35)

    for idx, r in enumerate(valid):
        sym  = r["name"]; ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.set_facecolor("#16213e")
        curve    = all_curves.get(sym, [])
        times    = [c["time"] for c in curve]
        equities = [c["equity"] for c in curve]
        color    = CAT_COLORS.get(r["cat"], "#4C9BE8")

        ax.plot(times, equities, color=color, linewidth=1.2)
        ax.axhline(INIT_CASH, color="#555", linewidth=0.8, linestyle="--")
        ax.fill_between(times, INIT_CASH, equities,
                        where=[e >= INIT_CASH for e in equities], alpha=0.18, color=color)
        ax.fill_between(times, INIT_CASH, equities,
                        where=[e < INIT_CASH for e in equities], alpha=0.18, color="#e74c3c")

        pf_s   = f"{r['pf']:.2f}" if r.get("pf", 0) < 99 else "∞"
        gpf    = r["gold_pf"]; diff = r["pf"] - gpf
        passed = (r.get("pf",0)>=2.0 and r.get("wr",0)>=0.65
                  and r.get("mdd",99)<=20.0 and r.get("kelly",0)>=0.45)
        badge  = "✅" if passed else ("⚠️" if r.get("pf",0) >= 1.8 else "❌")
        arrow  = "↑" if diff > 0 else "↓"

        ax.set_title(
            f"{sym} {badge}  v77 PF={pf_s} (Gold:{gpf:.2f} {arrow}{abs(diff):.2f})\n"
            f"WR={r.get('wr',0)*100:.1f}%  MDD={r.get('mdd',0):.1f}%  "
            f"Kelly={r.get('kelly',0):.3f}  {r.get('multiple',1):.1f}x",
            color="white", fontsize=8.5, pad=4)
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.set_ylabel("資産", color="#aaa", fontsize=7)

    for idx in range(n_sym, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "backtest_v77_rejects.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  → {out_path}")

    rows_csv = []
    for r in all_results:
        if r.get("n", 0) == 0: continue
        cfg = SYMBOL_CONFIG.get(r["name"], {})
        passed = (r.get("pf",0)>=2.0 and r.get("wr",0)>=0.65
                  and r.get("mdd",99)<=20.0 and r.get("kelly",0)>=0.45)
        rows_csv.append({"symbol": r["name"], "category": r["cat"],
            "quote_type": cfg.get("quote_type","?"), "spread_pips": cfg.get("spread",0),
            "gold_pf": r["gold_pf"], "v77_pf": round(r["pf"],3) if r["pf"]<99 else 999,
            "pf_diff": round(r["pf"]-r["gold_pf"],3),
            "trades": r["n"], "win_rate": round(r["wr"]*100,1),
            "mdd_pct": round(r["mdd"],2), "kelly": round(r["kelly"],3),
            "plus_months": r["plus_months"], "total_months": r["total_months"],
            "multiple": round(r["multiple"],3), "v77_adopted": passed})
    if rows_csv:
        csv_path = os.path.join(OUT_DIR, "backtest_v77_rejects.csv")
        pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
        print(f"  → {csv_path}")

    print("\n完了")

if __name__ == "__main__":
    main()
