"""
backtest_all_1m_adaptive.py
============================
1分足データが揃っている全銘柄バックテスト（Goldロジック統一）
- ロジック: Logic-A（日足EMA20方向一致 + E2エントリー）
- リスク管理: AdaptiveRiskManager（DD連動動的リスク）
- 初期資産: 100（JPY建て換算、比率追跡）
- 期間: 2025-01-01 〜 2026-02-28（全期間）
- 対象: AUDUSD / EURUSD / GBPUSD / NAS100 / SPX500 / US30 / XAUUSD
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import AdaptiveRiskManager, SYMBOL_CONFIG

# ── 設定 ─────────────────────────────────────────────────────────
START        = "2025-01-01"
END          = "2026-02-28"
INIT_CASH    = 100.0          # 初期資産（比率追跡用）
BASE_RISK    = 0.02           # 基本リスク2%
RR_RATIO     = 2.5
HALF_R       = 1.0
USDJPY_RATE  = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    {"name": "AUDUSD", "cat": "FX"},
    {"name": "EURUSD", "cat": "FX"},
    {"name": "GBPUSD", "cat": "FX"},
    {"name": "XAUUSD", "cat": "METALS"},
    {"name": "NAS100", "cat": "IDX"},
    {"name": "SPX500", "cat": "IDX"},
    {"name": "US30",   "cat": "IDX"},
]

CAT_COLORS = {"FX": "#4C9BE8", "METALS": "#F5A623", "IDX": "#7ED321"}

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    df = pd.read_csv(path)
    # カラム名を小文字に統一（1dファイルはOpen/High/Low/Close形式）
    df.columns = [c.lower() for c in df.columns]
    # タイムスタンプカラム検出（timestamp or datetime）
    tc = "timestamp" if "timestamp" in df.columns else "datetime" if "datetime" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def load_all(sym):
    def _f(tf):
        p = os.path.join(DATA_DIR, f"{sym}_{tf}.csv")
        return load_csv(p) if os.path.exists(p) else None
    d1m  = _f("1m")
    d15m = _f("15m")
    d4h  = _f("4h")
    d1d  = _f("1d")
    return d1m, d15m, d4h, d1d

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

def build_4h_and_1d(df4h, df1d=None):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)

    # 日足: 提供データがあれば使用、なければ4hからリサンプル
    if df1d is not None and len(df1d) > 0:
        d1 = df1d.copy()
    else:
        d1 = df4h.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna(subset=["open", "close"])

    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df15m):
    df = df15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

# ── エントリー（E2: スパイク除外） ───────────────────────────────
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

# ── フィルター ────────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction ==-1 and bar["close"] < bar["open"])

def check_klow(bar):
    o, l = bar["open"], bar["low"]
    return (min(bar["open"], bar["close"]) - l) / o < KLOW_THR if o > 0 else False

def check_ema_dist(bar):
    d = abs(bar["close"] - bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a > 0 and d >= a * A1_EMA_DIST_MIN

# ── シグナル生成（Gold Logic-A: 日足EMA20 + E2） ─────────────────
def generate_signals(d1m, d15m, d4h_full, d1d_full, spread, atr_1m_d, m1c):
    d4h, d1d = build_4h_and_1d(d4h_full, d1d_full)
    d1h      = build_1h(d15m)

    signals = []; used = set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct  = h1_times[i]
        h1_p1  = d1h.iloc[i-1]
        h1_p2  = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h <= 0:
            continue

        h4_before = d4h[d4h.index < h1_ct]
        if len(h4_before) < 2:
            continue
        h4_lat = h4_before.iloc[-1]
        if pd.isna(h4_lat.get("atr", np.nan)):
            continue
        trend  = h4_lat["trend"]
        h4_atr = h4_lat["atr"]

        # 日足EMA20方向一致（Gold Logic）
        d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
        if len(d1_before) == 0:
            continue
        if d1_before.iloc[-1]["trend1d"] != trend:
            continue

        # 共通フィルター
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat):        continue
        if not check_ema_dist(h4_lat):    continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction == 1:
            v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:
            v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol:
            continue

        et, ep = pick_e2(h1_ct, direction, spread, atr_1m_d, m1c)
        if et is None or et in used:
            continue

        raw = ep - spread if direction == 1 else ep + spread
        if direction == 1:
            sl   = min(v1, v2) - atr_1h * 0.15
            risk = raw - sl
        else:
            sl   = max(v1, v2) + atr_1h * 0.15
            risk = sl - raw

        if 0 < risk <= h4_atr * 2:
            tp = raw + direction * risk * RR_RATIO
            signals.append({
                "time": et, "dir": direction, "ep": ep,
                "sl": sl, "tp": tp, "risk": risk
            })
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
    if not signals:
        return [], INIT_CASH, 0, 0, []

    arm    = AdaptiveRiskManager(sym, base_risk_pct=BASE_RISK)
    m1t    = d1m.index
    m1h    = d1m["high"].values
    m1l    = d1m["low"].values
    equity = INIT_CASH
    peak   = INIT_CASH
    mdd    = 0.0
    trades = []
    equity_curve = [{"time": d1m.index[0], "equity": equity}]

    arm.update_peak(equity)

    for sig in signals:
        lot, eff_risk, _ = arm.calc_lot_adaptive(
            equity      = equity,
            sl_distance = sig["risk"],
            ref_price   = sig["ep"],
            usdjpy_rate = USDJPY_RATE,
        )
        sp = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t):
            continue

        ei, xp, result, half_done = _find_exit(
            m1h[sp:], m1l[sp:],
            sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"]
        )
        if result is None:
            continue

        half_pnl = 0.0
        if half_done:
            hp = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = arm.calc_pnl_jpy(
                sig["dir"], sig["ep"], hp, lot * 0.5, USDJPY_RATE, sig["ep"]
            )
            equity += half_pnl
            arm.update_peak(equity)
            rem = lot * 0.5
        else:
            rem = lot

        pnl    = arm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        arm.update_peak(equity)
        total  = half_pnl + pnl
        trades.append({"time": sig["time"], "result": result, "pnl": total, "equity": equity})
        equity_curve.append({"time": sig["time"], "equity": equity})

        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, peak, equity_curve

# ── 統計計算 ─────────────────────────────────────────────────────
def calc_stats(trades):
    if not trades:
        return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    loss = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    gw   = wins.sum(); gl = abs(loss.sum())
    pf   = gw / gl if gl > 0 else float("inf")

    # ケリー基準
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(loss.mean()) if len(loss) > 0 else 1
    kelly = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 else 0

    # 月次プラス
    df["month"] = pd.to_datetime(df["time"]).dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    total_months = len(monthly)

    return {
        "n": n, "wr": wr, "pf": pf,
        "kelly": kelly,
        "plus_months": plus_months,
        "total_months": total_months,
        "total_pnl": df["pnl"].sum(),
    }

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*72)
    print("  YAGAMI改 全銘柄バックテスト（Goldロジック / AdaptiveRisk）")
    print(f"  期間: {START} 〜 {END}  |  初期資産: {INIT_CASH}  |  基本リスク: {BASE_RISK*100:.0f}%")
    print("="*72)

    all_results = []
    all_curves  = {}

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        print(f"\n  [{sym}] データ読み込み中...", end=" ", flush=True)

        d1m_full, d15m_full, d4h_full, d1d_full = load_all(sym)

        # 1m/15m/4hは必須
        if d1m_full is None or d15m_full is None or d4h_full is None:
            print(f"データ不足 → スキップ")
            continue

        # 期間スライス
        d1m  = slice_period(d1m_full,  START, END)
        d15m = slice_period(d15m_full, START, END)
        d4h  = slice_period(d4h_full,  "2024-01-01", END)  # EMA計算用に広め
        d1d  = slice_period(d1d_full,  "2024-01-01", END) if d1d_full is not None else None

        if len(d1m) == 0:
            print("期間内データなし → スキップ")
            continue

        cfg    = SYMBOL_CONFIG.get(sym, {})
        spread = cfg.get("spread", 0) * cfg.get("pip", 0.0001)
        atr_1m = calc_atr(d1m, 10).to_dict()
        m1c    = {
            "idx":    d1m.index,
            "opens":  d1m["open"].values,
            "closes": d1m["close"].values,
            "highs":  d1m["high"].values,
            "lows":   d1m["low"].values,
        }

        print("シグナル生成中...", end=" ", flush=True)
        sigs = generate_signals(d1m, d15m, d4h, d1d, spread, atr_1m, m1c)
        print(f"{len(sigs)}シグナル  シミュレーション中...", end=" ", flush=True)

        trades, final_eq, mdd, peak, eq_curve = simulate(sigs, d1m, sym)
        st = calc_stats(trades)

        if st:
            st["mdd"]      = mdd
            st["final_eq"] = final_eq
            st["multiple"] = final_eq / INIT_CASH
            print(f"完了  PF={st['pf']:.2f}  WR={st['wr']*100:.1f}%  "
                  f"MDD={mdd:.1f}%  {INIT_CASH:.0f}→{final_eq:.1f}({st['multiple']:.1f}x)")
        else:
            print("トレードなし")

        all_results.append({**sym_info, **st} if st else {**sym_info, "n": 0})
        all_curves[sym] = eq_curve

    # ── 結果テーブル出力 ─────────────────────────────────────────
    print("\n" + "="*72)
    print("  ■ バックテスト結果サマリー")
    print(f"  {'銘柄':8} {'Cat':6} {'N':>5} {'WR':>6} {'PF':>6} {'MDD':>6} "
          f"{'Kelly':>7} {'月+':>5} {'倍率':>7}")
    print("-"*72)

    for r in all_results:
        sym = r["name"]
        if r.get("n", 0) == 0:
            print(f"  {sym:8} {r['cat']:6} {'N/A':>5}")
            continue
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 99 else "∞"
        print(
            f"  {sym:8} {r['cat']:6} {r['n']:>5} "
            f"{r['wr']*100:>5.1f}% {pf_s:>6} {r['mdd']:>5.1f}% "
            f"{r['kelly']:>6.3f} "
            f"{r['plus_months']:>2}/{r['total_months']:<2} "
            f"{r['multiple']:>6.1f}x"
        )

    # カテゴリ別avg PF
    print()
    for cat in ["FX", "METALS", "IDX"]:
        cat_pf = [r["pf"] for r in all_results if r.get("cat") == cat and r.get("pf", 0) < 99 and r.get("n", 0) > 0]
        if cat_pf:
            print(f"  {cat} avg PF: {np.mean(cat_pf):.2f}  ({len(cat_pf)}銘柄)")

    # ── エクイティカーブ描画 ─────────────────────────────────────
    print("\n  グラフ生成中...")

    valid_syms = [r["name"] for r in all_results if r.get("n", 0) > 0]
    n_sym = len(valid_syms)
    if n_sym == 0:
        print("  グラフ描画スキップ（トレードなし）")
        return

    fig = plt.figure(figsize=(18, 4 + 3 * ((n_sym + 2) // 3)))
    fig.patch.set_facecolor("#1a1a2e")

    # タイトル
    fig.suptitle(
        f"YAGAMI改 全銘柄バックテスト — Goldロジック / AdaptiveRisk\n"
        f"期間: {START} 〜 {END}  |  初期資産: {INIT_CASH}  |  基本リスク: {BASE_RISK*100:.0f}%",
        color="white", fontsize=13, y=0.98
    )

    cols = 3
    rows = (n_sym + cols - 1) // cols
    gs   = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

    for idx, sym in enumerate(valid_syms):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.set_facecolor("#16213e")

        curve = all_curves.get(sym, [])
        if not curve:
            continue
        times  = [c["time"] for c in curve]
        equities = [c["equity"] for c in curve]

        r = next((x for x in all_results if x["name"] == sym), {})
        cat = r.get("cat", "FX")
        color = CAT_COLORS.get(cat, "#4C9BE8")

        ax.plot(times, equities, color=color, linewidth=1.2)
        ax.axhline(INIT_CASH, color="#555", linewidth=0.8, linestyle="--")
        ax.fill_between(times, INIT_CASH, equities,
                        where=[e >= INIT_CASH for e in equities],
                        alpha=0.15, color=color)
        ax.fill_between(times, INIT_CASH, equities,
                        where=[e < INIT_CASH for e in equities],
                        alpha=0.15, color="#e74c3c")

        pf_s = f"{r['pf']:.2f}" if r.get('pf', 0) < 99 else "∞"
        ax.set_title(
            f"{sym} [{cat}]  PF={pf_s}  WR={r.get('wr',0)*100:.1f}%  "
            f"MDD={r.get('mdd',0):.1f}%  {r.get('multiple',1):.1f}x",
            color="white", fontsize=8.5, pad=4
        )
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333")
        ax.set_ylabel("資産", color="#aaa", fontsize=7)

    # 空きサブプロットを非表示
    for idx in range(n_sym, rows * cols):
        fig.add_subplot(gs[idx // cols, idx % cols]).set_visible(False)

    out_path = os.path.join(OUT_DIR, "backtest_all_1m_adaptive.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  → {out_path}")

    # CSV保存
    rows_csv = []
    for r in all_results:
        if r.get("n", 0) == 0:
            continue
        rows_csv.append({
            "symbol":       r["name"],
            "category":     r["cat"],
            "trades":       r["n"],
            "win_rate":     round(r["wr"] * 100, 1),
            "profit_factor": round(r["pf"], 3) if r["pf"] < 99 else 999,
            "mdd_pct":      round(r["mdd"], 2),
            "kelly":        round(r["kelly"], 3),
            "plus_months":  r["plus_months"],
            "total_months": r["total_months"],
            "multiple":     round(r["multiple"], 3),
            "final_equity": round(r["final_eq"], 2),
        })
    if rows_csv:
        csv_path = os.path.join(OUT_DIR, "backtest_all_1m_adaptive.csv")
        pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
        print(f"  → {csv_path}")

    print("\n完了")

if __name__ == "__main__":
    main()
