"""
backtest_logic_comparison.py
==============================
ロジック比較バックテスト（1分足エントリー全銘柄）

【対象銘柄】EURUSD / GBPUSD / AUDUSD / NAS100 / SPX500 / US30
【比較ロジック】
  Logic-A (Gold/v79A): 日足EMA20方向一致 + E2エントリー（スパイク除外）
  Logic-B (FX/v79BC):  ADX≥20 + 直近4本同方向(Streak) + E1エントリー（方向一致1m待ち）

【IS/OOS】
  IS:  2025-01-01 〜 2025-05-31（5ヶ月）
  OOS: 2025-06-01 〜 2026-02-28（9ヶ月）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

# ── 設定 ─────────────────────────────────────────────────────────
IS_START  = "2025-01-01"
IS_END    = "2025-05-31"
OOS_START = "2025-06-01"
OOS_END   = "2026-02-28"

INIT_CASH    = 1_000_000
RISK_PCT     = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
USDJPY_RATE  = 150.0
MAX_LOOKAHEAD = 20_000

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E1_MAX_WAIT_MIN = 5
E2_SPIKE_ATR    = 2.0
E2_WINDOW_MIN   = 3

ADX_MIN   = 20
STREAK_MIN = 4

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    {"name": "EURUSD",  "cat": "FX"},
    {"name": "GBPUSD",  "cat": "FX"},
    {"name": "AUDUSD",  "cat": "FX"},
    {"name": "NAS100",  "cat": "IDX"},
    {"name": "SPX500",  "cat": "IDX"},
    {"name": "US30",    "cat": "IDX"},
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

def slice_period(df, start, end):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def load_all(sym):
    def _f(tf):
        p = os.path.join(DATA_DIR, f"{sym}_{tf}.csv")
        return load_csv(p) if os.path.exists(p) else None
    return _f("1m"), _f("15m"), _f("4h")

# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    """ADX計算"""
    high = df["high"]; low = df["low"]; close = df["close"]
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # +DMが大きい場合のみ有効
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0.0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0.0
    tr = calc_atr(df, 1)  # TR
    # Wilder平滑化
    atr_w   = tr.ewm(alpha=1/n, adjust=False).mean()
    pdm_w   = plus_dm.ewm(alpha=1/n, adjust=False).mean()
    mdm_w   = minus_dm.ewm(alpha=1/n, adjust=False).mean()
    di_plus  = 100 * pdm_w / atr_w.replace(0, np.nan)
    di_minus = 100 * mdm_w / atr_w.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx.fillna(0)

def build_4h(df4h, need_1d=False):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    df["adx"]   = calc_adx(df, 14)
    d1 = None
    if need_1d:
        d1 = df.resample("1D").agg({"open":"first","high":"max","low":"min",
                                     "close":"last","volume":"sum"}).dropna(subset=["open","close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df15m):
    df = df15m.resample("1h").agg({"open":"first","high":"max","low":"min",
                                    "close":"last","volume":"sum"}).dropna(subset=["open","close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

# ── エントリー ────────────────────────────────────────────────────
def pick_e1(signal_time, direction, spread, m1c):
    """E1: 方向一致1m足を待って次足始値"""
    idx = m1c["idx"]
    s   = idx.searchsorted(signal_time, side="left")
    e   = idx.searchsorted(signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        o = m1c["opens"][i]; c = m1c["closes"][i]
        if direction == 1 and c <= o: continue
        if direction ==-1 and c >= o: continue
        ni = i + 1
        if ni >= len(idx): return None, None
        return idx[ni], m1c["opens"][ni] + (spread if direction == 1 else -spread)
    return None, None

def pick_e2(signal_time, direction, spread, atr_1m_d, m1c):
    """E2: スパイク除外して即エントリー"""
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

# ── フィルター共通 ────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction ==-1 and bar["close"] < bar["open"])

def check_klow(bar):
    o, l = bar["open"], bar["low"]
    return (min(bar["open"], bar["close"]) - l) / o < KLOW_THR if o > 0 else False

def check_ema_dist(bar):
    d = abs(bar["close"] - bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a > 0 and d >= a * A1_EMA_DIST_MIN

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m_oos, d15m_oos, d4h_full, spread, logic, atr_1m_d, m1c):
    """
    logic: "A" (Gold: 日足EMA20 + E2) or "B" (FX: ADX+Streak + E1)
    """
    need_1d = (logic == "A")
    d4h, d1d = build_4h(d4h_full, need_1d=need_1d)
    d1h = build_1h(d15m_oos)

    signals = []; used = set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct  = h1_times[i]
        h1_p1  = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h <= 0: continue

        h4_before = d4h[d4h.index < h1_ct]
        if len(h4_before) < max(2, STREAK_MIN): continue
        h4_lat = h4_before.iloc[-1]
        if pd.isna(h4_lat.get("atr", np.nan)): continue
        trend = h4_lat["trend"]; h4_atr = h4_lat["atr"]

        # ── Logic別フィルター ──
        if logic == "A":
            # 日足EMA20方向一致
            d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0: continue
            if d1_before.iloc[-1]["trend1d"] != trend: continue
        elif logic == "B":
            # ADX ≥ 20
            if h4_lat.get("adx", 0) < ADX_MIN: continue
            # 直近STREAK_MIN本が同方向
            recent = h4_before["trend"].iloc[-STREAK_MIN:].values
            if not all(t == trend for t in recent): continue

        # ── 共通フィルター ──
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if not check_ema_dist(h4_lat): continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction == 1:  v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:               v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol: continue

        if logic == "A":
            et, ep = pick_e2(h1_ct, direction, spread, atr_1m_d, m1c)
        else:
            et, ep = pick_e1(h1_ct, direction, spread, m1c)

        if et is None or et in used: continue

        raw = ep - spread if direction == 1 else ep + spread
        if direction == 1: sl = min(v1, v2) - atr_1h * 0.15; risk = raw - sl
        else:              sl = max(v1, v2) + atr_1h * 0.15; risk = sl - raw
        if 0 < risk <= h4_atr * 2:
            tp = raw + direction * risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk, "signal_time": h1_ct})
            used.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals

# ── シミュレーション ──────────────────────────────────────────────
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

def simulate(signals, d1m, sym):
    if not signals: return [], INIT_CASH, 0, 0
    rm  = RiskManager(sym, risk_pct=RISK_PCT)
    m1t = d1m.index; m1h = d1m["high"].values; m1l = d1m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m1t.searchsorted(sig["time"], side="right")
        if sp >= len(m1t): continue
        ei, xp, result, half_done = _find_exit(m1h[sp:], m1l[sp:],
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
        trades.append({"result": result, "pnl": total})

        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, peak

def calc_stats(trades, init=INIT_CASH):
    if not trades: return {}
    df  = pd.DataFrame(trades)
    n   = len(df)
    wins  = df[df["pnl"] > 0]["pnl"]
    loses = df[df["pnl"] < 0]["pnl"]
    wr  = len(wins) / n
    gw  = wins.sum(); gl = abs(loses.sum())
    pf  = gw / gl if gl > 0 else float("inf")
    total = df["pnl"].sum()
    return {"n": n, "wr": wr, "pf": pf, "total": total}

# ── メイン ───────────────────────────────────────────────────────
def run_period(d1m_full, d15m_full, d4h_full, sym, logic, start, end):
    d1m  = slice_period(d1m_full,  start, end)
    d15m = slice_period(d15m_full, start, end)
    if d1m is None or len(d1m) == 0: return {}

    cfg    = SYMBOL_CONFIG[sym]
    spread = cfg["spread"] * cfg["pip"]
    atr_1m = calc_atr(d1m, 10).to_dict()
    m1c    = {"idx":   d1m.index,
              "opens": d1m["open"].values,
              "closes":d1m["close"].values,
              "highs": d1m["high"].values,
              "lows":  d1m["low"].values}

    sigs  = generate_signals(d1m, d15m, d4h_full, spread, logic, atr_1m, m1c)
    trades, final_eq, mdd, _ = simulate(sigs, d1m, sym)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd; st["final_eq"] = final_eq
    return st

def main():
    print("\n" + "="*80)
    print("  ロジック比較バックテスト（1分足エントリー）")
    print(f"  IS:  {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
    print("="*80)

    header = f"{'銘柄':8} {'ロジック':6} | {'IS':^40} | {'OOS':^40}"
    subhdr = f"{'':16} | {'n':>5} {'WR':>6} {'PF':>6} {'MDD':>6} {'倍率':>6} | {'n':>5} {'WR':>6} {'PF':>6} {'MDD':>6} {'倍率':>6}"
    print(header)
    print(subhdr)
    print("-"*80)

    all_results = []

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        d1m_full, d15m_full, d4h_full = load_all(sym)
        if d1m_full is None or d15m_full is None or d4h_full is None:
            print(f"{sym}: データ不足 → スキップ")
            continue

        for logic in ["A", "B"]:
            label = f"Logic-{'A(日足EMA+E2)' if logic=='A' else 'B(ADX+Str+E1)'}"
            print(f"  {sym} {label} 計算中...", end=" ", flush=True)

            is_st  = run_period(d1m_full, d15m_full, d4h_full, sym, logic, IS_START,  IS_END)
            oos_st = run_period(d1m_full, d15m_full, d4h_full, sym, logic, OOS_START, OOS_END)
            print("完了")

            def fmt(st):
                if not st: return f"{'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6}"
                pf_s  = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
                mul   = st['final_eq'] / INIT_CASH
                return (f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} "
                        f"{st['mdd']:>5.1f}% {mul:>5.2f}x")

            logic_label = "A:日足EMA" if logic == "A" else "B:ADX+Str"
            row = f"{sym:8} {logic_label:9} | {fmt(is_st)} | {fmt(oos_st)}"
            print(row)

            all_results.append({
                "sym": sym, "cat": sym_info["cat"], "logic": logic,
                "is": is_st, "oos": oos_st
            })

        print()

    # ── サマリー ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  ■ OOS PF 比較サマリー（勝者太字相当）")
    print(f"  {'銘柄':8} | {'Logic-A PF':>12} {'Logic-B PF':>12} {'推奨':>8}")
    print("-"*80)
    for i in range(0, len(all_results), 2):
        r_a = all_results[i]; r_b = all_results[i+1]
        pf_a = r_a["oos"].get("pf", 0); pf_b = r_b["oos"].get("pf", 0)
        winner = "Logic-A" if pf_a > pf_b else "Logic-B"
        sym = r_a["sym"]
        print(f"  {sym:8} | {pf_a:>12.2f} {pf_b:>12.2f} {winner:>8}")

    print("\n  ■ カテゴリ別 avg OOS PF")
    for cat in ["FX", "IDX"]:
        cat_res = [r for r in all_results if r["cat"] == cat]
        for logic in ["A", "B"]:
            pfs = [r["oos"].get("pf", 0) for r in cat_res if r["logic"] == logic and r["oos"]]
            pfs_fin = [p for p in pfs if p < 99]
            avg = np.mean(pfs_fin) if pfs_fin else 0
            print(f"    {cat} Logic-{logic}: avg PF={avg:.2f}  ({len(pfs_fin)}銘柄)")

if __name__ == "__main__":
    main()
