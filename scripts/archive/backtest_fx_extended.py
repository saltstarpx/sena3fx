"""
backtest_fx_extended.py
==============================
FX拡張バックテスト（EURJPY / GBPJPY / NZDUSD / USDCAD / USDCHF / XAGUSD）

【データ】
  - EURJPY / GBPJPY: data/eurjpy_15m.csv + data/eurjpy_4h.csv（全期間連続）
  - NZDUSD / USDCAD / USDCHF / XAGUSD: data/{sym}_is_15m + _oos_15m を結合

【エントリー】
  15mを1m代用（1m足なし銘柄）
  E2方式: スパイク除外して最初の15m足で入場

【IS/OOS】
  IS:  2025-01-01 〜 2025-05-31
  OOS: 2025-06-01 〜 2026-02-28
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

IS_START  = "2025-01-01"
IS_END    = "2025-05-31"
OOS_START = "2025-06-01"
OOS_END   = "2026-02-28"

INIT_CASH    = 1_000_000
RISK_PCT     = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
USDJPY_RATE  = 150.0
MAX_LOOKAHEAD = 5_000   # 15m足なので1m版より小さくてよい

KLOW_THR        = 0.0015
A1_EMA_DIST_MIN = 1.0
A3_DEFAULT_TOL  = 0.30
E2_SPIKE_ATR    = 2.0
E2_WINDOW_BARS  = 3     # 15m足で最大3本待つ

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OHLC_DIR  = os.path.join(DATA_DIR, "ohlc")
OUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 銘柄定義: (name, cat, data_type)
# data_type: "full"=全期間1ファイル, "split"=IS/OOS分割
SYMBOLS = [
    {"name": "EURJPY",  "cat": "JPY_CROSS", "data": "full"},
    {"name": "GBPJPY",  "cat": "JPY_CROSS", "data": "full"},
    {"name": "NZDUSD",  "cat": "FX",        "data": "split"},
    {"name": "USDCAD",  "cat": "FX_INV",    "data": "split"},
    {"name": "USDCHF",  "cat": "FX_INV",    "data": "split"},
    {"name": "XAGUSD",  "cat": "METALS",    "data": "split"},
]

# ── データロード ──────────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    tc = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.rename(columns={tc: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])

def load_all(sym_info):
    sym  = sym_info["name"].lower()
    name = sym_info["name"]

    if sym_info["data"] == "full":
        # data/eurjpy_15m.csv, data/eurjpy_4h.csv
        d15m = load_csv(os.path.join(DATA_DIR, f"{sym}_15m.csv"))
        d4h  = load_csv(os.path.join(DATA_DIR, f"{sym}_4h.csv"))
        d1d  = None  # 日足は後でresampleで生成
    else:
        # IS+OOS を結合
        is15  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_15m.csv"))
        oos15 = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_15m.csv"))
        is4h  = load_csv(os.path.join(DATA_DIR, f"{sym}_is_4h.csv"))
        oos4h = load_csv(os.path.join(DATA_DIR, f"{sym}_oos_4h.csv"))
        d15m = pd.concat([is15, oos15]).sort_index() if (is15 is not None and oos15 is not None) else None
        d4h  = pd.concat([is4h,  oos4h]).sort_index() if (is4h  is not None and oos4h  is not None) else None
        if d15m is not None:
            d15m = d15m[~d15m.index.duplicated(keep="first")]
        if d4h is not None:
            d4h = d4h[~d4h.index.duplicated(keep="first")]
        d1d = None

    return d15m, d4h

def slice_period(df, start, end):
    if df is None or len(df) == 0:
        return None
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
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    # 日足: 4hからresample
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

# ── エントリー（15m版E2）────────────────────────────────────────
def pick_e2_15m(signal_time, direction, spread, atr_15m_d, m15c):
    """E2: 15m足でスパイク除外して即エントリー"""
    idx = m15c["idx"]
    s   = idx.searchsorted(signal_time, side="left")
    e   = min(s + E2_WINDOW_BARS, len(idx))
    for i in range(s, e):
        bar_range = m15c["highs"][i] - m15c["lows"][i]
        atr_val   = atr_15m_d.get(idx[i], np.nan)
        if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR:
            continue
        return idx[i], m15c["opens"][i] + (spread if direction == 1 else -spread)
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

# ── シグナル生成（Goldロジック = Logic-A）──────────────────────
def generate_signals(d15m_oos, d4h_full, spread, atr_15m_d, m15c):
    d4h, d1d = build_4h(d4h_full)
    d1h = build_1h(d15m_oos)

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
        trend = h4_lat["trend"]; h4_atr = h4_lat["atr"]

        # Goldロジック: 日足EMA20方向一致
        d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
        if len(d1_before) == 0: continue
        if d1_before.iloc[-1]["trend1d"] != trend: continue

        # 共通フィルター
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if not check_ema_dist(h4_lat): continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction == 1:  v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:               v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol: continue

        et, ep = pick_e2_15m(h1_ct, direction, spread, atr_15m_d, m15c)
        if et is None or et in used: continue

        raw = ep - spread if direction == 1 else ep + spread
        if direction == 1: sl = min(v1, v2) - atr_1h * 0.15; risk = raw - sl
        else:              sl = max(v1, v2) + atr_1h * 0.15; risk = sl - raw
        if 0 < risk <= h4_atr * 2:
            tp = raw + direction * risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk})
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

def simulate(signals, d15m, sym):
    if not signals: return [], INIT_CASH, 0, 0
    rm  = RiskManager(sym, risk_pct=RISK_PCT)
    m15t = d15m.index; m15h = d15m["high"].values; m15l = d15m["low"].values
    equity = INIT_CASH; trades = []; peak = INIT_CASH; mdd = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        sp  = m15t.searchsorted(sig["time"], side="right")
        if sp >= len(m15t): continue
        ei, xp, result, half_done = _find_exit(m15h[sp:], m15l[sp:],
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

# ── 1期間ラン ────────────────────────────────────────────────────
def run_period(d15m_full, d4h_full, sym, start, end):
    d15m = slice_period(d15m_full, start, end)
    if d15m is None or len(d15m) == 0: return {}

    cfg    = SYMBOL_CONFIG[sym]
    spread = cfg["spread"] * cfg["pip"]
    atr_15m = calc_atr(d15m, 10).to_dict()
    m15c    = {"idx":   d15m.index,
               "opens": d15m["open"].values,
               "closes":d15m["close"].values,
               "highs": d15m["high"].values,
               "lows":  d15m["low"].values}

    sigs  = generate_signals(d15m, d4h_full, spread, atr_15m, m15c)
    trades, final_eq, mdd, _ = simulate(sigs, d15m, sym)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd; st["final_eq"] = final_eq
    return st

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("  FX拡張バックテスト（Goldロジック / 15mエントリー）")
    print(f"  IS:  {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
    print("="*80)

    header = f"{'銘柄':8} {'カテゴリ':10} | {'IS':^38} | {'OOS':^38}"
    subhdr = f"{'':20} | {'n':>5} {'WR':>6} {'PF':>6} {'MDD':>6} {'倍率':>6} | {'n':>5} {'WR':>6} {'PF':>6} {'MDD':>6} {'倍率':>6}"
    print(header)
    print(subhdr)
    print("-"*80)

    all_results = []

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        d15m_full, d4h_full = load_all(sym_info)
        if d15m_full is None or d4h_full is None:
            print(f"{sym}: データ不足 → スキップ")
            continue

        print(f"  {sym} 計算中...", end=" ", flush=True)
        is_st  = run_period(d15m_full, d4h_full, sym, IS_START,  IS_END)
        oos_st = run_period(d15m_full, d4h_full, sym, OOS_START, OOS_END)
        print("完了")

        def fmt(st):
            if not st: return f"{'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6}"
            pf_s  = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            mul   = st['final_eq'] / INIT_CASH
            return (f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} "
                    f"{st['mdd']:>5.1f}% {mul:>5.2f}x")

        row = f"{sym:8} {sym_info['cat']:10} | {fmt(is_st)} | {fmt(oos_st)}"
        print(row)

        all_results.append({
            "sym": sym, "cat": sym_info["cat"],
            "is": is_st, "oos": oos_st
        })

    # ── サマリー ──────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  ■ OOS PF サマリー（Goldロジック）")
    print(f"  {'銘柄':8} {'カテゴリ':10} | {'OOS PF':>8} {'OOS WR':>8} {'OOS MDD':>8} {'判定':>6}")
    print("-"*80)
    for r in all_results:
        oos = r["oos"]
        if not oos:
            print(f"  {r['sym']:8} {r['cat']:10} | {'N/A':>8}")
            continue
        pf_s = f"{oos['pf']:.2f}" if oos['pf'] < 99 else "∞"
        judge = "✅" if oos["pf"] >= 1.5 else ("△" if oos["pf"] >= 1.2 else "❌")
        print(f"  {r['sym']:8} {r['cat']:10} | {pf_s:>8} {oos['wr']*100:>7.1f}% {oos['mdd']:>7.1f}% {judge:>6}")

    # CSV保存
    rows = []
    for r in all_results:
        for period, st in [("IS", r["is"]), ("OOS", r["oos"])]:
            if st:
                rows.append({"sym": r["sym"], "cat": r["cat"], "period": period,
                             "n": st["n"], "wr": round(st["wr"],4),
                             "pf": round(st["pf"],3), "mdd": round(st["mdd"],2),
                             "final_eq": round(st["final_eq"],0)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fx_extended_results.csv"), index=False)
    print(f"\n結果を results/fx_extended_results.csv に保存しました")

if __name__ == "__main__":
    main()
