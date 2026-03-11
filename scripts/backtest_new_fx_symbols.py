"""
backtest_new_fx_symbols.py
==========================
未テスト銘柄のロジック比較バックテスト

【対象銘柄】NZDUSD / USDCAD / USDCHF / EURJPY
【比較ロジック】
  Logic-A (Goldロジック): 日足EMA20方向一致 + E2エントリー（スパイク除外）
  Logic-B (ADX+Streak):   ADX≥20 + 直近4本同方向 + E1エントリー（方向一致1m待ち）

【IS/OOS】
  IS:  2025-01-01 〜 2025-05-31（5ヶ月）
  OOS: 2025-06-01 〜 2026-02-28（9ヶ月）

【データ構成】
  1m  : data/ohlc/{SYM}_1m.csv          （全期間 2025/1〜2026/2）
  4h  : data/{sym}_is_4h.csv + oos_4h   （IS: 2024/7〜 / OOS: 2025/3〜）
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

ADX_MIN    = 20
STREAK_MIN = 4

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    {"name": "NZDUSD", "cat": "FX"},
    {"name": "USDCAD", "cat": "FX"},
    {"name": "USDCHF", "cat": "FX"},
    {"name": "EURJPY", "cat": "JPY"},
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
    """
    1m: data/ohlc/{SYM}_1m.csv (全期間)
    4h: data/{sym}_is_4h.csv + data/{sym}_oos_4h.csv (concat)
    """
    sym_lower = sym.lower()

    # 1m
    p1m = os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv")
    if not os.path.exists(p1m):
        print(f"  {sym}: 1mデータ未発見 ({p1m})")
        return None, None, None
    d1m = load_csv(p1m)

    # 4h (IS + OOS を concat して全期間として使用)
    p4h_is  = os.path.join(DATA_DIR, f"{sym_lower}_is_4h.csv")
    p4h_oos = os.path.join(DATA_DIR, f"{sym_lower}_oos_4h.csv")
    p4h_all = os.path.join(DATA_DIR, f"{sym_lower}_4h.csv")  # 全期間ファイルがあれば

    if os.path.exists(p4h_all):
        d4h = load_csv(p4h_all)
    elif os.path.exists(p4h_is) and os.path.exists(p4h_oos):
        d4h = pd.concat([load_csv(p4h_is), load_csv(p4h_oos)])
        d4h = d4h[~d4h.index.duplicated(keep="first")].sort_index()
    else:
        # フォールバック: 1mから4hにリサンプル
        print(f"  {sym}: 4hファイル未発見 → 1mからリサンプル")
        d4h = d1m.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])

    return d1m, d1m, d4h  # d15m の代わりに d1m を渡す（build_1h が 1h にリサンプル）

# ── インジケーター ────────────────────────────────────────────────
def calc_atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()

def calc_adx(df, n=14):
    high = df["high"]; low = df["low"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask  = plus_dm < minus_dm;  plus_dm[mask]  = 0.0
    mask2 = minus_dm < plus_dm;  minus_dm[mask2] = 0.0
    tr      = calc_atr(df, 1)
    atr_w   = tr.ewm(alpha=1/n, adjust=False).mean()
    pdm_w   = plus_dm.ewm(alpha=1/n, adjust=False).mean()
    mdm_w   = minus_dm.ewm(alpha=1/n, adjust=False).mean()
    di_p    = 100 * pdm_w / atr_w.replace(0, np.nan)
    di_m    = 100 * mdm_w / atr_w.replace(0, np.nan)
    dx      = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
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
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])
        d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
        d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1

def build_1h(df_in):
    """入力が 1m でも 15m でも 1h にリサンプルして返す"""
    df = df_in.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

# ── エントリー ────────────────────────────────────────────────────
def pick_e1(signal_time, direction, spread, m1c):
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

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m_oos, d1m_for_1h, d4h_full, spread, logic, atr_1m_d, m1c):
    need_1d = (logic == "A")
    d4h, d1d = build_4h(d4h_full, need_1d=need_1d)
    d1h = build_1h(d1m_for_1h)  # 1m → 1h リサンプル

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
            if d1d is None: continue
            d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
            if len(d1_before) == 0: continue
            if d1_before.iloc[-1]["trend1d"] != trend: continue
        elif logic == "B":
            if h4_lat.get("adx", 0) < ADX_MIN: continue
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
        trades.append({"result": result, "pnl": total,
                       "month": sig["time"].strftime("%Y-%m")})

        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, peak

def calc_stats(trades, init=INIT_CASH):
    if not trades: return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins  = df[df["pnl"] > 0]["pnl"]
    loses = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    gw   = wins.sum(); gl = abs(loses.sum())
    pf   = gw / gl if gl > 0 else float("inf")
    total = df["pnl"].sum()
    # 月次プラス
    monthly = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    return {"n": n, "wr": wr, "pf": pf, "total": total,
            "plus_months": plus_months, "total_months": len(monthly)}

# ── 期間別実行 ────────────────────────────────────────────────────
def run_period(d1m_full, d4h_full, sym, logic, start, end):
    d1m = slice_period(d1m_full, start, end)
    if len(d1m) == 0: return {}

    cfg    = SYMBOL_CONFIG[sym]
    spread = cfg["spread"] * cfg["pip"]
    atr_1m = calc_atr(d1m, 10).to_dict()
    m1c    = {"idx":    d1m.index,
              "opens":  d1m["open"].values,
              "closes": d1m["close"].values,
              "highs":  d1m["high"].values,
              "lows":   d1m["low"].values}

    sigs  = generate_signals(d1m, d1m, d4h_full, spread, logic, atr_1m, m1c)
    trades, final_eq, mdd, _ = simulate(sigs, d1m, sym)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd; st["final_eq"] = final_eq
    return st

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*90)
    print("  新規銘柄ロジック比較バックテスト（NZDUSD / USDCAD / USDCHF / EURJPY）")
    print(f"  IS: {IS_START} 〜 {IS_END}  /  OOS: {OOS_START} 〜 {OOS_END}")
    print(f"  ロジック: A=Goldロジック(日足EMA+E2)  B=ADX+Streak+E1")
    print("="*90)

    print(f"\n  {'銘柄':8} {'ロジック':14} | "
          f"{'IS_n':>5} {'IS_WR':>6} {'IS_PF':>6} {'IS_MDD':>7} | "
          f"{'OOS_n':>5} {'OOS_WR':>6} {'OOS_PF':>6} {'OOS_MDD':>7} {'月+':>5} {'倍率':>6}")
    print("-"*90)

    all_results = []

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        print(f"\n  {sym} データロード中...", end=" ", flush=True)
        d1m_full, _, d4h_full = load_all(sym)
        if d1m_full is None:
            print("スキップ")
            continue
        print(f"1m:{len(d1m_full):,}行 / 4h:{len(d4h_full):,}行")

        for logic in ["A", "B"]:
            label = "A:Goldロジック" if logic == "A" else "B:ADX+Streak "
            print(f"    {sym} Logic-{logic} 計算中...", end=" ", flush=True)

            is_st  = run_period(d1m_full, d4h_full, sym, logic, IS_START,  IS_END)
            oos_st = run_period(d1m_full, d4h_full, sym, logic, OOS_START, OOS_END)
            print("完了")

            def fmt(st):
                if not st:
                    return f"{'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>7}"
                pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
                return (f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['mdd']:>6.1f}%")

            def fmt_oos(st):
                if not st:
                    return f"{'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'N/A':>5} {'N/A':>6}"
                pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
                mul  = st["final_eq"] / INIT_CASH
                pm   = f"{st['plus_months']}/{st['total_months']}"
                return (f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} {st['mdd']:>6.1f}% {pm:>5} {mul:>5.2f}x")

            print(f"  {sym:8} Logic-{label} | {fmt(is_st)} | {fmt_oos(oos_st)}")

            all_results.append({
                "sym": sym, "cat": sym_info["cat"], "logic": logic,
                "is": is_st, "oos": oos_st
            })

    # ── サマリー ──────────────────────────────────────────────────
    print("\n" + "="*90)
    print("  ■ OOS PF 比較サマリー")
    print(f"  {'銘柄':8} | {'Logic-A PF':>11} {'Logic-B PF':>11} | {'スプレッド':>6} {'推奨':>8} {'採用?':>6}")
    print("-"*60)

    for i in range(0, len(all_results), 2):
        r_a = all_results[i]; r_b = all_results[i+1]
        pf_a = r_a["oos"].get("pf", 0)
        pf_b = r_b["oos"].get("pf", 0)
        winner = "Logic-A" if pf_a >= pf_b else "Logic-B"
        best_pf = max(pf_a, pf_b)
        sym = r_a["sym"]
        cfg = SYMBOL_CONFIG.get(sym, {})
        sp  = cfg.get("spread", 0)
        adopt = "✅採用候補" if best_pf >= 2.0 else ("⚠️要検討" if best_pf >= 1.5 else "❌不採用")
        print(f"  {sym:8} | {pf_a:>11.2f} {pf_b:>11.2f} | {sp:>5.1f}pips {winner:>8} {adopt:>6}")

    print("\n  ■ カテゴリ別 avg OOS PF")
    for cat in ["FX", "JPY"]:
        cat_res = [r for r in all_results if r["cat"] == cat]
        if not cat_res: continue
        for logic in ["A", "B"]:
            pfs = [r["oos"].get("pf", 0) for r in cat_res
                   if r["logic"] == logic and r["oos"] and r["oos"].get("pf", 0) < 99]
            avg = np.mean(pfs) if pfs else 0
            print(f"    {cat} Logic-{logic}: avg OOS PF={avg:.2f}  ({len(pfs)}銘柄)")

    # ── CSVに保存 ─────────────────────────────────────────────────
    rows = []
    for r in all_results:
        row = {"sym": r["sym"], "cat": r["cat"], "logic": r["logic"]}
        for period, key in [("is", r["is"]), ("oos", r["oos"])]:
            for k, v in key.items():
                row[f"{period}_{k}"] = v
        rows.append(row)
    if rows:
        out_path = os.path.join(OUT_DIR, "backtest_new_fx_symbols.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  結果を保存: {out_path}")

if __name__ == "__main__":
    main()
