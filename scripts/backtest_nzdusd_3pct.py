"""
backtest_nzdusd_3pct.py
========================
NZDUSD 1銘柄集中バックテスト（リスク3%）

【設定】
  - 銘柄: NZDUSD のみ
  - リスク: 3% / トレード（2%→3%に引き上げ）
  - スプレッド: 0.3pips（OANDAプロコース生スプレッド）
  - 手数料: 5 USD/100,000通貨/片道（往復10 USD = 1,500円/lot@150円）
  - ロジック: Goldロジック（4H EMA20 + 日足EMA20 + KMID + KLOW + EMA距離）
  - エントリー: E2方式（スパイク除外 / 15m足 最大3本待ち）

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

# ── 期間設定 ────────────────────────────────────────────────────
IS_START  = "2025-01-01"
IS_END    = "2025-05-31"
OOS_START = "2025-06-01"
OOS_END   = "2026-02-28"

# ── 資金・リスク設定 ─────────────────────────────────────────────
INIT_CASH   = 1_000_000   # 初期資産 100万円
RISK_PCT    = 0.03        # ★ 3%リスク（集中戦略）
RR_RATIO    = 2.5
HALF_R      = 1.0
USDJPY_RATE = 150.0

# ── コスト設定（OANDAプロコース） ────────────────────────────────
SPREAD_PIPS          = 0.3    # 生スプレッド 0.3pips（片道、エントリー価格反映）
COMMISSION_USD_PER_LOT = 5.0  # 5 USD/100,000通貨/片道

def calc_commission_jpy(lot: float) -> float:
    """往復手数料（JPY）= lot/100,000 × 10 USD × 150 JPY/USD"""
    return (lot / 100_000) * COMMISSION_USD_PER_LOT * 2 * USDJPY_RATE

# ── フィルター定数 ────────────────────────────────────────────────
KLOW_THR       = 0.0015
EMA_DIST_MIN   = 1.0
PATTERN_TOL    = 0.30
E2_SPIKE_ATR   = 2.0
E2_WINDOW_BARS = 3
MAX_LOOKAHEAD  = 5_000

SYM = "NZDUSD"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


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


def load_nzdusd():
    syml = SYM.lower()
    is15  = load_csv(os.path.join(DATA_DIR, f"{syml}_is_15m.csv"))
    oos15 = load_csv(os.path.join(DATA_DIR, f"{syml}_oos_15m.csv"))
    is4h  = load_csv(os.path.join(DATA_DIR, f"{syml}_is_4h.csv"))
    oos4h = load_csv(os.path.join(DATA_DIR, f"{syml}_oos_4h.csv"))
    d15m = pd.concat([is15, oos15]).sort_index() if (is15 is not None and oos15 is not None) else None
    d4h  = pd.concat([is4h, oos4h]).sort_index() if (is4h is not None and oos4h is not None) else None
    if d15m is not None: d15m = d15m[~d15m.index.duplicated(keep="first")]
    if d4h  is not None: d4h  = d4h[~d4h.index.duplicated(keep="first")]
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
    d1 = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna(subset=["open", "close"])
    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1


def build_1h(df15m):
    df = df15m.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


# ── フィルター ─────────────────────────────────────────────────────
def check_kmid(bar, direction):
    return (direction == 1  and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])

def check_klow(bar):
    o, l = bar["open"], bar["low"]
    return (min(bar["open"], bar["close"]) - l) / o < KLOW_THR if o > 0 else False

def check_ema_dist(bar):
    d = abs(bar["close"] - bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a > 0 and d >= a * EMA_DIST_MIN


# ── E2エントリー（15m版） ─────────────────────────────────────────
def pick_e2_15m(signal_time, direction, spread_price, atr_15m_d, m15c):
    idx = m15c["idx"]
    s   = idx.searchsorted(signal_time, side="left")
    e   = min(s + E2_WINDOW_BARS, len(idx))
    for i in range(s, e):
        bar_range = m15c["highs"][i] - m15c["lows"][i]
        atr_val   = atr_15m_d.get(idx[i], np.nan)
        if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR:
            continue
        ep = m15c["opens"][i] + (spread_price if direction == 1 else -spread_price)
        return idx[i], ep
    return None, None


# ── シグナル生成（Goldロジック） ──────────────────────────────────
def generate_signals(d15m_period, d4h_full, spread_price, atr_15m_d, m15c):
    d4h, d1d = build_4h(d4h_full)
    d1h = build_1h(d15m_period)
    signals = []; used = set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct = h1_times[i]
        h1_p1 = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
        atr_1h = d1h.iloc[i]["atr"]
        if pd.isna(atr_1h) or atr_1h <= 0: continue

        h4_before = d4h[d4h.index < h1_ct]
        if len(h4_before) < 2: continue
        h4_lat = h4_before.iloc[-1]
        if pd.isna(h4_lat.get("atr", np.nan)): continue
        trend = h4_lat["trend"]; h4_atr = h4_lat["atr"]

        d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
        if len(d1_before) == 0: continue
        if d1_before.iloc[-1]["trend1d"] != trend: continue

        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if not check_ema_dist(h4_lat): continue

        tol = atr_1h * PATTERN_TOL
        direction = trend
        if direction == 1:  v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:               v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol: continue

        et, ep = pick_e2_15m(h1_ct, direction, spread_price, atr_15m_d, m15c)
        if et is None or et in used: continue

        raw = ep - spread_price if direction == 1 else ep + spread_price
        if direction == 1: sl = min(v1, v2) - atr_1h * 0.15; risk = raw - sl
        else:              sl = max(v1, v2) + atr_1h * 0.15; risk = sl - raw
        if 0 < risk <= h4_atr * 2:
            tp = raw + direction * risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk})
            used.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


# ── 決済探索 ──────────────────────────────────────────────────────
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
                    if lows[j]  <= be: return j, be, "win",  True
                    if highs[j] >= tp: return j, tp, "win",  True
                return -1, None, None, True
        else:
            if h  >= sl: return i, sl, "loss", False
            if lo <= tp: return i, tp, "win",  False
            if lo <= half:
                be = ep
                for j in range(i+1, lim):
                    if highs[j] >= be: return j, be, "win",  True
                    if lows[j]  <= tp: return j, tp, "win",  True
                return -1, None, None, True
    return -1, None, None, False


# ── シミュレーション ──────────────────────────────────────────────
def simulate(signals, d15m, risk_pct):
    if not signals:
        return [], INIT_CASH, 0, 0
    rm   = RiskManager(SYM, risk_pct=risk_pct)
    m15t = d15m.index
    m15h = d15m["high"].values
    m15l = d15m["low"].values
    equity = INIT_CASH; peak = INIT_CASH; mdd = 0.0
    trades = []; total_commission = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0:
            continue

        commission = calc_commission_jpy(lot)
        equity -= commission
        total_commission += commission

        sp = m15t.searchsorted(sig["time"], side="right")
        if sp >= len(m15t):
            continue

        ei, xp, result, half_done = _find_exit(
            m15h[sp:], m15l[sp:],
            sig["ep"], sig["sl"], sig["tp"],
            sig["risk"], sig["dir"]
        )
        if result is None:
            continue

        half_pnl = 0.0
        if half_done:
            hp       = sig["ep"] + sig["dir"] * sig["risk"] * HALF_R
            half_pnl = rm.calc_pnl_jpy(sig["dir"], sig["ep"], hp, lot * 0.5, USDJPY_RATE, sig["ep"])
            equity  += half_pnl
            rem      = lot * 0.5
        else:
            rem = lot

        pnl    = rm.calc_pnl_jpy(sig["dir"], sig["ep"], xp, rem, USDJPY_RATE, sig["ep"])
        equity += pnl
        total_pnl = half_pnl + pnl

        trades.append({"result": result, "pnl": total_pnl, "commission": commission})
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, total_commission


def calc_stats(trades, init=INIT_CASH):
    if not trades:
        return {}
    df   = pd.DataFrame(trades)
    n    = len(df)
    wins = df[df["pnl"] > 0]["pnl"]
    los  = df[df["pnl"] < 0]["pnl"]
    wr   = len(wins) / n
    pf   = wins.sum() / abs(los.sum()) if len(los) > 0 else float("inf")
    return {"n": n, "wr": wr, "pf": pf,
            "total_commission": df["commission"].sum()}


def run_period(d15m_full, d4h_full, start, end, risk_pct):
    d15m = slice_period(d15m_full, start, end)
    if d15m is None or len(d15m) == 0:
        return {}, 0.0

    pip       = SYMBOL_CONFIG[SYM]["pip"]
    spread_pr = SPREAD_PIPS * pip

    atr_15m = calc_atr(d15m, 10).to_dict()
    m15c = {
        "idx":   d15m.index,
        "opens": d15m["open"].values,
        "highs": d15m["high"].values,
        "lows":  d15m["low"].values,
    }

    sigs = generate_signals(d15m, d4h_full, spread_pr, atr_15m, m15c)
    trades, final_eq, mdd, total_comm = simulate(sigs, d15m, risk_pct)
    st = calc_stats(trades)
    if st:
        st["mdd"]       = mdd
        st["final_eq"]  = final_eq
        st["commission"]= total_comm
        st["multiplier"]= final_eq / INIT_CASH
    return st, total_comm


# ── 月次分析 ─────────────────────────────────────────────────────
def monthly_pnl(d15m_full, d4h_full, risk_pct):
    results = {}
    d15m_oos = slice_period(d15m_full, OOS_START, OOS_END)
    if d15m_oos is None: return results

    pip       = SYMBOL_CONFIG[SYM]["pip"]
    spread_pr = SPREAD_PIPS * pip
    atr_15m   = calc_atr(d15m_oos, 10).to_dict()
    m15c = {
        "idx":   d15m_oos.index,
        "opens": d15m_oos["open"].values,
        "highs": d15m_oos["high"].values,
        "lows":  d15m_oos["low"].values,
    }
    sigs = generate_signals(d15m_oos, d4h_full, spread_pr, atr_15m, m15c)

    rm = RiskManager(SYM, risk_pct=risk_pct)
    m15t = d15m_oos.index
    m15h = d15m_oos["high"].values
    m15l = d15m_oos["low"].values
    equity = INIT_CASH

    for sig in sigs:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0: continue
        equity -= calc_commission_jpy(lot)
        sp = m15t.searchsorted(sig["time"], side="right")
        if sp >= len(m15t): continue
        ei, xp, result, half_done = _find_exit(
            m15h[sp:], m15l[sp:], sig["ep"], sig["sl"], sig["tp"], sig["risk"], sig["dir"])
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
        ym = sig["time"].strftime("%Y-%m")
        results.setdefault(ym, []).append(half_pnl + pnl)

    return results


# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*75)
    print("  NZDUSD 集中戦略バックテスト（リスク3%）")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  初期資産: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%/trade  RR: {RR_RATIO}")
    print(f"  Raw spread: {SPREAD_PIPS}pips  手数料: {COMMISSION_USD_PER_LOT}USD/100k通貨/片道")
    print("="*75)

    d15m_full, d4h_full = load_nzdusd()
    if d15m_full is None or d4h_full is None:
        print("  ❌ データ不足")
        return

    print("  NZDUSD IS 計算中...", end=" ", flush=True)
    is_st,  _ = run_period(d15m_full, d4h_full, IS_START,  IS_END,  RISK_PCT)
    print("完了")
    print("  NZDUSD OOS 計算中...", end=" ", flush=True)
    oos_st, _ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, RISK_PCT)
    print("完了")

    def fmt(st, label):
        if not st:
            print(f"  {label}: データなし")
            return
        pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
        judge = "✅" if st["pf"] >= 3.0 else ("△" if st["pf"] >= 2.0 else "❌")
        print(f"\n  【{label}】")
        print(f"    トレード数:  {st['n']}")
        print(f"    勝率:        {st['wr']*100:.1f}%")
        print(f"    PF:          {pf_s}  {judge}")
        print(f"    MDD:         {st['mdd']:.1f}%")
        print(f"    資産倍率:    {st['multiplier']:.2f}x  (→ {st['final_eq']:,.0f}円)")
        print(f"    手数料合計:  {st['commission']:,.0f}円")

    fmt(is_st,  "IS  (2025/01-05)")
    fmt(oos_st, "OOS (2025/06-2026/02)")

    # IS/OOS 乖離チェック
    if is_st and oos_st and is_st.get("pf") and oos_st.get("pf"):
        ratio = oos_st["pf"] / is_st["pf"]
        flag  = "✅ 過学習なし" if ratio >= 0.7 else "⚠️ 過学習疑い"
        print(f"\n  OOS/IS PF比: {ratio:.2f}  {flag}")

    # リスク2%との比較
    print("\n" + "-"*75)
    print("  ■ リスク2% vs 3% 比較（OOS）")
    print("  2%計算中...", end=" ", flush=True)
    oos_2pct, _ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, 0.02)
    print("完了")
    print(f"\n  {'項目':15} {'2%リスク':>12} {'3%リスク':>12} {'差':>12}")
    print(f"  {'-'*51}")
    if oos_2pct and oos_st:
        print(f"  {'PF':15} {oos_2pct['pf']:>12.2f} {oos_st['pf']:>12.2f}")
        print(f"  {'勝率':15} {oos_2pct['wr']*100:>11.1f}% {oos_st['wr']*100:>11.1f}%")
        print(f"  {'MDD':15} {oos_2pct['mdd']:>11.1f}% {oos_st['mdd']:>11.1f}%  {'⚠️ 上昇' if oos_st['mdd'] > oos_2pct['mdd'] else '✅'}")
        print(f"  {'資産倍率':15} {oos_2pct['multiplier']:>11.2f}x {oos_st['multiplier']:>11.2f}x  (+{oos_st['multiplier']-oos_2pct['multiplier']:.2f}x)")
        print(f"  {'最終資産':15} {oos_2pct['final_eq']:>11,.0f}円 {oos_st['final_eq']:>11,.0f}円")
        print(f"  {'手数料':15} {oos_2pct['commission']:>11,.0f}円 {oos_st['commission']:>11,.0f}円")

    # 月次内訳（OOS）
    print("\n" + "-"*75)
    print("  ■ OOS 月次損益（3%リスク）")
    monthly = monthly_pnl(d15m_full, d4h_full, RISK_PCT)
    plus_months = 0; total_months = 0
    for ym in sorted(monthly.keys()):
        pnls = monthly[ym]
        total = sum(pnls)
        sign  = "+" if total >= 0 else ""
        flag  = "✅" if total >= 0 else "❌"
        print(f"    {ym}: {sign}{total:>10,.0f}円  ({len(pnls)}トレード)  {flag}")
        total_months += 1
        if total >= 0: plus_months += 1
    if total_months > 0:
        print(f"  プラス月: {plus_months}/{total_months}  ({plus_months/total_months*100:.0f}%)")


if __name__ == "__main__":
    main()
