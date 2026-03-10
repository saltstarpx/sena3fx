"""
backtest_xauusd_pro_course.py
==============================
XAUUSD プロコース想定バックテスト（リスク2%）

【コスト設定比較】
  旧設定: スプレッド 5.2pips（SYMBOL_CONFIG raw_spread）、手数料なし
  新設定: スプレッド 5.2pips + 手数料あり

【手数料（OANDAプロコース 貴金属）】
  $5 USD / 100oz / 片道（往復 $10/100oz = $0.10/oz）
  → lot_oz / 100 × 10 USD × 150 JPY/USD

【ロジック】
  Goldロジック: 4H EMA20 + 日足EMA20 + KMID + KLOW + EMA距離
  エントリー: E2方式（スパイク除外 / 15m足 最大3本待ち）
  SL: 二番底/天井 ± ATR×0.15  TP: リスク幅×2.5

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
INIT_CASH   = 1_000_000
RISK_PCT    = 0.02        # 2%リスク
RR_RATIO    = 2.5
HALF_R      = 1.0
USDJPY_RATE = 150.0

# ── コスト設定 ────────────────────────────────────────────────────
SPREAD_PIPS            = 5.2   # SYMBOL_CONFIGのraw_spread値（旧設定と同じ）
COMMISSION_USD_PER_LOT = 5.0   # $5/100oz/片道（OANDAプロコース 貴金属）

def calc_commission_jpy_metals(lot_oz: float) -> float:
    """往復手数料（JPY）= lot_oz/100 × 10 USD × 150 JPY/USD
    ※ 貴金属は1lot=100oz基準（FX 1lot=100,000通貨とは異なる）"""
    return (lot_oz / 100) * COMMISSION_USD_PER_LOT * 2 * USDJPY_RATE

# ── フィルター定数 ────────────────────────────────────────────────
KLOW_THR       = 0.0015
EMA_DIST_MIN   = 1.0
PATTERN_TOL    = 0.30
E2_SPIKE_ATR   = 2.0
E2_WINDOW_BARS = 3
MAX_LOOKAHEAD  = 5_000

SYM      = "XAUUSD"
OHLC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ohlc")


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


# ── E2エントリー ─────────────────────────────────────────────────
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
def simulate(signals, d15m, use_commission=True):
    if not signals:
        return [], INIT_CASH, 0, 0
    rm   = RiskManager(SYM, risk_pct=RISK_PCT)
    m15t = d15m.index
    m15h = d15m["high"].values
    m15l = d15m["low"].values
    equity = INIT_CASH; peak = INIT_CASH; mdd = 0.0
    trades = []; total_commission = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0:
            continue

        if use_commission:
            commission = calc_commission_jpy_metals(lot)
            equity -= commission
            total_commission += commission
        else:
            commission = 0.0

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

        trades.append({"result": result, "pnl": total_pnl, "commission": commission, "lot": lot})
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
            "total_commission": df["commission"].sum(),
            "avg_lot": df["lot"].mean()}


def run_period(d15m_full, d4h_full, start, end, use_commission):
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
    trades, final_eq, mdd, total_comm = simulate(sigs, d15m, use_commission)
    st = calc_stats(trades)
    if st:
        st["mdd"]       = mdd
        st["final_eq"]  = final_eq
        st["commission"]= total_comm
        st["multiplier"]= final_eq / INIT_CASH
    return st, total_comm


# ── 月次分析 ─────────────────────────────────────────────────────
def monthly_pnl(d15m_full, d4h_full, use_commission=True):
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

    rm = RiskManager(SYM, risk_pct=RISK_PCT)
    m15t = d15m_oos.index
    m15h = d15m_oos["high"].values
    m15l = d15m_oos["low"].values
    equity = INIT_CASH

    for sig in sigs:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0: continue
        if use_commission:
            equity -= calc_commission_jpy_metals(lot)
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
    print("  XAUUSD プロコース想定バックテスト（リスク2%）")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  初期資産: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%/trade  RR: {RR_RATIO}")
    print(f"  スプレッド: {SPREAD_PIPS}pips（SYMBOL_CONFIG raw_spread値）")
    print(f"  手数料: {COMMISSION_USD_PER_LOT}USD/100oz/片道（往復 10USD/100oz）")
    print("="*75)

    d15m_full = load_csv(os.path.join(OHLC_DIR, "XAUUSD_15m.csv"))
    d4h_full  = load_csv(os.path.join(OHLC_DIR, "XAUUSD_4h.csv"))

    if d15m_full is None or d4h_full is None:
        print("  ❌ データ不足")
        return

    # ── 旧設定（手数料なし）vs プロコース（手数料あり）
    print("\n  ■ IS 計算中...")
    is_old,  _ = run_period(d15m_full, d4h_full, IS_START,  IS_END,  use_commission=False)
    is_new,  _ = run_period(d15m_full, d4h_full, IS_START,  IS_END,  use_commission=True)

    print("  ■ OOS 計算中...")
    oos_old, _ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, use_commission=False)
    oos_new, _ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, use_commission=True)
    print("  完了\n")

    print(f"  {'':30} {'旧設定(手数料なし)':>18} {'Proコース(手数料あり)':>22}")
    print(f"  {'-'*72}")

    def cmp(label, key, fmt_fn=None):
        def fv(st, k):
            if not st or k not in st: return "N/A"
            v = st[k]
            return fmt_fn(v) if fmt_fn else (f"{v:.2f}" if v < 99 else "∞")
        print(f"  {'IS  '+label:30} {fv(is_old, key):>18} {fv(is_new, key):>22}")
        print(f"  {'OOS '+label:30} {fv(oos_old, key):>18} {fv(oos_new, key):>22}")
        print()

    cmp("トレード数", "n",          lambda v: f"{v:.0f}")
    cmp("勝率",       "wr",         lambda v: f"{v*100:.1f}%")
    cmp("PF",         "pf",         lambda v: f"{v:.2f}" if v < 99 else "∞")
    cmp("MDD",        "mdd",        lambda v: f"{v:.1f}%")   # すでに%単位
    cmp("資産倍率",   "multiplier", lambda v: f"{v:.2f}x")
    cmp("手数料合計", "commission", lambda v: f"{v:,.0f}円")

    # ── 合否判定
    print("="*75)
    print("  ■ OOS 合否判定（Proコース）")
    if oos_new:
        pf  = oos_new["pf"]
        wr  = oos_new["wr"]
        mdd = oos_new["mdd"]
        mul = oos_new["multiplier"]
        print(f"    PF:    {pf:.2f}  {'✅ PF≥2.0' if pf>=2.0 else '❌ PF<2.0'}")
        print(f"    勝率:  {wr*100:.1f}%  {'✅' if wr>=0.50 else '❌'}")
        print(f"    MDD:   {mdd:.1f}%  {'✅ ≤30%' if mdd<=30 else '⚠️ >30%'}")
        print(f"    倍率:  {mul:.2f}x → {mul*INIT_CASH/10000:.0f}万円")

    # IS/OOS乖離
    if is_new and oos_new and is_new.get("pf") and oos_new.get("pf"):
        ratio = oos_new["pf"] / is_new["pf"]
        flag  = "✅ 過学習なし" if ratio >= 0.7 else "⚠️ 過学習疑い"
        print(f"    OOS/IS PF比: {ratio:.2f}  {flag}")

    # ── 月次（プロコース）
    print("\n  ■ OOS 月次損益（Proコース / 2%リスク）")
    monthly = monthly_pnl(d15m_full, d4h_full, use_commission=True)
    plus_months = 0; total_months = 0
    for ym in sorted(monthly.keys()):
        pnls  = monthly[ym]
        total = sum(pnls)
        sign  = "+" if total >= 0 else ""
        flag  = "✅" if total >= 0 else "❌"
        print(f"    {ym}: {sign}{total:>10,.0f}円  ({len(pnls)}トレード)  {flag}")
        total_months += 1
        if total >= 0: plus_months += 1
    if total_months > 0:
        print(f"  プラス月: {plus_months}/{total_months}  ({plus_months/total_months*100:.0f}%)")

    # ── avg lot サマリー
    if oos_new and oos_new.get("avg_lot"):
        avg_oz = oos_new["avg_lot"]
        comm_per = calc_commission_jpy_metals(avg_oz)
        print(f"\n  【参考】平均lot: {avg_oz:.1f}oz → 手数料/trade: {comm_per:.0f}円")
        risk_jpy = INIT_CASH * RISK_PCT
        print(f"  1トレードリスク(初期): {risk_jpy:,.0f}円 → 手数料比: {comm_per/risk_jpy*100:.1f}%")


if __name__ == "__main__":
    main()
