"""
backtest_fx_pro_course.py
=========================
FX6銘柄 × OANDAプロコース想定バックテスト

【対象銘柄】
  EURUSD / GBPUSD / AUDUSD / NZDUSD / USDCAD / USDCHF

【コスト設定（OANDAプロコース）】
  - Raw spread（生スプレッド）: 主要ペア 0.1〜0.3pips（エントリー価格に反映）
  - 手数料: 5 USD / 100,000通貨 / 片道（往復10 USD = 1,500円/lot@150円）
             → 毎トレードequityから直接差引き

【ロジック】
  Goldロジック（Logic-A）: 4H EMA20 + 日足EMA20方向一致 + KMID + KLOW + EMA距離
  エントリー: E2方式（スパイク除外 / 15m足 最大3本待ち）
  SL: 二番底/天井 ± ATR×0.15
  TP: リスク幅 × 2.5（RR=2.5）
  半利確: 1R到達でポジション50%決済 → SLをBEへ

【IS/OOS】
  IS:  2025-01-01 〜 2025-05-31
  OOS: 2025-06-01 〜 2026-02-28

【データ】
  EURUSD/GBPUSD/AUDUSD: data/ohlc/{SYM}_15m.csv + _4h.csv
  NZDUSD/USDCAD/USDCHF: data/{sym}_is_15m.csv + _oos_15m.csv + is/oos_4h.csv
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
RISK_PCT    = 0.02        # 1トレード 2%リスク
RR_RATIO    = 2.5         # RR比
HALF_R      = 1.0         # 半利確タイミング（1R）
USDJPY_RATE = 150.0       # バックテスト固定レート

# ── OANDAプロコース コスト設定 ────────────────────────────────────
# Raw spread（生スプレッド）pips: エントリー価格に反映（片道）
PRO_RAW_SPREAD_PIPS = {
    "EURUSD": 0.1,
    "GBPUSD": 0.2,
    "AUDUSD": 0.2,
    "NZDUSD": 0.3,
    "USDCAD": 0.2,
    "USDCHF": 0.3,
}
# 手数料: 5 USD / 100,000通貨 / 片道（往復10 USD）
COMMISSION_USD_PER_LOT = 5.0   # per 100,000 units, per side

def calc_commission_jpy(lot: float) -> float:
    """往復手数料（JPY）= lot/100,000 × 10 USD × 150 JPY/USD"""
    return (lot / 100_000) * COMMISSION_USD_PER_LOT * 2 * USDJPY_RATE

# ── フィルター定数 ────────────────────────────────────────────────
KLOW_THR        = 0.0015   # 下ヒゲ比率 < 0.15%
EMA_DIST_MIN    = 1.0      # EMA距離 ≥ ATR×1.0
PATTERN_TOL     = 0.30     # 二番底/天井許容幅 ATR×0.30
E2_SPIKE_ATR    = 2.0      # スパイク除外閾値
E2_WINDOW_BARS  = 3        # 最大3本待ち（15m）
MAX_LOOKAHEAD   = 5_000    # 最大探索バー数

# ── ディレクトリ ─────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OHLC_DIR = os.path.join(DATA_DIR, "ohlc")
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 銘柄定義 ─────────────────────────────────────────────────────
SYMBOLS = [
    {"name": "EURUSD", "cat": "FX",     "data": "ohlc"},
    {"name": "GBPUSD", "cat": "FX",     "data": "ohlc"},
    {"name": "AUDUSD", "cat": "FX",     "data": "ohlc"},
    {"name": "NZDUSD", "cat": "FX",     "data": "split"},
    {"name": "USDCAD", "cat": "FX_INV", "data": "split"},
    {"name": "USDCHF", "cat": "FX_INV", "data": "split"},
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
    sym  = sym_info["name"]
    syml = sym.lower()

    if sym_info["data"] == "ohlc":
        d15m = load_csv(os.path.join(OHLC_DIR, f"{sym}_15m.csv"))
        d4h  = load_csv(os.path.join(OHLC_DIR, f"{sym}_4h.csv"))
    else:
        is15  = load_csv(os.path.join(DATA_DIR, f"{syml}_is_15m.csv"))
        oos15 = load_csv(os.path.join(DATA_DIR, f"{syml}_oos_15m.csv"))
        is4h  = load_csv(os.path.join(DATA_DIR, f"{syml}_is_4h.csv"))
        oos4h = load_csv(os.path.join(DATA_DIR, f"{syml}_oos_4h.csv"))
        d15m = pd.concat([is15,  oos15]).sort_index() if (is15  is not None and oos15 is not None) else None
        d4h  = pd.concat([is4h,  oos4h]).sort_index() if (is4h  is not None and oos4h is not None) else None
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
    # 日足: 4hからresample
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

        # 日足EMA20方向一致（Goldロジック）
        d1_before = d1d[d1d.index.normalize() < h1_ct.normalize()]
        if len(d1_before) == 0: continue
        if d1_before.iloc[-1]["trend1d"] != trend: continue

        # 共通フィルター
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if not check_ema_dist(h4_lat): continue

        # 二番底/天井パターン
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
def simulate(signals, d15m, sym):
    if not signals:
        return [], INIT_CASH, 0, 0
    rm   = RiskManager(sym, risk_pct=RISK_PCT)
    m15t = d15m.index
    m15h = d15m["high"].values
    m15l = d15m["low"].values
    equity = INIT_CASH; peak = INIT_CASH; mdd = 0.0
    trades = []
    total_commission = 0.0

    for sig in signals:
        lot = rm.calc_lot(equity, sig["risk"], sig["ep"], usdjpy_rate=USDJPY_RATE)
        if lot <= 0:
            continue

        # ── 往復手数料をequityから差引き ──
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

        trades.append({
            "result":     result,
            "pnl":        total_pnl,
            "commission": commission,
        })
        peak = max(peak, equity)
        mdd  = max(mdd, (peak - equity) / peak * 100)

    return trades, equity, mdd, total_commission


def calc_stats(trades, init=INIT_CASH):
    if not trades:
        return {}
    df    = pd.DataFrame(trades)
    n     = len(df)
    wins  = df[df["pnl"] > 0]["pnl"]
    loses = df[df["pnl"] < 0]["pnl"]
    wr    = len(wins) / n
    gw    = wins.sum(); gl = abs(loses.sum())
    pf    = gw / gl if gl > 0 else float("inf")
    total_commission = df["commission"].sum()
    return {
        "n": n, "wr": wr, "pf": pf,
        "total_commission": total_commission,
    }


# ── 1期間ラン ────────────────────────────────────────────────────
def run_period(d15m_full, d4h_full, sym, start, end):
    d15m = slice_period(d15m_full, start, end)
    if d15m is None or len(d15m) == 0:
        return {}, 0.0

    pip       = SYMBOL_CONFIG[sym]["pip"]
    raw_pips  = PRO_RAW_SPREAD_PIPS[sym]
    spread_pr = raw_pips * pip   # 価格単位のraw spread（エントリー価格反映分）

    atr_15m = calc_atr(d15m, 10).to_dict()
    m15c    = {
        "idx":    d15m.index,
        "opens":  d15m["open"].values,
        "highs":  d15m["high"].values,
        "lows":   d15m["low"].values,
    }

    sigs                     = generate_signals(d15m, d4h_full, spread_pr, atr_15m, m15c)
    trades, final_eq, mdd, total_comm = simulate(sigs, d15m, sym)
    st = calc_stats(trades)
    if st:
        st["mdd"]       = mdd
        st["final_eq"]  = final_eq
        st["commission"]= total_comm
        st["multiplier"]= final_eq / INIT_CASH
    return st, total_comm


# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*90)
    print("  FX6銘柄 OANDAプロコース想定バックテスト（Goldロジック）")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  初期資産: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%/trade  RR: {RR_RATIO}")
    print(f"  Raw spread: 0.1〜0.3pips（エントリー価格反映）")
    print(f"  手数料:     5 USD/100,000通貨/片道 往復10 USD（equityから差引き）")
    print("="*90)
    print(f"  {'銘柄':8} {'Cat':8} | {'IS n':>5} {'IS WR':>6} {'IS PF':>6} {'IS MDD':>6} {'IS倍率':>6} "
          f"{'IS手数料':>10} | {'OOS n':>5} {'OOS WR':>6} {'OOS PF':>7} {'OOS MDD':>7} {'OOS倍率':>7} {'OOS手数料':>10}")
    print("-"*90)

    all_results = []

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        d15m_full, d4h_full = load_all(sym_info)
        if d15m_full is None or d4h_full is None:
            print(f"  {sym}: データ不足 → スキップ")
            continue

        print(f"  {sym} 計算中...", end=" ", flush=True)
        is_st,  _ = run_period(d15m_full, d4h_full, sym, IS_START,  IS_END)
        oos_st, _ = run_period(d15m_full, d4h_full, sym, OOS_START, OOS_END)
        print("完了")

        def fmt(st):
            if not st:
                return f"{'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>10}"
            pf_s = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
            comm = f"{st['commission']:,.0f}円"
            return (f"{st['n']:>5} {st['wr']*100:>5.1f}% {pf_s:>6} "
                    f"{st['mdd']:>5.1f}% {st['multiplier']:>5.2f}x {comm:>10}")

        print(f"  {sym:8} {sym_info['cat']:8} | {fmt(is_st)} | {fmt(oos_st)}")
        all_results.append({
            "sym": sym, "cat": sym_info["cat"],
            "raw_spread": PRO_RAW_SPREAD_PIPS[sym],
            "is": is_st, "oos": oos_st,
        })

    # ── OOSサマリー ───────────────────────────────────────────────
    print("\n" + "="*90)
    print("  ■ OOS サマリー（OANDAプロコース / Goldロジック）")
    print(f"  {'銘柄':8} {'Cat':8} {'Raw sp':>8} | {'OOS PF':>8} {'OOS WR':>8} {'OOS MDD':>8} "
          f"{'OOS倍率':>8} {'手数料合計':>12} {'判定':>6}")
    print("-"*90)
    oos_pf_list = []
    for r in all_results:
        oos = r["oos"]
        if not oos:
            print(f"  {r['sym']:8} {r['cat']:8} {'N/A':>8} | {'N/A':>8}")
            continue
        pf_s  = f"{oos['pf']:.2f}" if oos['pf'] < 99 else "∞"
        judge = "✅" if oos["pf"] >= 2.0 else ("△" if oos["pf"] >= 1.5 else "❌")
        comm  = f"{oos['commission']:,.0f}円"
        print(f"  {r['sym']:8} {r['cat']:8} {r['raw_spread']:>6.1f}p | "
              f"{pf_s:>8} {oos['wr']*100:>7.1f}% {oos['mdd']:>7.1f}% "
              f"{oos['multiplier']:>7.2f}x {comm:>12} {judge:>6}")
        oos_pf_list.append(oos["pf"])

    if oos_pf_list:
        print(f"\n  FX avg OOS PF = {sum(oos_pf_list)/len(oos_pf_list):.2f}")

    # ── コスト比較 ─────────────────────────────────────────────────
    print("\n  ■ スプレッド設定比較")
    print(f"  {'銘柄':8} | {'旧設定(pips)':>12} {'Proコース生値(pips)':>20} {'手数料(往復)':>14}")
    print("-"*60)
    old_spreads = {"EURUSD": 0.0, "GBPUSD": 0.1, "AUDUSD": 0.0,
                   "NZDUSD": 0.5, "USDCAD": 0.1, "USDCHF": 0.2}
    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        old = old_spreads.get(sym, "?")
        new = PRO_RAW_SPREAD_PIPS[sym]
        print(f"  {sym:8} | {old:>12} → {new:>18.1f}   +0.5pips/片道手数料")

    # ── CSV保存 ───────────────────────────────────────────────────
    rows = []
    for r in all_results:
        for period, st in [("IS", r["is"]), ("OOS", r["oos"])]:
            if not st: continue
            rows.append({
                "sym": r["sym"], "cat": r["cat"], "period": period,
                "raw_spread_pips": r["raw_spread"],
                "n": st["n"], "wr": round(st["wr"]*100, 1),
                "pf": round(st["pf"], 2),
                "mdd": round(st["mdd"], 1),
                "multiplier": round(st["multiplier"], 3),
                "commission_jpy": round(st["commission"], 0),
            })
    if rows:
        out_csv = os.path.join(OUT_DIR, "fx_pro_course_results.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  → 結果保存: {out_csv}")


if __name__ == "__main__":
    main()
