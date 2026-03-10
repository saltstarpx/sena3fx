"""
backtest_nzdusd_3pct.py
========================
NZDUSD 1銘柄集中バックテスト（アダプティブリスク 2〜3%）

【設定】
  - 銘柄: NZDUSD のみ
  - スプレッド: 0.3pips（OANDAプロコース生スプレッド）
  - 手数料: 5 USD/100,000通貨/片道（往復10 USD = 1,500円/lot@150円）
  - ロジック: Goldロジック（4H EMA20 + 日足EMA20 + KMID + KLOW + EMA距離）
  - エントリー: E2方式（スパイク除外 / 15m足 最大3本待ち）

【アダプティブリスク】
  - 初期リスク: 2%
  - 勝ち: +0.5%（上限 3%）→ 2.0 / 2.5 / 3.0 の3段階
  - 負け: -0.5%（下限 2%）
  - ※ stepはOOS最適化なし（カーブフィッティング防止）
  - 固定2% / 固定3% / アダプティブ 三者比較

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
RISK_PCT    = 0.03        # 固定3%（比較用）
RR_RATIO    = 2.5
HALF_R      = 1.0
USDJPY_RATE = 150.0

# ── アダプティブリスク設定 ────────────────────────────────────────
RISK_INIT   = 0.02        # 初期リスク（下限と同じ）
RISK_MIN    = 0.02        # 下限
RISK_MAX    = 0.03        # 上限（2.0 / 2.5 / 3.0 の3段階）
RISK_STEP   = 0.005       # 勝ち→+0.5% / 負け→-0.5%（カーブフィッティング防止の固定値）

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
MIN_RISK_PIPS  = 3        # 最低SL幅（pips）: 縮退シグナル防止（ゼロ除算→巨大ロット対策）

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
        if SYMBOL_CONFIG[SYM]["pip"] * MIN_RISK_PIPS <= risk <= h4_atr * 2:
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


# ── アダプティブリスク シミュレーション ─────────────────────────
def simulate_adaptive(signals, d15m):
    """勝ち→+0.5% / 負け→-0.5%、下限2%・上限3%のアダプティブリスク"""
    if not signals:
        return [], INIT_CASH, 0, 0
    current_risk = RISK_INIT
    rm   = RiskManager(SYM, risk_pct=current_risk)
    m15t = d15m.index
    m15h = d15m["high"].values
    m15l = d15m["low"].values
    equity = INIT_CASH; peak = INIT_CASH; mdd = 0.0
    trades = []; total_commission = 0.0

    for sig in signals:
        rm.risk_pct = current_risk
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

        # ── リスク更新（次のトレードに反映） ──────────────────────
        if total_pnl > 0:
            current_risk = round(min(current_risk + RISK_STEP, RISK_MAX), 4)
        else:
            current_risk = round(max(current_risk - RISK_STEP, RISK_MIN), 4)

        trades.append({"result": result, "pnl": total_pnl,
                       "commission": commission, "risk_used": rm.risk_pct})
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
    result = {"n": n, "wr": wr, "pf": pf,
              "total_commission": df["commission"].sum()}
    if "risk_used" in df.columns:
        result["avg_risk"] = df["risk_used"].mean()
        result["risk_dist"] = df["risk_used"].value_counts().sort_index().to_dict()
    return result


def _build_m15c(d15m):
    return {"idx": d15m.index, "opens": d15m["open"].values,
            "highs": d15m["high"].values, "lows": d15m["low"].values}


def run_period(d15m_full, d4h_full, start, end, risk_pct, adaptive=False):
    d15m = slice_period(d15m_full, start, end)
    if d15m is None or len(d15m) == 0:
        return {}, 0.0

    spread_pr = SPREAD_PIPS * SYMBOL_CONFIG[SYM]["pip"]
    atr_15m   = calc_atr(d15m, 10).to_dict()
    m15c      = _build_m15c(d15m)
    sigs      = generate_signals(d15m, d4h_full, spread_pr, atr_15m, m15c)

    if adaptive:
        trades, final_eq, mdd, total_comm = simulate_adaptive(sigs, d15m)
    else:
        trades, final_eq, mdd, total_comm = simulate(sigs, d15m, risk_pct)

    st = calc_stats(trades)
    if st:
        st["mdd"]       = mdd
        st["final_eq"]  = final_eq
        st["commission"]= total_comm
        st["multiplier"]= final_eq / INIT_CASH
    return st, total_comm


# ── 月次分析 ─────────────────────────────────────────────────────
def monthly_pnl_adaptive(d15m_full, d4h_full):
    """アダプティブリスクで月次損益＋各月の平均リスク水準を返す"""
    d15m_oos = slice_period(d15m_full, OOS_START, OOS_END)
    if d15m_oos is None: return {}

    spread_pr = SPREAD_PIPS * SYMBOL_CONFIG[SYM]["pip"]
    atr_15m   = calc_atr(d15m_oos, 10).to_dict()
    sigs      = generate_signals(d15m_oos, d4h_full, spread_pr, atr_15m, _build_m15c(d15m_oos))

    current_risk = RISK_INIT
    rm   = RiskManager(SYM, risk_pct=current_risk)
    m15t = d15m_oos.index
    m15h = d15m_oos["high"].values
    m15l = d15m_oos["low"].values
    equity = INIT_CASH
    results = {}   # ym -> {"pnls":[], "risks":[]}

    for sig in sigs:
        rm.risk_pct = current_risk
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
        total_pnl = half_pnl + pnl

        ym = sig["time"].strftime("%Y-%m")
        results.setdefault(ym, {"pnls": [], "risks": []})
        results[ym]["pnls"].append(total_pnl)
        results[ym]["risks"].append(current_risk * 100)

        if total_pnl > 0:
            current_risk = round(min(current_risk + RISK_STEP, RISK_MAX), 4)
        else:
            current_risk = round(max(current_risk - RISK_STEP, RISK_MIN), 4)

    return results


# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*80)
    print("  NZDUSD 集中戦略バックテスト（アダプティブリスク 2〜3%, 0.5%刻み）")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  初期資産: {INIT_CASH:,}円  RR: {RR_RATIO}")
    print(f"  Raw spread: {SPREAD_PIPS}pips  手数料: {COMMISSION_USD_PER_LOT}USD/100k/片道")
    print(f"  アダプティブ: 初期{RISK_INIT*100:.0f}%  勝ち+{RISK_STEP*100:.1f}%  負け-{RISK_STEP*100:.1f}%"
          f"  [{RISK_MIN*100:.0f}%〜{RISK_MAX*100:.0f}%]")
    print(f"  固定比較: 固定2% / 固定{RISK_PCT*100:.0f}% / アダプティブ")
    print("="*80)

    d15m_full, d4h_full = load_nzdusd()
    if d15m_full is None or d4h_full is None:
        print("  ❌ データ不足")
        return

    print("  計算中...", end=" ", flush=True)
    is_2,   _ = run_period(d15m_full, d4h_full, IS_START,  IS_END,  0.02)
    is_3,   _ = run_period(d15m_full, d4h_full, IS_START,  IS_END,  0.03)
    is_ada, _ = run_period(d15m_full, d4h_full, IS_START,  IS_END,  None, adaptive=True)
    oos_2,  _ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, 0.02)
    oos_3,  _ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, 0.03)
    oos_ada,_ = run_period(d15m_full, d4h_full, OOS_START, OOS_END, None, adaptive=True)
    print("完了\n")

    # ── 3戦略比較テーブル ──────────────────────────────────────────
    W = 14
    hdr = f"  {'':20} {'固定2%':>{W}} {'固定3%':>{W}} {'アダプティブ':>{W}}"
    sep = "  " + "-" * (20 + W*3 + 2)

    def row(label, is_fn, oos_fn):
        vals = []
        for st_is, st_oos in [(is_2, oos_2), (is_3, oos_3), (is_ada, oos_ada)]:
            iv = is_fn(st_is)  if st_is  else "N/A"
            ov = oos_fn(st_oos) if st_oos else "N/A"
            vals.append(f"{iv}/{ov}")
        print(f"  {label:20} {vals[0]:>{W}} {vals[1]:>{W}} {vals[2]:>{W}}")

    print(hdr)
    print(f"  {'':20} {'IS / OOS':>{W}} {'IS / OOS':>{W}} {'IS / OOS':>{W}}")
    print(sep)
    row("トレード数",
        lambda s: f"{s['n']}",
        lambda s: f"{s['n']}")
    row("勝率",
        lambda s: f"{s['wr']*100:.1f}%",
        lambda s: f"{s['wr']*100:.1f}%")
    row("PF",
        lambda s: f"{s['pf']:.2f}" if s['pf']<99 else "∞",
        lambda s: f"{s['pf']:.2f}" if s['pf']<99 else "∞")
    row("MDD",
        lambda s: f"{s['mdd']:.1f}%",
        lambda s: f"{s['mdd']:.1f}%")
    row("資産倍率",
        lambda s: f"{s['multiplier']:.2f}x",
        lambda s: f"{s['multiplier']:.2f}x")
    row("最終資産(万円)",
        lambda s: f"{s['final_eq']/10000:.0f}万",
        lambda s: f"{s['final_eq']/10000:.0f}万")
    row("手数料合計",
        lambda s: f"{s['commission']/10000:.0f}万",
        lambda s: f"{s['commission']/10000:.0f}万")

    # アダプティブの平均リスク
    if oos_ada and oos_ada.get("avg_risk"):
        print(f"\n  【アダプティブ OOS 平均リスク水準: {oos_ada['avg_risk']*100:.2f}%】")
        if oos_ada.get("risk_dist"):
            print("  リスク分布（リスク水準: トレード数）:")
            for rv, cnt in sorted(oos_ada["risk_dist"].items()):
                bar = "█" * min(int(cnt/10)+1, 30)
                print(f"    {rv*100:.1f}%: {cnt:>4}件  {bar}")

    # IS/OOS 乖離チェック
    print("\n" + "-"*80)
    print("  ■ 過学習チェック（OOS PF / IS PF）")
    for label, is_st, oos_st in [("固定2%", is_2, oos_2), ("固定3%", is_3, oos_3),
                                   ("アダプティブ", is_ada, oos_ada)]:
        if is_st and oos_st and is_st.get("pf") and oos_st.get("pf"):
            r = oos_st["pf"] / is_st["pf"]
            flag = "✅ 過学習なし" if r >= 0.7 else "⚠️ 過学習疑い"
            print(f"    {label:12}: OOS/IS={r:.2f}  {flag}")

    # ── OOS 月次内訳（アダプティブ） ────────────────────────────────
    print("\n" + "-"*80)
    print("  ■ OOS 月次損益（アダプティブリスク）")
    print(f"  {'月':>8} {'損益':>12} {'件数':>6} {'平均R%':>8}  判定")
    print(f"  {'-'*50}")
    monthly = monthly_pnl_adaptive(d15m_full, d4h_full)
    plus_months = 0; total_months = 0
    for ym in sorted(monthly.keys()):
        data  = monthly[ym]
        total = sum(data["pnls"])
        cnt   = len(data["pnls"])
        avgr  = sum(data["risks"]) / cnt if cnt else 0
        sign  = "+" if total >= 0 else ""
        flag  = "✅" if total >= 0 else "❌"
        print(f"  {ym:>8}: {sign}{total:>10,.0f}円  {cnt:>4}件  {avgr:>6.1f}%  {flag}")
        total_months += 1
        if total >= 0: plus_months += 1
    if total_months > 0:
        print(f"  プラス月: {plus_months}/{total_months}  ({plus_months/total_months*100:.0f}%)")

    # ── サマリー ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  ■ OOS 最終サマリー")
    for label, st in [("固定2%", oos_2), ("固定3%", oos_3), ("アダプティブ", oos_ada)]:
        if not st: continue
        pf_s  = f"{st['pf']:.2f}" if st['pf'] < 99 else "∞"
        judge = "✅" if st["pf"] >= 2.0 and st["mdd"] <= 35 else "⚠️"
        print(f"    {label:12}: PF={pf_s}  MDD={st['mdd']:.1f}%  "
              f"倍率={st['multiplier']:.1f}x  最終={st['final_eq']/10000:.0f}万円  {judge}")


if __name__ == "__main__":
    main()
