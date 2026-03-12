"""
backtest_all_symbols.py
========================
全銘柄 × 3ロジック 統合バックテスト＋定量分析

【3ロジック（武器）】
  Logic-A (GOLDYAGAMI)    : 日足EMA20方向一致 + EMA距離≥ATR×1.0 + E2エントリー
  Logic-B (ADX+Streak)    : ADX≥20 + 直近4本同方向 + E1エントリー
  Logic-C (オーパーツYAGAMI): 4H EMA20のみ + 1H確認足方向 + 最初の1m足（v77ピュア）

【対象銘柄】
  FX_USD  : EURUSD / GBPUSD / AUDUSD / NZDUSD / USDCAD / USDCHF
  FX_JPY  : USDJPY / EURJPY / GBPJPY / AUDJPY / CADJPY
  FX_CROSS: EURGBP
  METALS  : XAUUSD / XAGUSD
  INDICES : SPX500

【定量分析】
  ① IS/OOS 過学習チェック  : OOS_PF / IS_PF ≥ 0.70 → PASS
  ② 勝率有意性（二項検定） : OOS WR vs 50% 帰無仮説, p < 0.05 → 有意
  ③ 月次安定性            : 月次プラス率 ≥ 80% → PASS
  ④ 総合判定              : PF≥2.0 + 過学習PASS + 月次PASS → ✅採用

【IS/OOS】
  IS:  2025-01-01 〜 2025-05-31（5ヶ月）
  OOS: 2025-06-01 〜 2026-02-28（9ヶ月）
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

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
E0_WINDOW_MIN   = 2
ADX_MIN    = 20
STREAK_MIN = 4

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_OHLC = os.path.join(BASE_DIR, "data", "ohlc")
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUT_DIR       = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 銘柄定義 ─────────────────────────────────────────────────────
SYMBOLS = [
    # FX USD
    {"name": "EURUSD",  "cat": "FX_USD"},
    {"name": "GBPUSD",  "cat": "FX_USD"},
    {"name": "AUDUSD",  "cat": "FX_USD"},
    {"name": "NZDUSD",  "cat": "FX_USD"},
    {"name": "USDCAD",  "cat": "FX_USD"},
    {"name": "USDCHF",  "cat": "FX_USD"},
    # FX JPY
    {"name": "USDJPY",  "cat": "FX_JPY"},
    {"name": "EURJPY",  "cat": "FX_JPY"},
    {"name": "GBPJPY",  "cat": "FX_JPY"},
    {"name": "AUDJPY",  "cat": "FX_JPY"},
    {"name": "CADJPY",  "cat": "FX_JPY"},
    # FX クロス
    {"name": "EURGBP",  "cat": "FX_CROSS"},
    # 貴金属
    {"name": "XAUUSD",  "cat": "METALS"},
    {"name": "XAGUSD",  "cat": "METALS"},
    # 指数
    {"name": "SPX500",  "cat": "INDICES"},
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
    sym_lower = sym.lower()

    # 1m: ohlcディレクトリ優先
    p1m = os.path.join(DATA_DIR_OHLC, f"{sym}_1m.csv")
    if not os.path.exists(p1m):
        p1m = os.path.join(DATA_DIR, f"{sym_lower}_1m.csv")
    if not os.path.exists(p1m):
        return None, None

    d1m = load_csv(p1m)

    # 4h: ohlc全期間 → IS+OOS concat → 1mリサンプル の順で試行
    p4h_all = os.path.join(DATA_DIR_OHLC, f"{sym}_4h.csv")
    p4h_is  = os.path.join(DATA_DIR, f"{sym_lower}_is_4h.csv")
    p4h_oos = os.path.join(DATA_DIR, f"{sym_lower}_oos_4h.csv")
    p4h_all2 = os.path.join(DATA_DIR, f"{sym_lower}_4h.csv")

    if os.path.exists(p4h_all):
        d4h = load_csv(p4h_all)
    elif os.path.exists(p4h_all2):
        d4h = load_csv(p4h_all2)
    elif os.path.exists(p4h_is) and os.path.exists(p4h_oos):
        d4h = pd.concat([load_csv(p4h_is), load_csv(p4h_oos)])
        d4h = d4h[~d4h.index.duplicated(keep="first")].sort_index()
    else:
        d4h = d1m.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "close"])

    return d1m, d4h

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
    tr     = calc_atr(df, 1)
    atr_w  = tr.ewm(alpha=1/n, adjust=False).mean()
    pdm_w  = plus_dm.ewm(alpha=1/n, adjust=False).mean()
    mdm_w  = minus_dm.ewm(alpha=1/n, adjust=False).mean()
    di_p   = 100 * pdm_w / atr_w.replace(0, np.nan)
    di_m   = 100 * mdm_w / atr_w.replace(0, np.nan)
    dx     = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
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
    df = df_in.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

# ── エントリー ────────────────────────────────────────────────────
def pick_e0(signal_time, spread, direction, m1c):
    """オーパーツ: 2分以内の最初の1m足（方向・スパイク除外なし）"""
    idx = m1c["idx"]
    s = idx.searchsorted(signal_time, side="left")
    e = idx.searchsorted(signal_time + pd.Timedelta(minutes=E0_WINDOW_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        return idx[i], m1c["opens"][i] + (spread if direction == 1 else -spread)
    return None, None

def pick_e1(signal_time, direction, spread, m1c):
    """ADX+Streak: 5分以内の方向一致1m足の次足"""
    idx = m1c["idx"]
    s = idx.searchsorted(signal_time, side="left")
    e = idx.searchsorted(signal_time + pd.Timedelta(minutes=E1_MAX_WAIT_MIN), side="left")
    for i in range(s, min(e, len(idx))):
        o = m1c["opens"][i]; c = m1c["closes"][i]
        if direction == 1 and c <= o: continue
        if direction ==-1 and c >= o: continue
        ni = i + 1
        if ni >= len(idx): return None, None
        return idx[ni], m1c["opens"][ni] + (spread if direction == 1 else -spread)
    return None, None

def pick_e2(signal_time, direction, spread, atr_1m_d, m1c):
    """GOLDYAGAMI: スパイク除外の3分以内始値"""
    idx = m1c["idx"]
    s = idx.searchsorted(signal_time, side="left")
    e = idx.searchsorted(signal_time + pd.Timedelta(minutes=max(2, E2_WINDOW_MIN)), side="left")
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
    o = bar["open"]; l = bar["low"]
    return (min(bar["open"], bar["close"]) - l) / o < KLOW_THR if o > 0 else False

def check_ema_dist(bar):
    d = abs(bar["close"] - bar["ema20"]); a = bar["atr"]
    return not pd.isna(a) and a > 0 and d >= a * A1_EMA_DIST_MIN

# ── シグナル生成 ──────────────────────────────────────────────────
def generate_signals(d1m, d4h_full, spread, logic, atr_1m_d, m1c):
    need_1d = (logic == "A")
    d4h, d1d = build_4h(d4h_full, need_1d=need_1d)
    d1h = build_1h(d1m)

    signals = []; used = set()
    h1_times = d1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct = h1_times[i]
        h1_p1 = d1h.iloc[i-1]; h1_p2 = d1h.iloc[i-2]
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
        # Logic-C: 追加フィルターなし

        # ── 共通フィルター ──
        if not check_kmid(h4_lat, trend): continue
        if not check_klow(h4_lat): continue
        if logic != "C" and not check_ema_dist(h4_lat): continue

        tol = atr_1h * A3_DEFAULT_TOL
        direction = trend
        if direction == 1: v1, v2 = h1_p2["low"],  h1_p1["low"]
        else:              v1, v2 = h1_p2["high"], h1_p1["high"]
        if abs(v1 - v2) > tol: continue

        # Logic-C: 1H確認足の方向チェック
        if logic == "C":
            if direction == 1 and h1_p1["close"] <= h1_p1["open"]: continue
            if direction ==-1 and h1_p1["close"] >= h1_p1["open"]: continue

        if logic == "A":
            et, ep = pick_e2(h1_ct, direction, spread, atr_1m_d, m1c)
        elif logic == "C":
            et, ep = pick_e0(h1_ct, spread, direction, m1c)
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
    df    = pd.DataFrame(trades)
    n     = len(df)
    wins  = df[df["pnl"] > 0]["pnl"]
    loses = df[df["pnl"] < 0]["pnl"]
    wr    = len(wins) / n
    gw    = wins.sum(); gl = abs(loses.sum())
    pf    = gw / gl if gl > 0 else float("inf")
    monthly = df.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    # 二項検定: WR vs 50%
    p_val = stats.binomtest(len(wins), n, 0.5, alternative="greater").pvalue
    return {"n": n, "wr": wr, "pf": pf,
            "plus_months": plus_months, "total_months": len(monthly),
            "p_val": p_val}

# ── 期間別実行 ────────────────────────────────────────────────────
def run_period(d1m_full, d4h_full, sym, logic, start, end):
    d1m = slice_period(d1m_full, start, end)
    if len(d1m) == 0: return {}
    cfg    = SYMBOL_CONFIG.get(sym, {})
    if not cfg: return {}
    spread = cfg["spread"] * cfg["pip"]
    atr_1m = calc_atr(d1m, 10).to_dict()
    m1c    = {"idx":    d1m.index,
              "opens":  d1m["open"].values,
              "closes": d1m["close"].values,
              "highs":  d1m["high"].values,
              "lows":   d1m["low"].values}
    sigs  = generate_signals(d1m, d4h_full, spread, logic, atr_1m, m1c)
    trades, final_eq, mdd, _ = simulate(sigs, d1m, sym)
    st = calc_stats(trades)
    if st: st["mdd"] = mdd; st["final_eq"] = final_eq
    return st

# ── 定量分析 ──────────────────────────────────────────────────────
def overfitting_check(is_pf, oos_pf, threshold=0.70):
    """OOS/IS PF比がthreshold以上なら過学習なし"""
    if is_pf <= 0 or is_pf == float("inf"): return "N/A"
    ratio = oos_pf / is_pf
    return "✅PASS" if ratio >= threshold else f"❌FAIL({ratio:.2f})"

def significance_label(p_val):
    if p_val < 0.001: return "***"
    if p_val < 0.01:  return "** "
    if p_val < 0.05:  return "*  "
    return "ns "

def monthly_check(plus_months, total_months, threshold=0.80):
    if total_months == 0: return "N/A"
    rate = plus_months / total_months
    return "✅PASS" if rate >= threshold else f"❌{rate:.0%}"

def adoption_verdict(pf, of_check, monthly, n):
    if n < 30: return "❌データ不足"
    if "FAIL" in of_check: return "❌過学習"
    if pf >= 2.0 and "✅" in monthly: return "✅採用候補"
    if pf >= 1.5: return "⚠️要検討"
    return "❌不採用"

# ── メイン ───────────────────────────────────────────────────────
def main():
    print("\n" + "="*100)
    print("  全銘柄 × 3ロジック 統合バックテスト＋定量分析")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  Logic-A=GOLDYAGAMI  Logic-B=ADX+Streak  Logic-C=オーパーツYAGAMI")
    print("="*100)

    all_results = []

    for sym_info in SYMBOLS:
        sym = sym_info["name"]
        if sym not in SYMBOL_CONFIG:
            print(f"\n  {sym}: SYMBOL_CONFIGに未定義 → スキップ")
            continue

        print(f"\n  [{sym_info['cat']}] {sym} データロード中...", end=" ", flush=True)
        d1m_full, d4h_full = load_all(sym)
        if d1m_full is None:
            print("1mデータ未発見 → スキップ")
            continue
        print(f"1m:{len(d1m_full):,}行 / 4h:{len(d4h_full):,}行")

        for logic in ["A", "B", "C"]:
            lname = {"A": "GOLDYAGAMI   ", "B": "ADX+Streak   ", "C": "オーパーツ   "}[logic]
            print(f"    {sym} Logic-{logic}({lname}) 計算中...", end=" ", flush=True)
            is_st  = run_period(d1m_full, d4h_full, sym, logic, IS_START,  IS_END)
            oos_st = run_period(d1m_full, d4h_full, sym, logic, OOS_START, OOS_END)
            print("完了")

            all_results.append({
                "sym": sym, "cat": sym_info["cat"], "logic": logic,
                "lname": lname.strip(),
                "is": is_st, "oos": oos_st
            })

    # ── 結果表示 ─────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ OOS 詳細結果（全銘柄 × 全ロジック）")
    print(f"  {'銘柄':8} {'ロジック':16} | {'n':>5} {'WR':>6} {'PF':>6} {'MDD':>7} {'月+':>5} | "
          f"{'IS PF':>6} {'過学習':>12} {'WR有意':>6} {'月次':>8} | {'判定':>10}")
    print("-"*110)

    for r in all_results:
        oos = r["oos"]; is_ = r["is"]
        if not oos:
            print(f"  {r['sym']:8} {r['lname']:16} | {'N/A':>5} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'N/A':>5} | "
                  f"{'N/A':>6} {'N/A':>12} {'N/A':>6} {'N/A':>8} | {'❌データなし':>10}")
            continue

        n    = oos.get("n", 0)
        wr   = oos.get("wr", 0)
        pf   = oos.get("pf", 0)
        mdd  = oos.get("mdd", 0)
        pm   = oos.get("plus_months", 0)
        tm   = oos.get("total_months", 0)
        pval = oos.get("p_val", 1.0)
        is_pf = is_.get("pf", 0) if is_ else 0

        pf_s  = f"{pf:.2f}" if pf < 99 else "∞"
        is_pf_s = f"{is_pf:.2f}" if is_pf < 99 else "∞"
        of    = overfitting_check(is_pf, pf)
        sig   = significance_label(pval)
        mc    = monthly_check(pm, tm)
        verdict = adoption_verdict(pf, of, mc, n)

        print(f"  {r['sym']:8} Logic-{r['logic']}:{r['lname']:13} | "
              f"{n:>5} {wr*100:>5.1f}% {pf_s:>6} {mdd:>6.1f}% {pm:>2}/{tm:<2} | "
              f"{is_pf_s:>6} {of:>12} {sig:>6} {mc:>8} | {verdict:>10}")

    # ── 銘柄別ベストロジック ──────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 銘柄別ベストロジック＆採用判定")
    print(f"  {'銘柄':8} {'カテゴリ':10} | {'Logic-A':>8} {'Logic-B':>8} {'Logic-C':>8} | {'ベスト':>10} {'採用?':>10}")
    print("-"*75)

    sym_summary = {}
    for r in all_results:
        sym = r["sym"]
        if sym not in sym_summary:
            sym_summary[sym] = {"cat": r["cat"], "A": {}, "B": {}, "C": {}}
        sym_summary[sym][r["logic"]] = r["oos"]

    csv_rows = []
    for sym, d in sym_summary.items():
        pfs = {lg: d[lg].get("pf", 0) if d[lg] else 0 for lg in ["A", "B", "C"]}
        best_lg = max(pfs, key=lambda x: pfs[x])
        best_pf = pfs[best_lg]
        best_oos = d[best_lg]
        best_is  = next((r["is"] for r in all_results
                         if r["sym"] == sym and r["logic"] == best_lg), {})

        of    = overfitting_check(best_is.get("pf", 0) if best_is else 0, best_pf)
        mc    = monthly_check(best_oos.get("plus_months", 0),
                              best_oos.get("total_months", 0))
        verdict = adoption_verdict(best_pf, of, mc, best_oos.get("n", 0))

        pf_str = {lg: f"{pfs[lg]:.2f}" if pfs[lg] < 99 else "∞" for lg in ["A","B","C"]}
        print(f"  {sym:8} {d['cat']:10} | "
              f"{pf_str['A']:>8} {pf_str['B']:>8} {pf_str['C']:>8} | "
              f"Logic-{best_lg}:{best_pf:.2f} {verdict:>10}")

        csv_rows.append({
            "sym": sym, "cat": d["cat"],
            "best_logic": best_lg, "best_pf": best_pf,
            "pf_A": pfs["A"], "pf_B": pfs["B"], "pf_C": pfs["C"],
            "oos_wr": best_oos.get("wr", 0),
            "oos_mdd": best_oos.get("mdd", 0),
            "plus_months": best_oos.get("plus_months", 0),
            "total_months": best_oos.get("total_months", 0),
            "p_val": best_oos.get("p_val", 1.0),
            "overfitting": of,
            "monthly_check": mc,
            "verdict": verdict,
        })

    # ── カテゴリ別集計 ────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ カテゴリ別 avg OOS PF")
    print(f"  {'カテゴリ':12} | {'Logic-A avg':>12} {'Logic-B avg':>12} {'Logic-C avg':>12} | {'採用銘柄数':>10}")
    print("-"*65)

    for cat in ["FX_USD", "FX_JPY", "FX_CROSS", "METALS", "INDICES"]:
        cat_res = {sym: d for sym, d in sym_summary.items()
                   if d["cat"] == cat}
        if not cat_res: continue
        avgs = {}
        for lg in ["A", "B", "C"]:
            pfs = [d[lg].get("pf", 0) for d in cat_res.values()
                   if d[lg] and d[lg].get("pf", 0) < 99 and d[lg].get("n", 0) >= 30]
            avgs[lg] = np.mean(pfs) if pfs else 0.0
        adopted = sum(1 for r in csv_rows if r["cat"] == cat and "採用" in r["verdict"])
        print(f"  {cat:12} | {avgs['A']:>12.2f} {avgs['B']:>12.2f} {avgs['C']:>12.2f} | {adopted:>10}銘柄")

    # ── 採用候補一覧 ──────────────────────────────────────────────
    print("\n" + "="*100)
    print("  ■ 採用候補・要検討 サマリー")
    print(f"  {'銘柄':8} {'ベストロジック':20} {'OOS PF':>8} {'WR':>7} {'MDD':>7} {'月次':>6} {'過学習':>12} {'判定':>10}")
    print("-"*90)
    for r in sorted(csv_rows, key=lambda x: -x["best_pf"]):
        if r["verdict"] not in ["✅採用候補", "⚠️要検討"]: continue
        lg_name = {"A": "GOLDYAGAMI", "B": "ADX+Streak", "C": "オーパーツ"}[r["best_logic"]]
        print(f"  {r['sym']:8} Logic-{r['best_logic']}:{lg_name:15} "
              f"{r['best_pf']:>8.2f} {r['oos_wr']*100:>6.1f}% {r['oos_mdd']:>6.1f}% "
              f"{r['plus_months']:>2}/{r['total_months']:<2} {r['overfitting']:>12} {r['verdict']:>10}")

    # ── CSV保存 ───────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "backtest_all_symbols.csv")
    pd.DataFrame(csv_rows).to_csv(out_path, index=False)

    # 詳細CSV（全ロジック分）
    detail_rows = []
    for r in all_results:
        oos = r["oos"]; is_ = r["is"]
        row = {"sym": r["sym"], "cat": r["cat"], "logic": r["logic"], "lname": r["lname"]}
        for period, d in [("is", is_), ("oos", oos)]:
            for k, v in (d if d else {}).items():
                row[f"{period}_{k}"] = v
        detail_rows.append(row)
    detail_path = os.path.join(OUT_DIR, "backtest_all_symbols_detail.csv")
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    print(f"\n  結果を保存:")
    print(f"    {out_path}")
    print(f"    {detail_path}")

if __name__ == "__main__":
    main()
