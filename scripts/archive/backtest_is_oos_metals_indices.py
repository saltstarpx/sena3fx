"""
IS/OOS分割バックテスト: 貴金属・株価指数 F1+F3フィルター
=============================================================
対象: XAUUSD / XAGUSD / NAS100 / US30 / US500(SPX500)
戦略: F1（UTC5〜15時）+ F3（1H優先）
ロット: 固定2%  RR: 2.5  半利確: +1R → 50%決済 + SLをBEへ

IS期間:  2024-07-01 〜 2025-03-31
OOS期間: 2025-04-01 〜 2026-02-27

判定基準: OOS PF≥1.5 かつ 月次黒字≥70%

注意: XAGUSD 1m が存在しない場合は 15m でシミュレーション。
      NAS100/US30/US500 の 1h は 15m からリサンプル。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ── パラメータ ────────────────────────────────────────────
INIT_CASH  = 1_000_000
RISK_PCT   = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
KLOW_THR   = 0.0015
GOOD_HOURS = list(range(5, 16))   # UTC 5〜15時

IS_START  = "2024-07-01"
IS_END    = "2025-03-31"
OOS_START = "2025-04-01"
OOS_END   = "2026-02-27"

# 銘柄設定: {表示名: {sym: データファイル prefix, rm_sym: RiskManager銘柄名}}
PAIRS_CONFIG = {
    "XAUUSD": {"sym": "xauusd", "rm_sym": "XAUUSD"},
    "XAGUSD": {"sym": "xagusd", "rm_sym": "XAGUSD"},
    "NAS100": {"sym": "nas100", "rm_sym": "NAS100"},
    "US30":   {"sym": "us30",   "rm_sym": "US30"},
    "US500":  {"sym": "spx500", "rm_sym": "SPX500"},
}
PAIRS = list(PAIRS_CONFIG.keys())

COLORS = {
    "XAUUSD": "#d97706",   # gold
    "XAGUSD": "#6b7280",   # silver
    "NAS100": "#84cc16",   # green
    "US30":   "#f59e0b",   # amber
    "US500":  "#06b6d4",   # cyan
}

# ── データロード ──────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def load_period_data(sym, tf, period):
    """
    期間('is'/'oos')のデータを取得。
    優先順: IS/OOS分割ファイル → 全期間ファイルをスライス
    """
    p_start = IS_START if period == "is" else OOS_START
    p_end   = IS_END   if period == "is" else OOS_END

    # 1. IS/OOS 分割ファイル
    split_path = os.path.join(DATA_DIR, f"{sym}_{period}_{tf}.csv")
    if os.path.exists(split_path):
        df = load_csv(split_path)
        if df is not None and len(df) > 0:
            return df[(df.index >= p_start) & (df.index <= p_end)].copy()

    # 2. 全期間ファイルをスライス
    full_path = os.path.join(DATA_DIR, f"{sym}_{tf}.csv")
    if os.path.exists(full_path):
        df = load_csv(full_path)
        if df is not None and len(df) > 0:
            return df[(df.index >= p_start) & (df.index <= p_end)].copy()

    return None


def load_best_intraday(sym, period):
    """
    1m → 15m 優先順でイントラデイデータを取得。
    1m がなければ 15m で代用（XAGUSD など）。
    """
    for tf in ["1m", "15m"]:
        df = load_period_data(sym, tf, period)
        if df is not None and len(df) > 0:
            return df, tf
    return None, None


def load_1h_with_fallback(sym, period):
    """
    1h データを取得。なければ 15m からリサンプルして返す。
    """
    df = load_period_data(sym, "1h", period)
    if df is not None and len(df) > 0:
        return df, "1h"

    # 15m → 1h リサンプル
    df15 = load_period_data(sym, "15m", period)
    if df15 is not None and len(df15) > 0:
        vol_col = "volume" if "volume" in df15.columns else None
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if vol_col:
            agg["volume"] = "sum"
        df1h = df15.resample("1h").agg(agg).dropna(subset=["open", "close"])
        return df1h, "15m→1h"

    return None, None

# ── テクニカル指標 ────────────────────────────────────────
def calculate_atr(df, period=14):
    high_low   = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df, span=20):
    df = df.copy()
    df["atr"]   = calculate_atr(df)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df


def check_kmid(bar, direction):
    return (direction == 1 and bar["close"] > bar["open"]) or \
           (direction == -1 and bar["close"] < bar["open"])


def check_klow(bar, direction):
    o = bar["open"]; c = bar["close"]; l = bar["low"]; h = bar["high"]
    if o <= 0:
        return True
    ratio = (min(o, c) - l) / o if direction == 1 else (h - max(o, c)) / o
    return ratio < KLOW_THR

# ── シグナル生成（F1+F3ロジック）────────────────────────
def generate_signals_f1f3(data_intra, intra_tf, data_1h_raw, data_4h,
                           spread_pips, pip_size):
    """
    F1（UTC5〜15時）+ F3（1H優先）でシグナルを生成。
    data_intra: エントリー実行・SL/TP判定用のイントラデイデータ
    intra_tf:   "1m" or "15m"（エントリーウィンドウの幅を決定）
    """
    entry_window = (pd.Timedelta(minutes=2)  if intra_tf == "1m"
               else pd.Timedelta(minutes=15))

    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)

    # 1h に統一（data_1h_raw は 1h または 15m 由来）
    data_1h = data_1h_raw.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    data_1h = add_indicators(data_1h)

    signals    = []
    used_times = set()

    # ── 4H足シグナル ────────────────────────────────────────
    for i in range(2, len(data_4h)):
        t   = data_4h.index[i]
        cur = data_4h.iloc[i]
        p1  = data_4h.iloc[i - 1]
        p2  = data_4h.iloc[i - 2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        if t.hour not in GOOD_HOURS:
            continue
        trend = cur["trend"]
        tol   = atr * 0.3

        for direction in [1, -1]:
            if direction == 1:
                if not (trend == 1 and abs(p2["low"] - p1["low"]) <= tol
                        and p1["close"] > p1["open"]):
                    continue
                if not check_kmid(p1, 1) or not check_klow(p1, 1):
                    continue
                sl = min(p2["low"], p1["low"]) - atr * 0.15
            else:
                if not (trend == -1 and abs(p2["high"] - p1["high"]) <= tol
                        and p1["close"] < p1["open"]):
                    continue
                if not check_kmid(p1, -1) or not check_klow(p1, -1):
                    continue
                sl = max(p2["high"], p1["high"]) + atr * 0.15

            h1b = data_1h[data_1h.index <= t]
            if len(h1b) == 0:
                continue
            h1l = h1b.iloc[-1]
            if pd.isna(h1l["trend"]) or h1l["trend"] != direction:
                continue

            m1w = data_intra[
                (data_intra.index >= t) &
                (data_intra.index < t + entry_window)
            ]
            if len(m1w) == 0:
                continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times:
                continue

            raw_ep = eb["open"]
            ep     = raw_ep + spread if direction == 1 else raw_ep - spread
            risk   = raw_ep - sl if direction == 1 else sl - raw_ep
            if risk <= 0 or risk > atr * 3:
                continue
            tp = (raw_ep + risk * RR_RATIO if direction == 1
                  else raw_ep - risk * RR_RATIO)

            signals.append({"time": et, "dir": direction,
                             "ep": ep, "sl": sl, "tp": tp,
                             "risk": risk, "tf": "4h"})
            used_times.add(et)
            break

    # ── 1H足シグナル（F3: 1H優先） ──────────────────────────
    for i in range(2, len(data_1h)):
        t   = data_1h.index[i]
        cur = data_1h.iloc[i]
        p1  = data_1h.iloc[i - 1]
        p2  = data_1h.iloc[i - 2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0:
            continue
        if t.hour not in GOOD_HOURS:
            continue

        h4b = data_4h[data_4h.index <= t]
        if len(h4b) == 0:
            continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l["atr"]) or pd.isna(h4l["ema20"]):
            continue
        trend  = h4l["trend"]
        h4_atr = h4l["atr"]
        tol    = atr * 0.3

        for direction in [1, -1]:
            if direction == 1:
                if not (trend == 1 and abs(p2["low"] - p1["low"]) <= tol
                        and p1["close"] > p1["open"]):
                    continue
                if not check_kmid(h4l, 1) or not check_klow(h4l, 1):
                    continue
                sl = min(p2["low"], p1["low"]) - atr * 0.15
            else:
                if not (trend == -1 and abs(p2["high"] - p1["high"]) <= tol
                        and p1["close"] < p1["open"]):
                    continue
                if not check_kmid(h4l, -1) or not check_klow(h4l, -1):
                    continue
                sl = max(p2["high"], p1["high"]) + atr * 0.15

            m1w = data_intra[
                (data_intra.index >= t) &
                (data_intra.index < t + entry_window)
            ]
            if len(m1w) == 0:
                continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times:
                continue

            raw_ep = eb["open"]
            ep     = raw_ep + spread if direction == 1 else raw_ep - spread
            risk   = raw_ep - sl if direction == 1 else sl - raw_ep
            if risk <= 0 or risk > h4_atr * 2:
                continue
            tp = (raw_ep + risk * RR_RATIO if direction == 1
                  else raw_ep - risk * RR_RATIO)

            signals.append({"time": et, "dir": direction,
                             "ep": ep, "sl": sl, "tp": tp,
                             "risk": risk, "tf": "1h"})
            used_times.add(et)
            break

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()

# ── トレードシミュレーション ─────────────────────────────
def simulate_trades(sigs, data_intra, usdjpy_intra, rm, period_start):
    if len(sigs) == 0:
        return [], [(pd.Timestamp(period_start, tz="UTC"), INIT_CASH)]

    usdjpy_init = 150.0
    if usdjpy_intra is not None and len(usdjpy_intra) > 0:
        usdjpy_init = float(usdjpy_intra.iloc[0]["close"])

    m1_times  = data_intra.index.values
    m1_highs  = data_intra["high"].values
    m1_lows   = data_intra["low"].values
    uj_times  = usdjpy_intra.index.values   if usdjpy_intra is not None else None
    uj_closes = usdjpy_intra["close"].values if usdjpy_intra is not None else None

    equity      = INIT_CASH
    eq_timeline = [(sigs.iloc[0]["time"], equity)]
    trades      = []

    for _, sig in sigs.iterrows():
        ep        = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk      = sig["risk"]; direction = sig["dir"]
        entry_time = sig["time"]

        start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
        if start_idx >= len(m1_times):
            continue

        lot = rm.calc_lot(INIT_CASH, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0:
            continue

        half_tp = ((ep + (tp - ep) * (HALF_R / RR_RATIO)) if direction == 1
                   else (ep - (ep - tp) * (HALF_R / RR_RATIO)))

        half_done  = False
        sl_current = sl
        result     = None
        exit_idx   = None

        for i in range(start_idx, len(m1_times)):
            h = m1_highs[i]; l = m1_lows[i]
            if direction == 1:
                if l <= sl_current:
                    result = "SL"; exit_idx = i; break
                if not half_done and h >= half_tp:
                    half_done = True; sl_current = ep
                if h >= tp:
                    result = "TP"; exit_idx = i; break
            else:
                if h >= sl_current:
                    result = "SL"; exit_idx = i; break
                if not half_done and l <= half_tp:
                    half_done = True; sl_current = ep
                if l <= tp:
                    result = "TP"; exit_idx = i; break

        if result is None:
            result = "BE" if half_done else "OPEN"
            exit_idx = len(m1_times) - 1
        if result == "OPEN":
            continue

        exit_time = pd.Timestamp(m1_times[exit_idx])
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")

        exit_price = sl_current if result == "SL" else tp

        usdjpy_at_exit = usdjpy_init
        if uj_times is not None:
            uj_idx = np.searchsorted(uj_times, m1_times[exit_idx], side="right") - 1
            if uj_idx >= 0:
                usdjpy_at_exit = uj_closes[uj_idx]

        if result == "TP":
            pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                    usdjpy_rate=usdjpy_init, ref_price=ep) +
                   rm.calc_pnl_jpy(direction, ep, tp, lot * 0.5,
                                    usdjpy_rate=usdjpy_at_exit, ref_price=ep)
                   ) if half_done else \
                  rm.calc_pnl_jpy(direction, ep, tp, lot,
                                   usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        elif result == "SL":
            pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                    usdjpy_rate=usdjpy_init, ref_price=ep)
                   ) if half_done else \
                  rm.calc_pnl_jpy(direction, ep, exit_price, lot,
                                   usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        else:  # BE
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot * 0.5,
                                    usdjpy_rate=usdjpy_init, ref_price=ep)

        equity += pnl
        eq_timeline.append((exit_time, equity))
        trades.append({
            "entry_time": entry_time, "exit_time": exit_time,
            "result": result, "pnl": pnl, "equity": equity,
            "tf": sig["tf"], "dir": direction,
        })

    return trades, eq_timeline

# ── 指標計算 ─────────────────────────────────────────────
def calc_metrics(trades, eq_timeline, period_label):
    if not trades:
        return {"period": period_label, "n": 0, "wr": 0, "pf": 0,
                "sharpe": 0, "mdd": 0, "monthly_pos": 0,
                "total_profit": 0, "final_equity": INIT_CASH}

    df = pd.DataFrame(trades)
    n  = len(df)
    wr = (df["result"] == "TP").mean() * 100
    gp = df[df["pnl"] > 0]["pnl"].sum()
    gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")

    df["exit_month"] = pd.to_datetime(df["exit_time"]).dt.to_period("M")
    monthly = df.groupby("exit_month")["pnl"].sum()
    sharpe  = (monthly.mean() / monthly.std() * np.sqrt(12)
               if len(monthly) > 1 else 0.0)

    eq_df   = pd.DataFrame(eq_timeline, columns=["time","equity"]).sort_values("time")
    eq_vals = eq_df["equity"].values
    peak    = np.maximum.accumulate(eq_vals)
    mdd     = (peak - eq_vals).max()

    monthly_pos  = (monthly > 0).sum() / len(monthly) * 100
    total_profit = eq_df["equity"].iloc[-1] - INIT_CASH
    final_equity = eq_df["equity"].iloc[-1]

    return {
        "period": period_label, "n": n, "wr": wr, "pf": pf,
        "sharpe": sharpe, "mdd": mdd, "monthly_pos": monthly_pos,
        "total_profit": total_profit, "final_equity": final_equity,
        "monthly_pnl": monthly,
    }

# ══════════════════════════════════════════════════════════
# メイン実行
# ══════════════════════════════════════════════════════════
print("=" * 65)
print("IS/OOS分割バックテスト: 貴金属・株価指数 F1+F3")
print("=" * 65)
print(f"IS期間:  {IS_START} 〜 {IS_END}")
print(f"OOS期間: {OOS_START} 〜 {OOS_END}")
print(f"銘柄: {', '.join(PAIRS)}")
print(f"ロット: 固定{RISK_PCT*100:.0f}%  RR: {RR_RATIO}  半利確: +{HALF_R}R\n")

# USDJPY（通貨換算用: 指数・貴金属はType D/B → USD/JPY必要）
uj_is  = load_period_data("usdjpy", "15m", "is")
uj_oos = load_period_data("usdjpy", "15m", "oos")

if uj_is is None:  # フォールバック: USDJPY全期間をスライス
    uj_full = load_csv(os.path.join(DATA_DIR, "usdjpy_15m.csv"))
    if uj_full is not None:
        uj_is  = uj_full[(uj_full.index >= IS_START)  & (uj_full.index <= IS_END)]
        uj_oos = uj_full[(uj_full.index >= OOS_START) & (uj_full.index <= OOS_END)]

all_results = {}

for pair in PAIRS:
    cfg = PAIRS_CONFIG[pair]
    sym    = cfg["sym"]
    rm_sym = cfg["rm_sym"]

    print(f"\n{'─'*50}")
    print(f"[{pair}] (データ prefix: {sym}, RiskManager: {rm_sym})")
    all_results[pair] = {}

    for period, p_start, p_end, uj_data in [
        ("IS",  IS_START,  IS_END,  uj_is),
        ("OOS", OOS_START, OOS_END, uj_oos),
    ]:
        # データ読み込み
        d_intra, intra_tf = load_best_intraday(sym, period.lower())
        d1h_raw, h1_src   = load_1h_with_fallback(sym, period.lower())
        d4h               = load_period_data(sym, "4h", period.lower())

        if d_intra is None:
            print(f"  [{period}] イントラデイデータ不足 → スキップ")
            continue
        if d1h_raw is None:
            print(f"  [{period}] 1Hデータ不足 → スキップ")
            continue
        if d4h is None or len(d4h) < 10:
            print(f"  [{period}] 4Hデータ不足 ({d4h is None or len(d4h)}) → スキップ")
            continue

        try:
            rm = RiskManager(rm_sym, risk_pct=RISK_PCT)
        except ValueError as e:
            print(f"  [{period}] RiskManager エラー: {e}")
            continue

        # シグナル生成
        sigs = generate_signals_f1f3(
            d_intra, intra_tf, d1h_raw, d4h,
            rm.spread_pips, rm.pip_size
        )

        # シミュレーション（USDJPYは通貨換算用）
        trades, eq_tl = simulate_trades(sigs, d_intra, uj_data, rm, p_start)

        metrics = calc_metrics(trades, eq_tl, period)
        metrics["pair"]     = pair
        metrics["intra_tf"] = intra_tf
        metrics["h1_src"]   = h1_src

        all_results[pair][period] = {
            "trades": trades, "eq_tl": eq_tl, "metrics": metrics,
        }

        pf_str = f"{metrics['pf']:.2f}" if metrics['pf'] != float("inf") else "∞"
        print(
            f"  [{period}] 件数:{metrics['n']:4d} "
            f"勝率:{metrics['wr']:5.1f}% "
            f"PF:{pf_str:>6} "
            f"シャープ:{metrics['sharpe']:6.2f} "
            f"MDD:{metrics['mdd']/10000:6.1f}万 "
            f"月次黒字:{metrics['monthly_pos']:.0f}% "
            f"[intra:{intra_tf} 1h:{h1_src}]"
        )

# ── 集計 ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("全銘柄合算サマリー")
print(f"{'='*65}")

summary = {}
for period in ["IS", "OOS"]:
    all_trades = []
    for pair in PAIRS:
        if pair not in all_results or period not in all_results[pair]:
            continue
        for t in all_results[pair][period]["trades"]:
            all_trades.append({**t, "pair": pair})

    if not all_trades:
        summary[period] = None
        continue

    df_all = pd.DataFrame(all_trades).sort_values("exit_time")
    n_all  = len(df_all)
    wr_all = (df_all["result"] == "TP").mean() * 100
    gp_all = df_all[df_all["pnl"] > 0]["pnl"].sum()
    gl_all = abs(df_all[df_all["pnl"] < 0]["pnl"].sum())
    pf_all = gp_all / gl_all if gl_all > 0 else float("inf")

    df_all["exit_month"] = pd.to_datetime(df_all["exit_time"]).dt.to_period("M")
    monthly_all = df_all.groupby("exit_month")["pnl"].sum()
    sharpe_all  = (monthly_all.mean() / monthly_all.std() * np.sqrt(12)
                   if len(monthly_all) > 1 else 0.0)
    monthly_pos = (monthly_all > 0).sum() / len(monthly_all) * 100

    df_all["cum_pnl"]      = df_all["pnl"].cumsum()
    df_all["total_equity"] = len(PAIRS) * INIT_CASH + df_all["cum_pnl"]

    eq_vals = df_all["total_equity"].values
    peak    = np.maximum.accumulate(
        np.concatenate([[len(PAIRS) * INIT_CASH], eq_vals])
    )
    mdd_all = (peak[1:] - eq_vals).max() if len(eq_vals) > 0 else 0

    total_pnl = df_all["pnl"].sum()
    final_eq  = len(PAIRS) * INIT_CASH + total_pnl

    summary[period] = {
        "n": n_all, "wr": wr_all, "pf": pf_all,
        "sharpe": sharpe_all, "mdd": mdd_all,
        "monthly_pos": monthly_pos,
        "total_pnl": total_pnl, "final_eq": final_eq,
        "monthly": monthly_all, "trades_df": df_all,
    }

    pf_str = f"{pf_all:.2f}" if pf_all != float("inf") else "∞"
    print(f"\n[{period}] 件数:{n_all:4d} 勝率:{wr_all:.1f}% PF:{pf_str:>6} "
          f"シャープ:{sharpe_all:.2f} MDD:{mdd_all/10000:.1f}万 "
          f"月次黒字:{monthly_pos:.0f}%")
    print(f"       合計損益:{total_pnl/10000:+.1f}万円  "
          f"最終資産:{final_eq/10000:.0f}万円")

# ── 判定 ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("判定結果")
print(f"{'='*65}")

oos = summary.get("OOS")
if oos is None:
    print("OOSデータなし → 判定不能")
    passed = False
else:
    pf_ok  = oos["pf"]  >= 1.5
    mpr_ok = oos["monthly_pos"] >= 70.0
    pf_s   = f"{oos['pf']:.2f}" if oos["pf"] != float("inf") else "∞"
    print(f"OOS PF:      {pf_s:>6} {'✓ PASS' if pf_ok  else '✗ FAIL'} (基準: ≥ 1.5)")
    print(f"OOS月次黒字: {oos['monthly_pos']:.0f}% {'✓ PASS' if mpr_ok else '✗ FAIL'} (基準: ≥ 70%)")
    passed = pf_ok and mpr_ok
    print(f"\n最終判定: {'★ PASS ★' if passed else '✗ FAIL'}")

# ── 銘柄別サマリーテーブル ────────────────────────────────
print(f"\n{'='*65}")
print("銘柄別サマリー")
print(f"{'='*65}")
print(f"{'銘柄':<8} {'期間':<5} {'件数':>4} {'勝率':>6} {'PF':>6} "
      f"{'シャープ':>8} {'MDD万':>7} {'月次黒':>6} {'intra'}")
print("─" * 65)
for pair in PAIRS:
    for period in ["IS", "OOS"]:
        if pair not in all_results or period not in all_results[pair]:
            continue
        m   = all_results[pair][period]["metrics"]
        pf_s = f"{m['pf']:.2f}" if m['pf'] != float("inf") else "∞"
        print(
            f"{pair:<8} {period:<5} {m['n']:>4} {m['wr']:>5.1f}% "
            f"{pf_s:>6} {m['sharpe']:>8.2f} "
            f"{m['mdd']/10000:>7.1f} {m['monthly_pos']:>5.0f}% "
            f"{m['intra_tf']}"
        )

# ══════════════════════════════════════════════════════════
# チャート生成
# ══════════════════════════════════════════════════════════
print(f"\nチャート生成中...")

for period_label in ["IS", "OOS"]:
    p_data = summary.get(period_label)
    if p_data is None:
        continue

    fig = plt.figure(figsize=(20, 14))
    p_title = "2024-07-01 〜 2025-03-31" if period_label == "IS" \
              else "2025-04-01 〜 2026-02-27"
    fig.suptitle(
        f"F1+F3バックテスト 貴金属・株価指数 [{period_label}] {p_title}\n"
        f"固定ロット2% | RR={RR_RATIO} | 半利確+{HALF_R}R",
        fontsize=13, fontweight="bold"
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # 上段: 銘柄別エクイティカーブ
    ax_pairs = fig.add_subplot(gs[0, :])
    for pair in PAIRS:
        if pair not in all_results or period_label not in all_results[pair]:
            continue
        eq_tl = all_results[pair][period_label]["eq_tl"]
        if not eq_tl:
            continue
        eq_df  = pd.DataFrame(eq_tl, columns=["time","equity"]).sort_values("time")
        m      = all_results[pair][period_label]["metrics"]
        profit = m["final_equity"] - INIT_CASH
        label  = f"{pair}  最終:{m['final_equity']/10000:.0f}万（{profit/10000:+.0f}万）"
        ax_pairs.plot(eq_df["time"], eq_df["equity"] / 10000,
                      color=COLORS[pair], linewidth=1.8, label=label, alpha=0.9)

    ax_pairs.axhline(INIT_CASH / 10000, color="gray",
                     linestyle="--", linewidth=0.8, alpha=0.6)
    ax_pairs.set_ylabel("資産（万円）", fontsize=10)
    ax_pairs.set_title(f"銘柄別エクイティカーブ [{period_label}]", fontsize=11)
    ax_pairs.legend(fontsize=8, loc="upper left", ncol=3)
    ax_pairs.grid(True, alpha=0.3)
    ax_pairs.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax_pairs.xaxis.set_major_locator(mdates.MonthLocator())
    ax_pairs.tick_params(axis="x", labelsize=8)

    # 中段左: 5銘柄合算エクイティカーブ
    ax_sum = fig.add_subplot(gs[1, 0])
    df_all = p_data["trades_df"]
    if len(df_all) > 0:
        init_total = len(PAIRS) * INIT_CASH
        ax_sum.plot(df_all["exit_time"], df_all["total_equity"] / 10000,
                    color="#1e40af", linewidth=2.0)
        ax_sum.axhline(init_total / 10000, color="gray",
                       linestyle="--", linewidth=0.8)
        ax_sum.fill_between(
            df_all["exit_time"],
            init_total / 10000, df_all["total_equity"] / 10000,
            alpha=0.15, color="#3b82f6"
        )
        final_total = df_all["total_equity"].iloc[-1]
        ax_sum.set_title(
            f"5銘柄合算  最終:{final_total/10000:.0f}万円", fontsize=10
        )
    ax_sum.set_ylabel("合計資産（万円）", fontsize=10)
    ax_sum.grid(True, alpha=0.3)
    ax_sum.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax_sum.xaxis.set_major_locator(mdates.MonthLocator())
    ax_sum.tick_params(axis="x", labelsize=8)

    # 中段右: 月次PnL（合算）
    ax_monthly = fig.add_subplot(gs[1, 1])
    monthly = p_data["monthly"]
    if len(monthly) > 0:
        cols_m = ["#22c55e" if v > 0 else "#ef4444" for v in monthly.values]
        ax_monthly.bar(range(len(monthly)), monthly.values / 10000,
                       color=cols_m, alpha=0.85)
        ax_monthly.axhline(0, color="black", linewidth=0.8)
        ax_monthly.set_xticks(range(len(monthly)))
        ax_monthly.set_xticklabels(
            [str(m) for m in monthly.index], rotation=45, fontsize=7
        )
        pos_months = (monthly > 0).sum()
        ax_monthly.set_title(
            f"月次PnL（5銘柄合算） {pos_months}/{len(monthly)}ヶ月黒字",
            fontsize=10
        )
    ax_monthly.set_ylabel("万円", fontsize=10)
    ax_monthly.grid(True, alpha=0.3, axis="y")

    # 下段: 銘柄別月次PnL（最大4銘柄）
    pairs_with_data = [
        p for p in PAIRS
        if p in all_results and period_label in all_results[p]
           and all_results[p][period_label]["trades"]
    ]
    for idx in range(min(4, len(pairs_with_data))):
        pair  = pairs_with_data[idx]
        ax    = fig.add_subplot(gs[2, idx % 2])
        df_p  = pd.DataFrame(all_results[pair][period_label]["trades"])
        df_p["exit_month"] = pd.to_datetime(df_p["exit_time"]).dt.to_period("M")
        m_pnl = df_p.groupby("exit_month")["pnl"].sum()
        cols  = ["#22c55e" if v > 0 else "#ef4444" for v in m_pnl.values]
        ax.bar(range(len(m_pnl)), m_pnl.values / 10000, color=cols, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{pair} 月次PnL", fontsize=9)
        ax.set_ylabel("万円", fontsize=8)
        ax.set_xticks(range(len(m_pnl)))
        ax.set_xticklabels(
            [str(m)[-5:] for m in m_pnl.index], rotation=45, fontsize=6
        )
        ax.grid(True, alpha=0.3, axis="y")

    s = p_data
    pf_s = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "∞"
    fig.text(
        0.5, 0.01,
        f"[{period_label}] 件数:{s['n']} | 勝率:{s['wr']:.1f}% | "
        f"PF:{pf_s} | シャープ:{s['sharpe']:.2f} | "
        f"MDD:{s['mdd']/10000:.1f}万 | 月次黒字:{s['monthly_pos']:.0f}%  "
        f"損益:{s['total_pnl']/10000:+.1f}万円",
        ha="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#dbeafe" if period_label == "IS" else "#dcfce7",
                  edgecolor="#3b82f6" if period_label == "IS" else "#22c55e",
                  alpha=0.9)
    )

    out_path = os.path.join(OUT_DIR, f"metals_indices_{period_label.lower()}_equity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out_path}")


# ── IS vs OOS 比較チャート ────────────────────────────────
if summary.get("IS") and summary.get("OOS"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("IS/OOS比較: F1+F3バックテスト（貴金属・株価指数 5銘柄合算）",
                 fontsize=13, fontweight="bold")

    p_colors = {"IS": "#3b82f6", "OOS": "#22c55e"}

    for ax, pl in zip(axes[:2], ["IS", "OOS"]):
        p    = summary[pl]
        df_a = p["trades_df"]
        if len(df_a) == 0:
            continue
        init_t = len(PAIRS) * INIT_CASH
        ax.plot(df_a["exit_time"], df_a["total_equity"] / 10000,
                color=p_colors[pl], linewidth=2.0)
        ax.axhline(init_t / 10000, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(
            df_a["exit_time"], init_t / 10000, df_a["total_equity"] / 10000,
            alpha=0.2, color=p_colors[pl]
        )
        pf_s = f"{p['pf']:.2f}" if p["pf"] != float("inf") else "∞"
        ax.set_title(
            f"[{pl}] PF:{pf_s} | 勝率:{p['wr']:.1f}% | 月次黒字:{p['monthly_pos']:.0f}%",
            fontsize=10
        )
        ax.set_ylabel("合計資産（万円）", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", labelsize=8)

    ax3 = axes[2]
    for pl, col in p_colors.items():
        m_d = summary[pl]["monthly"]
        if len(m_d) == 0:
            continue
        bar_cols = [col if v > 0 else "#ef4444" for v in m_d.values]
        ax3.bar(
            [f"{pl}:{str(k)[-5:]}" for k in m_d.index],
            m_d.values / 10000, color=bar_cols, alpha=0.75, label=pl
        )
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("月次PnL（IS+OOS）", fontsize=10)
    ax3.set_ylabel("万円", fontsize=9)
    ax3.tick_params(axis="x", labelsize=5, rotation=70)
    ax3.grid(True, alpha=0.3, axis="y")

    pf_is_s  = f"{summary['IS']['pf']:.2f}"  if summary['IS']['pf']  != float('inf') else "∞"
    pf_oos_s = f"{summary['OOS']['pf']:.2f}" if summary['OOS']['pf'] != float('inf') else "∞"
    verdict  = "★ PASS ★" if passed else "✗ FAIL"
    fig.text(
        0.5, 0.01,
        f"IS: PF={pf_is_s} / 月次黒字={summary['IS']['monthly_pos']:.0f}%  →  "
        f"OOS: PF={pf_oos_s} / 月次黒字={summary['OOS']['monthly_pos']:.0f}%  "
        f"判定: {verdict}",
        ha="center", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#d1fae5" if passed else "#fee2e2",
                  edgecolor="#10b981" if passed else "#ef4444",
                  alpha=0.9)
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out_path = os.path.join(OUT_DIR, "metals_indices_is_oos_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out_path}")

# ── CSV 出力 ──────────────────────────────────────────────
for period_label in ["IS", "OOS"]:
    all_trades = []
    for pair in PAIRS:
        if pair not in all_results or period_label not in all_results[pair]:
            continue
        for t in all_results[pair][period_label]["trades"]:
            all_trades.append({**t, "pair": pair})
    if all_trades:
        csv_path = os.path.join(OUT_DIR, f"metals_indices_{period_label.lower()}_trades.csv")
        pd.DataFrame(all_trades).sort_values("entry_time").to_csv(csv_path, index=False)
        print(f"  保存: {csv_path}")

print(f"\n{'='*65}")
print(f"完了: 最終判定 → {'★ PASS ★' if passed else '✗ FAIL'}")
print(f"{'='*65}")
