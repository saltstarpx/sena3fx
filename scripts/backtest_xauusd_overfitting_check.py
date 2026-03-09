"""
backtest_xauusd_overfitting_check.py
=======================================
XAUUSD 追加フィルター IS/OOS 過学習チェック

【検証対象】（lean_baseline に追加）
  +EMA距離（ATR×1.0）: OOSで+0.72と大幅改善 → IS期間でも同様か確認
  +ATR拡張（ボラ拡大）: OOSで+0.84の最大改善 → IS期間でも同様か確認
  +1H EMA傾き         : OOSで+0.13の小幅改善
  +A1+A2（EMA距離+傾き）: OOSで+0.61

【判定基準（過学習なし条件）】
  OOS改善 ≥ IS改善 × 0.7 → 合格（OOSで再現性あり）
  IS改善が過大でOOS乖離 → 過学習フラグ
  IS/OOS両期間でPF改善 → 最も信頼できる追加効果

【期間】
  IS:  XAUUSD IS データ（2025/1〜2025/2）
  OOS: 2025-03-03 〜 2026-02-27

【lean baseline（引き算後確定）】
  KMID + KLOW + 1D_EMA + E2エントリー
  セッション/ADX/確認足/Streak(XAUUSD) = なし
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.risk_manager import RiskManager, SYMBOL_CONFIG

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INIT_CASH   = 1_000_000
RISK_PCT    = 0.02
RR_RATIO    = 2.5
HALF_R      = 1.0
KLOW_THR    = 0.0015
USDJPY_RATE = 150.0
MAX_LOOKAHEAD = 20_000

A1_EMA_DIST_MIN   = 1.0
A2_EMA_SLOPE_BARS = 3
A4_ATR_PERIOD     = 5
A3_DEFAULT_TOL    = 0.30

E2_SPIKE_ATR_MULT = 2.0
E2_ALT_WINDOW_MIN = 3

IS_START  = "2025-01-01"
IS_END    = "2025-02-28"
OOS_START = "2025-03-03"
OOS_END   = "2026-02-27"

VARIANTS = [
    ("lean_baseline", False, False, False),
    ("+EMA_dist",     True,  False, False),
    ("+ATR_expand",   False, False, True),
    ("+1H_slope",     False, True,  False),
    ("+A1+A2",        True,  True,  False),
]


def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    ts = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.rename(columns={ts: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])


def slice_period(df, start, end):
    if df is None: return None
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()


def calc_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def build_4h(df4h, need_1d=True):
    df = df4h.copy()
    df["atr"]   = calc_atr(df, 14)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    d1 = df.resample("1D").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    d1["ema20"]   = d1["close"].ewm(span=20, adjust=False).mean()
    d1["trend1d"] = np.where(d1["close"] > d1["ema20"], 1, -1)
    return df, d1


def build_1h(data_15m):
    df = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    df["atr"]     = calc_atr(df, 14)
    df["ema20"]   = df["close"].ewm(span=20, adjust=False).mean()
    df["atr_avg"] = df["atr"].rolling(A4_ATR_PERIOD).mean()
    return df


def check_kmid(bar, direction):
    o, c = bar["open"], bar["close"]
    return (direction == 1 and c > o) or (direction == -1 and c < o)


def check_klow(bar):
    o, c, l = bar["open"], bar["close"], bar["low"]
    return (min(o,c) - l) / o < KLOW_THR if o > 0 else False


def check_ema_dist(h4_bar, direction):
    dist = abs(h4_bar["close"] - h4_bar["ema20"])
    atr  = h4_bar["atr"]
    if pd.isna(atr) or atr <= 0: return False
    return dist >= atr * A1_EMA_DIST_MIN


def check_1h_ema_slope(data_1h, signal_time, direction):
    h1b = data_1h[data_1h.index < signal_time]
    if len(h1b) < A2_EMA_SLOPE_BARS + 1: return False
    ema_vals = h1b["ema20"].iloc[-(A2_EMA_SLOPE_BARS + 1):]
    slope = ema_vals.iloc[-1] - ema_vals.iloc[0]
    return slope > 0 if direction == 1 else slope < 0


def check_atr_expand(data_1h, signal_time):
    h1b = data_1h[data_1h.index < signal_time]
    if len(h1b) < 2: return False
    latest = h1b.iloc[-1]
    atr_now = latest["atr"]; atr_avg = latest["atr_avg"]
    if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg <= 0: return False
    return atr_now > atr_avg


def pick_entry_1m(signal_time, direction, spread, atr_1m, m1_cache):
    m1_idx = m1_cache["idx"]
    start  = m1_idx.searchsorted(signal_time, side="left")
    win_min = max(2, E2_ALT_WINDOW_MIN)
    end_time = signal_time + pd.Timedelta(minutes=win_min)
    end = m1_idx.searchsorted(end_time, side="left")
    for i in range(start, min(end, len(m1_idx))):
        bar_range = m1_cache["highs"][i] - m1_cache["lows"][i]
        if atr_1m is not None:
            atr_val = atr_1m.get(m1_idx[i], np.nan)
            if not np.isnan(atr_val) and bar_range > atr_val * E2_SPIKE_ATR_MULT:
                continue
        return m1_idx[i], m1_cache["opens"][i] + (spread if direction == 1 else -spread)
    return None, None


def generate_signals(data_1m, data_15m, data_4h,
                     spread_pips, pip_size, variant_flags,
                     atr_1m=None, m1_cache=None):
    use_ema_dist, use_slope, use_atr_expand = variant_flags
    spread = spread_pips * pip_size
    data_4h, data_1d = build_4h(data_4h, need_1d=True)
    data_1h = build_1h(data_15m)

    if m1_cache is None:
        m1_cache = {
            "idx":    data_1m.index,
            "opens":  data_1m["open"].values,
            "closes": data_1m["close"].values,
            "highs":  data_1m["high"].values,
            "lows":   data_1m["low"].values,
        }

    signals    = []
    used_times = set()
    h1_times   = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index < h1_ct]
        if len(h4_before) < 2: continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)): continue

        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # 日足EMA20（XAUUSD必須）
        d1_before = data_1d[data_1d.index.normalize() < h1_ct.normalize()]
        if len(d1_before) == 0: continue
        if d1_before.iloc[-1]["trend1d"] != trend: continue

        if not check_kmid(h4_latest, trend): continue
        if not check_klow(h4_latest): continue

        if use_ema_dist and not check_ema_dist(h4_latest, trend): continue
        if use_slope    and not check_1h_ema_slope(data_1h, h1_ct, trend): continue
        if use_atr_expand and not check_atr_expand(data_1h, h1_ct): continue

        tol = atr_val * A3_DEFAULT_TOL

        for direction in [1, -1]:
            if trend != direction: continue
            if direction == 1:
                v1, v2 = h1_prev2["low"],  h1_prev1["low"]
            else:
                v1, v2 = h1_prev2["high"], h1_prev1["high"]
            if abs(v1 - v2) > tol: continue

            et, ep = pick_entry_1m(h1_ct, direction, spread, atr_1m, m1_cache)
            if et is None or et in used_times: continue

            raw = ep - spread if direction == 1 else ep + spread
            if direction == 1:
                sl   = min(v1, v2) - atr_val * 0.15
                risk = raw - sl
            else:
                sl   = max(v1, v2) + atr_val * 0.15
                risk = sl - raw

            if 0 < risk <= h4_atr * 2:
                tp = raw + direction * risk * RR_RATIO
                signals.append({
                    "time": et, "dir": direction,
                    "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                })
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return signals


def _find_exit(highs, lows, ep, sl, tp, risk, direction):
    half_price = ep + direction * risk * HALF_R
    limit = min(len(highs), MAX_LOOKAHEAD)
    for i in range(limit):
        h = highs[i]; lo = lows[i]
        if direction == 1:
            if lo <= sl:  return i, sl, "loss", False, -1
            if h  >= tp:  return i, tp, "win",  False, -1
            if h  >= half_price:
                be_sl = ep
                for j in range(i+1, limit):
                    if lows[j]  <= be_sl: return j, be_sl, "win", True, i
                    if highs[j] >= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
        else:
            if h  >= sl:  return i, sl, "loss", False, -1
            if lo <= tp:  return i, tp, "win",  False, -1
            if lo <= half_price:
                be_sl = ep
                for j in range(i+1, limit):
                    if highs[j] >= be_sl: return j, be_sl, "win", True, i
                    if lows[j]  <= tp:    return j, tp,    "win", True, i
                return -1, None, None, True, i
    return -1, None, None, False, -1


def simulate(signals, data_1m, symbol):
    if not signals: return [], [INIT_CASH]
    rm = RiskManager(symbol, risk_pct=RISK_PCT)
    equity = INIT_CASH; trades = []; eq_curve = [INIT_CASH]
    m1_times = data_1m.index
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values

    for sig in signals:
        direction = sig["dir"]; ep = sig["ep"]
        sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        lot = rm.calc_lot(equity, risk, ep, usdjpy_rate=USDJPY_RATE)

        start_pos = m1_times.searchsorted(sig["time"], side="right")
        if start_pos >= len(m1_times): continue

        exit_i, exit_price, result, half_done, half_i = _find_exit(
            m1_highs[start_pos:], m1_lows[start_pos:],
            ep, sl, tp, risk, direction
        )
        if result is None: continue

        if half_done and half_i >= 0:
            half_ep = ep + direction * risk * HALF_R
            equity += rm.calc_pnl_jpy(direction, ep, half_ep, lot*0.5, USDJPY_RATE, ep)
            remaining_lot = lot * 0.5
        else:
            remaining_lot = lot

        equity += rm.calc_pnl_jpy(direction, ep, exit_price, remaining_lot, USDJPY_RATE, ep)
        exit_time = m1_times[start_pos + exit_i]
        trades.append({
            "entry_time": sig["time"], "exit_time": exit_time,
            "dir": direction, "result": result, "equity": equity
        })
        eq_curve.append(equity)

    return trades, eq_curve


def calc_stats(trades, eq_curve, label):
    if not trades:
        return {"label": label, "n": 0, "wr": 0.0, "pf": 0.0, "mdd_pct": 0.0}
    df  = pd.DataFrame(trades)
    n   = len(df)
    wr  = (df["result"] == "win").mean()
    eq  = np.array(eq_curve)
    dlt = np.diff(eq)
    gw  = dlt[dlt > 0].sum()
    gl  = abs(dlt[dlt < 0].sum())
    pf  = gw / gl if gl > 0 else float("inf")
    peak = np.maximum.accumulate(eq)
    mdd  = abs(((eq - peak) / peak).min()) * 100
    return {"label": label, "n": n,
            "wr": round(wr*100,1), "pf": round(pf,2), "mdd_pct": round(mdd,1)}


def run_period(d1m, d15m, d4h, start, end, spread_pips, pip_size, label):
    d1m_s  = slice_period(d1m,  start, end)
    d15m_s = slice_period(d15m, start, end)
    d4h_s  = slice_period(d4h,  start, end)
    if d1m_s is None or len(d1m_s) == 0:
        return {}
    atr_1m = calc_atr(d1m_s, 10).to_dict()
    m1_cache = {
        "idx":    d1m_s.index,
        "opens":  d1m_s["open"].values,
        "closes": d1m_s["close"].values,
        "highs":  d1m_s["high"].values,
        "lows":   d1m_s["low"].values,
    }
    results = {}
    for (vlabel, use_ema_dist, use_slope, use_atr_expand) in VARIANTS:
        vflags = (use_ema_dist, use_slope, use_atr_expand)
        sigs = generate_signals(d1m_s, d15m_s, d4h_s,
                                spread_pips, pip_size, vflags,
                                atr_1m, m1_cache)
        trades, eq_curve = simulate(sigs, d1m_s, "XAUUSD")
        stats = calc_stats(trades, eq_curve, vlabel)
        stats["period"] = label
        results[vlabel] = stats
    return results


def main():
    ohlc_dir = os.path.join(DATA_DIR, "ohlc")
    sym_upper = "XAUUSD"; sym_lower = "xauusd"

    def _load(tf):
        p = os.path.join(ohlc_dir, f"{sym_upper}_{tf}.csv")
        if os.path.exists(p): return load_csv(p)
        return load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_{tf}.csv"))

    d1m  = _load("1m")
    d15m = _load("15m")
    d4h  = _load("4h")
    cfg  = SYMBOL_CONFIG.get("XAUUSD", {})
    spread_pips = cfg.get("spread", 5.2)
    pip_size    = cfg.get("pip",    0.01)

    print("XAUUSD 過学習チェック（IS / OOS 両期間比較）")
    print(f"IS:  {IS_START} 〜 {IS_END}")
    print(f"OOS: {OOS_START} 〜 {OOS_END}\n")

    print("▶ IS期間 実行中...")
    is_results = run_period(d1m, d15m, d4h, IS_START, IS_END,
                            spread_pips, pip_size, "IS")
    print("▶ OOS期間 実行中...")
    oos_results = run_period(d1m, d15m, d4h, OOS_START, OOS_END,
                             spread_pips, pip_size, "OOS")

    # 集計テーブル
    print("\n" + "="*72)
    print("  XAUUSD IS vs OOS 比較（lean_baseline との PF差）")
    print("="*72)
    print(f"\n{'バリアント':16s}  {'IS n':>6} {'IS PF':>6} {'IS diff':>8}  {'OOS n':>6} {'OOS PF':>6} {'OOS diff':>9}  判定")
    print("-"*75)

    bl_is  = is_results.get("lean_baseline", {}).get("pf", 0)
    bl_oos = oos_results.get("lean_baseline", {}).get("pf", 0)

    all_rows = []
    for (vlabel, *_) in VARIANTS:
        is_s  = is_results.get(vlabel,  {})
        oos_s = oos_results.get(vlabel, {})
        is_pf  = is_s.get("pf",  0)
        oos_pf = oos_s.get("pf", 0)
        is_n   = is_s.get("n",   0)
        oos_n  = oos_s.get("n",  0)
        is_diff  = is_pf  - bl_is
        oos_diff = oos_pf - bl_oos

        if vlabel == "lean_baseline":
            verdict = "baseline"
        else:
            # 過学習判定: OOS改善 ≥ IS改善×0.7 かつ 両期間でプラス
            if is_diff <= 0 and oos_diff > 0:
                verdict = "✅ OOS専用改善"
            elif is_diff > 0 and oos_diff > 0:
                ratio = oos_diff / is_diff if is_diff > 0 else 99
                if ratio >= 0.7:
                    verdict = f"✅ 過学習なし(ratio={ratio:.2f})"
                else:
                    verdict = f"⚠️ 過学習疑い(ratio={ratio:.2f})"
            elif is_diff > 0 and oos_diff <= 0:
                verdict = "❌ IS過学習"
            else:
                verdict = "❌ 両期間で悪化"

        s_is  = f"+{is_diff:.2f}"  if is_diff  >= 0 else f"{is_diff:.2f}"
        s_oos = f"+{oos_diff:.2f}" if oos_diff >= 0 else f"{oos_diff:.2f}"
        print(f"  {vlabel:16s}  {is_n:6d} {is_pf:6.2f} {s_is:>8}  {oos_n:6d} {oos_pf:6.2f} {s_oos:>9}  {verdict}")
        all_rows.append({
            "variant": vlabel,
            "is_n": is_n, "is_pf": is_pf, "is_diff": round(is_diff,3),
            "oos_n": oos_n, "oos_pf": oos_pf, "oos_diff": round(oos_diff,3),
            "verdict": verdict
        })

    # グラフ
    df_plot = pd.DataFrame([r for r in all_rows if r["variant"] != "lean_baseline"])
    variants_plot = df_plot["variant"].tolist()
    x = np.arange(len(variants_plot))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.bar(x - w/2, df_plot["is_pf"],  w, label="IS",  alpha=0.8, color="steelblue")
    ax1.bar(x + w/2, df_plot["oos_pf"], w, label="OOS", alpha=0.8, color="darkorange")
    ax1.axhline(bl_is,  color="steelblue",   linestyle="--", linewidth=1, label=f"baseline IS={bl_is:.2f}")
    ax1.axhline(bl_oos, color="darkorange",   linestyle="--", linewidth=1, label=f"baseline OOS={bl_oos:.2f}")
    ax1.set_title("XAUUSD IS vs OOS — PF比較", fontsize=11)
    ax1.set_ylabel("PF")
    ax1.set_xticks(x); ax1.set_xticklabels(variants_plot, rotation=25, fontsize=8)
    ax1.legend(fontsize=7)
    for xi, (ip, op) in zip(x, zip(df_plot["is_pf"], df_plot["oos_pf"])):
        ax1.text(xi-w/2, ip+0.02, f"{ip:.2f}", ha="center", va="bottom", fontsize=7, color="steelblue")
        ax1.text(xi+w/2, op+0.02, f"{op:.2f}", ha="center", va="bottom", fontsize=7, color="darkorange")

    ax2 = axes[1]
    colors_is  = ["green" if v >= 0 else "red" for v in df_plot["is_diff"]]
    colors_oos = ["green" if v >= 0 else "red" for v in df_plot["oos_diff"]]
    ax2.bar(x - w/2, df_plot["is_diff"],  w, label="IS改善",  alpha=0.8, color=colors_is)
    ax2.bar(x + w/2, df_plot["oos_diff"], w, label="OOS改善", alpha=0.8,
            color=colors_oos, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("XAUUSD IS vs OOS — lean_baseline からの改善幅\n（緑=改善/赤=悪化）", fontsize=10)
    ax2.set_ylabel("PF差（vs lean_baseline）")
    ax2.set_xticks(x); ax2.set_xticklabels(variants_plot, rotation=25, fontsize=8)
    ax2.legend(fontsize=7)
    for xi, (id_, od) in zip(x, zip(df_plot["is_diff"], df_plot["oos_diff"])):
        s_i = f"+{id_:.2f}" if id_ >= 0 else f"{id_:.2f}"
        s_o = f"+{od:.2f}" if od >= 0 else f"{od:.2f}"
        ax2.text(xi-w/2, id_+(0.01 if id_>=0 else -0.04), s_i, ha="center", va="bottom", fontsize=7)
        ax2.text(xi+w/2, od+(0.01 if od>=0 else -0.04), s_o, ha="center", va="bottom", fontsize=7)

    plt.suptitle("XAUUSD 追加フィルター 過学習チェック（IS/OOS比較）", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "xauusd_overfitting_check.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nグラフ保存: {out_png}")

    out_csv = os.path.join(OUT_DIR, "xauusd_overfitting_check.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"CSV保存:   {out_csv}")

    # 最終サマリー
    print("\n\n■ 最終判定（採用推奨）")
    for r in all_rows:
        if r["variant"] == "lean_baseline": continue
        print(f"  {r['variant']:16s}: {r['verdict']}")


if __name__ == "__main__":
    main()
