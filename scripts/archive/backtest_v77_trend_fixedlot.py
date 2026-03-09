"""
backtest_v77_trend_fixedlot.py
==============================
旧ロジック（緩い条件）をトレンドフォロー戦略として固定ロットで評価

シグナル条件（旧ロジック）:
- 前2本の安値差 ≤ ATR×0.3 + 直近足が陽線（二番底）
- 前2本の高値差 ≤ ATR×0.3 + 直近足が陰線（二番天井）
- KMIDフィルターあり、KLOWフィルターあり
- クールダウンなし（毎時間エントリー可）
- 3点構造チェックなし、実体サイズ条件なし

固定ロット: 初期資産×2%リスクで計算した値を期間中固定
v77修正版（3点構造）との最終資産比較チャートも出力

期間: 2025-11-01 〜 2026-02-27（OOS期間）
モード: Hybridのみ（4H+1H）
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
KLOW_THR  = 0.0015
START     = "2025-11-01"
END       = "2026-02-27"

PAIRS = {
    "USDJPY": {"sym": "usdjpy"},
    "EURUSD": {"sym": "eurusd"},
    "GBPUSD": {"sym": "gbpusd"},
    "AUDUSD": {"sym": "audusd"},
    "USDCAD": {"sym": "usdcad"},
    "USDCHF": {"sym": "usdchf"},
    "NZDUSD": {"sym": "nzdusd"},
    "EURJPY": {"sym": "eurjpy"},
    "GBPJPY": {"sym": "gbpjpy"},
    "EURGBP": {"sym": "eurgbp"},
    "US30":   {"sym": "us30"},
    "SPX500": {"sym": "spx500"},
}

def load_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.rename(columns={ts_col: "timestamp"}).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def try_load(sym, tf):
    for name in [f"{sym}_oos_{tf}.csv", f"{sym}_{tf}.csv"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p): return load_csv(p)
    return None

def slice_period(df, start, end):
    return df[(df.index >= start) & (df.index <= end)].copy()

def calculate_atr(df, period=14):
    high_low   = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, span=20, atr_period=14):
    df = df.copy()
    df["atr"]   = calculate_atr(df, atr_period)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid(bar, direction):
    o = bar["open"]; c = bar["close"]
    return (direction == 1 and c > o) or (direction == -1 and c < o)

def check_klow(bar, direction):
    o = bar["open"]; c = bar["close"]; l = bar["low"]; h = bar["high"]
    if direction == 1:
        ratio = (min(o,c) - l) / o if o > 0 else 0
    else:
        ratio = (h - max(o,c)) / o if o > 0 else 0
    return ratio < KLOW_THR

# ── 旧ロジック: 緩い条件のシグナル生成 ───────────────────
def generate_signals_trend(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    """
    旧ロジック（トレンドフォロー型）
    - 前2本の安値/高値差がATR×0.3以内 + 方向に沿った足
    - クールダウンなし
    - 3点構造チェックなし
    """
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    data_1h = add_indicators(data_1h)

    signals = []; used_times = set()

    # ── 4Hシグナル ──
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        t   = h4_times[i]
        cur = data_4h.iloc[i]
        p1  = data_4h.iloc[i-1]
        p2  = data_4h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        trend     = cur["trend"]
        tolerance = atr * 0.3

        if trend == 1:
            if abs(p2["low"] - p1["low"]) <= tolerance and p1["close"] > p1["open"]:
                if not check_kmid(p1, 1): continue
                if not check_klow(p1, 1): continue
                sl = min(p2["low"], p1["low"]) - atr * 0.15
                m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
                if len(m1w) == 0: continue
                eb = m1w.iloc[0]; et = eb.name
                if et in used_times: continue
                raw_ep = eb["open"]
                ep = raw_ep + spread
                risk = raw_ep - sl
                if risk <= 0 or risk > atr * 3: continue
                tp = raw_ep + risk * rr_ratio
                signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
                used_times.add(et)

        elif trend == -1:
            if abs(p2["high"] - p1["high"]) <= tolerance and p1["close"] < p1["open"]:
                if not check_kmid(p1, -1): continue
                if not check_klow(p1, -1): continue
                sl = max(p2["high"], p1["high"]) + atr * 0.15
                m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
                if len(m1w) == 0: continue
                eb = m1w.iloc[0]; et = eb.name
                if et in used_times: continue
                raw_ep = eb["open"]
                ep = raw_ep - spread
                risk = sl - raw_ep
                if risk <= 0 or risk > atr * 3: continue
                tp = raw_ep - risk * rr_ratio
                signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
                used_times.add(et)

    # ── 1Hシグナル ──
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        t   = h1_times[i]
        cur = data_1h.iloc[i]
        p1  = data_1h.iloc[i-1]
        p2  = data_1h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue

        h4b = data_4h[data_4h.index <= t]
        if len(h4b) == 0: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l["atr"]) or pd.isna(h4l["ema20"]): continue
        trend  = h4l["trend"]
        h4_atr = h4l["atr"]
        tolerance = atr * 0.3

        if trend == 1:
            if abs(p2["low"] - p1["low"]) <= tolerance and p1["close"] > p1["open"]:
                if not check_kmid(h4l, 1): continue
                if not check_klow(h4l, 1): continue
                sl = min(p2["low"], p1["low"]) - atr * 0.15
                m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
                if len(m1w) == 0: continue
                eb = m1w.iloc[0]; et = eb.name
                if et in used_times: continue
                raw_ep = eb["open"]
                ep = raw_ep + spread
                risk = raw_ep - sl
                if risk <= 0 or risk > h4_atr * 2: continue
                tp = raw_ep + risk * rr_ratio
                signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                used_times.add(et)

        elif trend == -1:
            if abs(p2["high"] - p1["high"]) <= tolerance and p1["close"] < p1["open"]:
                if not check_kmid(h4l, -1): continue
                if not check_klow(h4l, -1): continue
                sl = max(p2["high"], p1["high"]) + atr * 0.15
                m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
                if len(m1w) == 0: continue
                eb = m1w.iloc[0]; et = eb.name
                if et in used_times: continue
                raw_ep = eb["open"]
                ep = raw_ep - spread
                risk = sl - raw_ep
                if risk <= 0 or risk > h4_atr * 2: continue
                tp = raw_ep - risk * rr_ratio
                signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
                used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals)

# ── 固定ロットシミュレーション ────────────────────────────
def simulate_fixed(signals, data_1m, init_cash, risk_pct, half_r, symbol, usdjpy_1m=None):
    rm = RiskManager(symbol, risk_pct=risk_pct)
    if signals is None or len(signals) == 0:
        return [], [init_cash]

    usdjpy_init = 150.0
    if usdjpy_1m is not None and len(usdjpy_1m) > 0:
        usdjpy_init = usdjpy_1m.iloc[0]["close"]

    equity = init_cash; eq_series = [equity]; trades = []

    for _, sig in signals.iterrows():
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]
        entry_time = sig["time"]
        future = data_1m[data_1m.index > entry_time]
        if len(future) == 0: continue

        lot = rm.calc_lot(init_cash, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0: continue

        half_tp = (ep + (tp - ep) * (half_r / RR_RATIO) if direction == 1
                   else ep - (ep - tp) * (half_r / RR_RATIO))
        half_done = False; sl_current = sl
        result = None; exit_time = None; exit_price = None

        for bar_time, bar in future.iterrows():
            if direction == 1:
                if bar["low"] <= sl_current:
                    result = "SL"; exit_price = sl_current; exit_time = bar_time; break
                if not half_done and bar["high"] >= half_tp:
                    half_done = True; sl_current = ep
                if bar["high"] >= tp:
                    result = "TP"; exit_price = tp; exit_time = bar_time; break
            else:
                if bar["high"] >= sl_current:
                    result = "SL"; exit_price = sl_current; exit_time = bar_time; break
                if not half_done and bar["low"] <= half_tp:
                    half_done = True; sl_current = ep
                if bar["low"] <= tp:
                    result = "TP"; exit_price = tp; exit_time = bar_time; break

        if result is None:
            result = "BE" if half_done else "OPEN"
            exit_price = sl_current; exit_time = future.index[-1]
        if result == "OPEN": continue

        usdjpy_at_exit = usdjpy_init
        if usdjpy_1m is not None:
            uj = usdjpy_1m[usdjpy_1m.index <= exit_time]
            if len(uj) > 0: usdjpy_at_exit = uj.iloc[-1]["close"]

        if result == "TP":
            if half_done:
                pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=usdjpy_init, ref_price=ep)
                     + rm.calc_pnl_jpy(direction, ep, tp, lot*0.5, usdjpy_rate=usdjpy_at_exit, ref_price=ep))
            else:
                pnl = rm.calc_pnl_jpy(direction, ep, tp, lot, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        elif result == "SL":
            if half_done:
                pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=usdjpy_init, ref_price=ep)
            else:
                pnl = rm.calc_pnl_jpy(direction, ep, exit_price, lot, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        else:
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=usdjpy_init, ref_price=ep)

        equity += pnl; eq_series.append(equity)
        trades.append({"entry_time": entry_time, "exit_time": exit_time,
                       "symbol": symbol, "direction": direction,
                       "ep": ep, "sl": sl, "tp": tp,
                       "result": result, "pnl": pnl, "equity": equity, "tf": sig["tf"]})
    return trades, eq_series

def calc_stats(trades, eq, label):
    if not trades:
        return {"label": label, "n": 0, "winrate": 0, "pf": 0, "return_pct": 0,
                "return_abs": 0, "mdd_pct": 0, "monthly_plus": ""}
    df = pd.DataFrame(trades)
    wins = df[df["result"] == "TP"]
    n = len(df); wr = len(wins) / n if n > 0 else 0
    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss   = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    mdd = ((eq_arr - peak) / peak).min()
    ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0]
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    monthly_plus = f"{(monthly > 0).sum()}/{len(monthly)}"
    return {"label": label, "n": n, "winrate": wr * 100, "pf": pf,
            "return_pct": ret * 100, "return_abs": eq[-1] - eq[0],
            "mdd_pct": abs(mdd) * 100, "monthly_plus": monthly_plus}

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"旧ロジック（トレンドフォロー型）固定ロット  {START} 〜 {END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%（固定）  RR: {RR_RATIO}")
print("=" * 70)

trend_results = []; trend_trades = []; trend_eq = {}

for pair, cfg in PAIRS.items():
    sym = cfg["sym"]
    rm  = RiskManager(pair, risk_pct=RISK_PCT)
    spread = rm.spread_pips; pip = rm.pip_size

    d1m  = try_load(sym, "1m")
    d15m = try_load(sym, "15m")
    d4h  = try_load(sym, "4h")
    if any(d is None for d in [d1m, d15m, d4h]):
        print(f"  [{pair}] SKIP: データ不足"); continue

    d1m  = slice_period(d1m,  START, END)
    d15m = slice_period(d15m, START, END)
    d4h  = slice_period(d4h,  START, END)
    if len(d1m) == 0:
        print(f"  [{pair}] SKIP: 期間内データなし"); continue

    usdjpy_1m = None
    if rm.quote_type != "A":
        uj = try_load("usdjpy", "1m")
        if uj is not None: usdjpy_1m = slice_period(uj, START, END)

    sigs = generate_signals_trend(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
    trades, eq = simulate_fixed(sigs, d1m, INIT_CASH, RISK_PCT, HALF_R, pair, usdjpy_1m)
    stats = calc_stats(trades, eq, pair)
    stats["pair"] = pair; stats["n_signals"] = len(sigs)
    trend_results.append(stats); trend_eq[pair] = eq
    for t in trades: trend_trades.append(t)

    print(f"  [{pair}] {stats['n']}件(シグナル{len(sigs)}) | 勝率{stats['winrate']:.1f}% | "
          f"PF{stats['pf']:.2f} | リターン{stats['return_pct']:+.1f}% | "
          f"MDD{stats['mdd_pct']:.1f}% | 月次+{stats['monthly_plus']}")

df_trend = pd.DataFrame(trend_results)
df_trend.to_csv(os.path.join(OUT_DIR, "v77_trend_results.csv"), index=False)
pd.DataFrame(trend_trades).to_csv(os.path.join(OUT_DIR, "v77_trend_trades.csv"), index=False)

# v77修正版（固定ロット）の結果を読み込み
df_v77 = pd.read_csv(os.path.join(OUT_DIR, "v77_fixedlot_results.csv"))

# ── 比較チャート ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"トレンドフォロー型 vs v77修正版（固定ロット）\n{START} 〜 {END}", fontsize=13, fontweight="bold")

pairs_list = df_trend["pair"].tolist()
x = np.arange(len(pairs_list)); w = 0.35

# 左上: PF比較
ax1 = axes[0][0]
pf_v77   = [df_v77[df_v77["pair"]==p]["pf"].values[0] if p in df_v77["pair"].values else 0 for p in pairs_list]
pf_trend = df_trend["pf"].tolist()
ax1.bar(x - w/2, pf_v77,   w, label="v77修正版（3点構造）", color="#3b82f6", alpha=0.8)
ax1.bar(x + w/2, pf_trend, w, label="旧ロジック（トレンドフォロー）", color="#22c55e", alpha=0.8)
ax1.axhline(1.0, color="red", linestyle="--", linewidth=1)
ax1.axhline(1.5, color="orange", linestyle="--", linewidth=1)
ax1.set_xticks(x); ax1.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("PF"); ax1.set_title("プロフィットファクター比較")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

# 右上: リターン比較
ax2 = axes[0][1]
ret_v77   = [df_v77[df_v77["pair"]==p]["return_pct"].values[0] if p in df_v77["pair"].values else 0 for p in pairs_list]
ret_trend = df_trend["return_pct"].tolist()
ax2.bar(x - w/2, ret_v77,   w, label="v77修正版", color="#3b82f6", alpha=0.8)
ax2.bar(x + w/2, ret_trend, w, label="旧ロジック", color="#22c55e", alpha=0.8)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("リターン (%)"); ax2.set_title("リターン比較")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

# 左下: MDD比較
ax3 = axes[1][0]
mdd_v77   = [df_v77[df_v77["pair"]==p]["mdd_pct"].values[0] if p in df_v77["pair"].values else 0 for p in pairs_list]
mdd_trend = df_trend["mdd_pct"].tolist()
ax3.bar(x - w/2, mdd_v77,   w, label="v77修正版", color="#3b82f6", alpha=0.8)
ax3.bar(x + w/2, mdd_trend, w, label="旧ロジック", color="#22c55e", alpha=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=8)
ax3.set_ylabel("MDD (%)"); ax3.set_title("最大ドローダウン比較（小さいほど良い）")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

# 右下: 件数比較
ax4 = axes[1][1]
n_v77   = [df_v77[df_v77["pair"]==p]["n"].values[0] if p in df_v77["pair"].values else 0 for p in pairs_list]
n_trend = df_trend["n"].tolist()
ax4.bar(x - w/2, n_v77,   w, label="v77修正版", color="#3b82f6", alpha=0.8)
ax4.bar(x + w/2, n_trend, w, label="旧ロジック", color="#22c55e", alpha=0.8)
ax4.set_xticks(x); ax4.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=8)
ax4.set_ylabel("トレード件数"); ax4.set_title("トレード件数比較")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "v77_trend_vs_v77_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── サマリー ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("旧ロジック（トレンドフォロー型）全銘柄サマリー")
print("=" * 70)
print(df_trend[["pair","n","winrate","pf","return_pct","mdd_pct","monthly_plus"]].to_string(index=False))

print("\n" + "=" * 70)
print("比較: 旧ロジック vs v77修正版（固定ロット）")
print("=" * 70)
merged = df_trend[["pair","pf","return_pct","mdd_pct","n"]].rename(
    columns={"pf":"pf_trend","return_pct":"ret_trend","mdd_pct":"mdd_trend","n":"n_trend"})
comp = df_v77[["pair","pf","return_pct","mdd_pct","n"]].rename(
    columns={"pf":"pf_v77","return_pct":"ret_v77","mdd_pct":"mdd_v77","n":"n_v77"})
cmp = merged.merge(comp, on="pair")
print(cmp.to_string(index=False))

trend_winners = cmp[cmp["pf_trend"] > cmp["pf_v77"]]["pair"].tolist()
v77_winners   = cmp[cmp["pf_v77"] > cmp["pf_trend"]]["pair"].tolist()
print(f"\n旧ロジックが優位: {trend_winners}")
print(f"v77修正版が優位: {v77_winners}")
print(f"\n結果保存: results/v77_trend_vs_v77_comparison.png")
