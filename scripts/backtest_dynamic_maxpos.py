"""
backtest_dynamic_maxpos.py
===========================
資産規模連動型ポジション上限バックテスト

動的上限ルール:
  資産 < 500万円  → 最大10件
  500万 ≤ 資産 < 1000万円 → 最大15件
  資産 ≥ 1000万円 → 最大20件

比較対象（前回結果を再利用）:
  - 固定10件
  - 固定20件
  - 動的（10→15→20）

期間: 2025-09-01 〜 2026-02-27
銘柄: AUDUSD / USDCHF / USDCAD / EURJPY
ロット: 総資産×2%固定（初期資産ベース）
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
START     = "2025-09-01"
END       = "2026-02-27"

PAIRS   = ["AUDUSD", "USDCHF", "USDCAD", "EURJPY"]
SYM_MAP = {"AUDUSD":"audusd","USDCHF":"usdchf","USDCAD":"usdcad","EURJPY":"eurjpy"}

# 動的上限の閾値
DYNAMIC_RULES = [
    (5_000_000,  10),   # 資産 < 500万 → 10件
    (10_000_000, 15),   # 500万 ≤ 資産 < 1000万 → 15件
    (float("inf"), 20), # 1000万以上 → 20件
]

def get_dynamic_limit(equity):
    for threshold, limit in DYNAMIC_RULES:
        if equity < threshold:
            return limit
    return 20

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

def generate_signals(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    data_1h = add_indicators(data_1h)
    signals = []; used_times = set()

    for i in range(2, len(data_4h)):
        t = data_4h.index[i]; cur = data_4h.iloc[i]
        p1 = data_4h.iloc[i-1]; p2 = data_4h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        trend = cur["trend"]; tol = atr * 0.3
        if trend == 1 and abs(p2["low"] - p1["low"]) <= tol and p1["close"] > p1["open"]:
            if not check_kmid(p1, 1) or not check_klow(p1, 1): continue
            sl = min(p2["low"], p1["low"]) - atr * 0.15
            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue
            raw_ep = eb["open"]; ep = raw_ep + spread; risk = raw_ep - sl
            if risk <= 0 or risk > atr * 3: continue
            signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl,
                            "tp": raw_ep + risk * rr_ratio, "risk": risk, "tf": "4h"})
            used_times.add(et)
        elif trend == -1 and abs(p2["high"] - p1["high"]) <= tol and p1["close"] < p1["open"]:
            if not check_kmid(p1, -1) or not check_klow(p1, -1): continue
            sl = max(p2["high"], p1["high"]) + atr * 0.15
            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue
            raw_ep = eb["open"]; ep = raw_ep - spread; risk = sl - raw_ep
            if risk <= 0 or risk > atr * 3: continue
            signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl,
                            "tp": raw_ep - risk * rr_ratio, "risk": risk, "tf": "4h"})
            used_times.add(et)

    for i in range(2, len(data_1h)):
        t = data_1h.index[i]; cur = data_1h.iloc[i]
        p1 = data_1h.iloc[i-1]; p2 = data_1h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        h4b = data_4h[data_4h.index <= t]
        if len(h4b) == 0: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l["atr"]) or pd.isna(h4l["ema20"]): continue
        trend = h4l["trend"]; h4_atr = h4l["atr"]; tol = atr * 0.3
        if trend == 1 and abs(p2["low"] - p1["low"]) <= tol and p1["close"] > p1["open"]:
            if not check_kmid(h4l, 1) or not check_klow(h4l, 1): continue
            sl = min(p2["low"], p1["low"]) - atr * 0.15
            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue
            raw_ep = eb["open"]; ep = raw_ep + spread; risk = raw_ep - sl
            if risk <= 0 or risk > h4_atr * 2: continue
            signals.append({"time": et, "dir": 1, "ep": ep, "sl": sl,
                            "tp": raw_ep + risk * rr_ratio, "risk": risk, "tf": "1h"})
            used_times.add(et)
        elif trend == -1 and abs(p2["high"] - p1["high"]) <= tol and p1["close"] < p1["open"]:
            if not check_kmid(h4l, -1) or not check_klow(h4l, -1): continue
            sl = max(p2["high"], p1["high"]) + atr * 0.15
            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue
            raw_ep = eb["open"]; ep = raw_ep - spread; risk = sl - raw_ep
            if risk <= 0 or risk > h4_atr * 2: continue
            signals.append({"time": et, "dir": -1, "ep": ep, "sl": sl,
                            "tp": raw_ep - risk * rr_ratio, "risk": risk, "tf": "1h"})
            used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()


def simulate_with_maxpos(all_signals_dict, data_1m_dict, usdjpy_1m_dict,
                          init_cash, risk_pct, half_r, max_pos_mode):
    """
    max_pos_mode: int（固定）または "dynamic"（資産連動）
    """
    rm_dict = {pair: RiskManager(pair, risk_pct=risk_pct) for pair in all_signals_dict}
    all_sigs = []
    for pair, sigs in all_signals_dict.items():
        if len(sigs) == 0: continue
        for _, row in sigs.iterrows():
            all_sigs.append({**row.to_dict(), "pair": pair})
    all_sigs.sort(key=lambda x: x["time"])

    usdjpy_rates = {}
    for pair in all_signals_dict:
        if usdjpy_1m_dict.get(pair) is not None and len(usdjpy_1m_dict[pair]) > 0:
            usdjpy_rates[pair] = usdjpy_1m_dict[pair].iloc[0]["close"]
        else:
            usdjpy_rates[pair] = 150.0

    equity = init_cash
    eq_timeline = [(pd.Timestamp(START, tz="UTC"), equity)]
    trades = []; open_positions = []

    for sig in all_sigs:
        pair = sig["pair"]; entry_time = sig["time"]

        # 決済済みポジションを精算
        still_open = []
        for pos in open_positions:
            if pos["exit_time"] <= entry_time:
                equity += pos["pnl"]
                eq_timeline.append((pos["exit_time"], equity))
                trades.append({**pos["trade"], "equity": equity})
            else:
                still_open.append(pos)
        open_positions = still_open

        # 上限取得
        if max_pos_mode == "dynamic":
            max_pos = get_dynamic_limit(equity)
        else:
            max_pos = max_pos_mode

        if len(open_positions) >= max_pos:
            continue

        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]
        d1m = data_1m_dict[pair]
        future = d1m[d1m.index > entry_time]
        if len(future) == 0: continue

        rm = rm_dict[pair]
        usdjpy_init = usdjpy_rates[pair]
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

        uj_1m = usdjpy_1m_dict.get(pair)
        usdjpy_at_exit = usdjpy_init
        if uj_1m is not None:
            uj = uj_1m[uj_1m.index <= exit_time]
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

        trade_rec = {"entry_time": entry_time, "exit_time": exit_time,
                     "pair": pair, "direction": direction,
                     "ep": ep, "sl": sl, "tp": tp,
                     "result": result, "pnl": pnl, "tf": sig["tf"]}
        open_positions.append({"exit_time": exit_time, "pair": pair, "pnl": pnl, "trade": trade_rec})

    for pos in open_positions:
        equity += pos["pnl"]
        eq_timeline.append((pos["exit_time"], equity))
        trades.append({**pos["trade"], "equity": equity})

    return trades, equity, eq_timeline


def calc_stats(trades, eq_timeline, label):
    if not trades:
        return {"label": label, "n": 0, "winrate": 0, "pf": 0,
                "return_pct": 0, "final_equity": INIT_CASH,
                "mdd_pct": 0, "monthly_plus": "", "eq_timeline": eq_timeline}
    df = pd.DataFrame(trades)
    wins = df[df["result"] == "TP"]
    n = len(df); wr = len(wins) / n if n > 0 else 0
    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss   = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    eq_df = pd.DataFrame(eq_timeline, columns=["time","equity"]).sort_values("time")
    eq_arr = eq_df["equity"].values
    peak = np.maximum.accumulate(eq_arr)
    mdd = ((eq_arr - peak) / peak).min()
    ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0]
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    monthly_plus = f"{(monthly > 0).sum()}/{len(monthly)}"
    return {"label": label, "n": n, "winrate": wr * 100, "pf": pf,
            "return_pct": ret * 100, "final_equity": eq_arr[-1],
            "mdd_pct": abs(mdd) * 100, "monthly_plus": monthly_plus,
            "eq_timeline": eq_timeline}


# ── データ読み込み ────────────────────────────────────────
print("データ読み込み中...")
data_1m_dict = {}; data_15m_dict = {}; data_4h_dict = {}
usdjpy_1m_dict = {}; sigs_dict = {}
uj_raw = try_load("usdjpy", "1m")

for pair in PAIRS:
    sym = SYM_MAP[pair]
    d1m = try_load(sym, "1m"); d15m = try_load(sym, "15m"); d4h = try_load(sym, "4h")
    if any(d is None for d in [d1m, d15m, d4h]):
        print(f"  [{pair}] SKIP"); continue
    d1m  = slice_period(d1m,  START, END)
    d15m = slice_period(d15m, START, END)
    d4h  = slice_period(d4h,  START, END)
    if len(d1m) == 0: continue
    data_1m_dict[pair] = d1m; data_15m_dict[pair] = d15m; data_4h_dict[pair] = d4h
    rm = RiskManager(pair, risk_pct=RISK_PCT)
    usdjpy_1m_dict[pair] = slice_period(uj_raw, START, END) if rm.quote_type != "A" and uj_raw is not None else None
    sigs_dict[pair] = generate_signals(d1m, d15m, d4h, rm.spread_pips, rm.pip_size, rr_ratio=RR_RATIO)
    print(f"  [{pair}] シグナル数: {len(sigs_dict[pair])}")

# ── 3パターンでシミュレーション ──────────────────────────
print("\n" + "=" * 70)
print(f"3パターン比較  {START} 〜 {END}")
print("=" * 70)

patterns = [
    ("固定10件",          10),
    ("動的（10→15→20）", "dynamic"),
    ("固定20件",          20),
]

results = {}
for label, mode in patterns:
    print(f"\n[{label}] シミュレーション中...")
    trades, final_eq, eq_tl = simulate_with_maxpos(
        sigs_dict, data_1m_dict, usdjpy_1m_dict,
        INIT_CASH, RISK_PCT, HALF_R, mode)
    stats = calc_stats(trades, eq_tl, label)
    results[label] = stats
    pd.DataFrame(trades).to_csv(
        os.path.join(OUT_DIR, f"v77_dynamic_{label.replace('（','_').replace('）','').replace('→','_')}.csv"),
        index=False)
    print(f"  件数:{stats['n']} 勝率:{stats['winrate']:.1f}% PF:{stats['pf']:.2f} "
          f"リターン:{stats['return_pct']:+.1f}% 最終資産:{stats['final_equity']/10000:.0f}万円 "
          f"MDD:{stats['mdd_pct']:.1f}% 月次+:{stats['monthly_plus']}")

# ── サマリー ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("3パターン比較サマリー")
print("=" * 70)
rows = []
for label, s in results.items():
    rows.append({"パターン": label, "件数": s["n"], "勝率%": f"{s['winrate']:.1f}",
                 "PF": f"{s['pf']:.2f}", "リターン%": f"{s['return_pct']:+.1f}",
                 "最終資産(万円)": f"{s['final_equity']/10000:.0f}",
                 "MDD%": f"{s['mdd_pct']:.1f}", "月次+": s["monthly_plus"]})
df_sum = pd.DataFrame(rows)
print(df_sum.to_string(index=False))
df_sum.to_csv(os.path.join(OUT_DIR, "v77_dynamic_summary.csv"), index=False)

# ── チャート ──────────────────────────────────────────────
colors = {"固定10件": "#3b82f6", "動的（10→15→20）": "#f97316", "固定20件": "#22c55e"}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"旧ロジック◎銘柄（AUDUSD/USDCHF/USDCAD/EURJPY）\n3パターン比較  {START} 〜 {END}",
             fontsize=13, fontweight="bold")

# 左: エクイティカーブ
ax1 = axes[0]
for label, s in results.items():
    eq_df = pd.DataFrame(s["eq_timeline"], columns=["time","equity"]).sort_values("time")
    ax1.plot(eq_df["time"], eq_df["equity"] / 10000,
             label=f"{label}（最終{s['final_equity']/10000:.0f}万円）",
             color=colors[label], linewidth=2)

# 閾値ライン
ax1.axhline(500, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax1.axhline(1000, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax1.text(pd.Timestamp(START, tz="UTC"), 510, "500万（→15件に切替）", fontsize=8, color="gray")
ax1.text(pd.Timestamp(START, tz="UTC"), 1010, "1000万（→20件に切替）", fontsize=8, color="gray")

ax1.set_ylabel("資産（万円）"); ax1.set_title("エクイティカーブ")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))

# 右: 指標比較（PF / MDD / リターン）
ax2 = axes[1]
labels = list(results.keys())
x = np.arange(len(labels)); w = 0.25

pfs  = [results[l]["pf"] for l in labels]
mdds = [results[l]["mdd_pct"] for l in labels]
rets = [results[l]["return_pct"] / 100 for l in labels]  # 倍率で表示

b1 = ax2.bar(x - w, pfs,  w, label="PF", color="#3b82f6", alpha=0.85)
b2 = ax2.bar(x,     mdds, w, label="MDD (%)", color="#ef4444", alpha=0.85)
b3 = ax2.bar(x + w, [r * 10 for r in rets], w, label="リターン (×10%表示)", color="#22c55e", alpha=0.85)

for bar, v in zip(b1, pfs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#3b82f6")
for bar, v in zip(b2, mdds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#ef4444")
for bar, v in zip(b3, [r * 100 for r in rets]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"+{v:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#22c55e")

ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
ax2.set_title("PF / MDD / リターン比較")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "v77_dynamic_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n結果保存: results/v77_dynamic_comparison.png")
