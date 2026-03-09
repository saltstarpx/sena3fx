"""
analyze_dynamic_maxpos.py
=========================
前回の全シグナルCSV（制限なし）から動的ポジション上限をシミュレーション
既存の固定10件・固定20件の結果と合わせて3パターン比較チャートを作成

動的ルール:
  資産 < 500万円  → 最大10件
  500万 ≤ 資産 < 1000万円 → 最大15件
  資産 ≥ 1000万円 → 最大20件
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

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
START     = "2025-09-01"
END       = "2026-02-27"
PAIRS     = ["AUDUSD", "USDCHF", "USDCAD", "EURJPY"]
SYM_MAP   = {"AUDUSD":"audusd","USDCHF":"usdchf","USDCAD":"usdcad","EURJPY":"eurjpy"}

DYNAMIC_RULES = [
    (5_000_000,  10),
    (10_000_000, 15),
    (float("inf"), 20),
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
    return ratio < 0.0015

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
                            "tp": raw_ep + risk * rr_ratio, "risk": risk, "tf": "4h", "pair": None})
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
                            "tp": raw_ep - risk * rr_ratio, "risk": risk, "tf": "4h", "pair": None})
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
                            "tp": raw_ep + risk * rr_ratio, "risk": risk, "tf": "1h", "pair": None})
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
                            "tp": raw_ep - risk * rr_ratio, "risk": risk, "tf": "1h", "pair": None})
            used_times.add(et)

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()


# ── 動的シミュレーション（1分足走査なし・既存PnLを再利用） ──
# 制限なしCSVには全シグナルのPnL・exit_timeが記録されている
# これを時系列でソートし、動的上限を適用してフィルタリング

print("制限なしCSVから全シグナルを読み込み中...")
df_all = pd.read_csv(os.path.join(OUT_DIR, "v77_maxpos_unlimited.csv"))
df_all["entry_time"] = pd.to_datetime(df_all["entry_time"], utc=True)
df_all["exit_time"]  = pd.to_datetime(df_all["exit_time"],  utc=True)
df_all = df_all.sort_values("entry_time").reset_index(drop=True)
print(f"  全シグナル数: {len(df_all)}")

def simulate_dynamic_from_csv(df_all, init_cash, mode):
    """
    mode: int（固定）または "dynamic"
    df_all: 制限なし版の全トレード（entry_time, exit_time, pnl 含む）
    """
    equity = init_cash
    eq_timeline = [(df_all["entry_time"].iloc[0], equity)]
    trades = []; open_positions = []  # {"exit_time": ..., "pnl": ...}

    for _, row in df_all.iterrows():
        entry_time = row["entry_time"]

        # 決済済みを精算
        still_open = []
        for pos in open_positions:
            if pos["exit_time"] <= entry_time:
                equity += pos["pnl"]
                eq_timeline.append((pos["exit_time"], equity))
                trades.append({**pos["row"], "equity": equity})
            else:
                still_open.append(pos)
        open_positions = still_open

        # 上限判定
        if mode == "dynamic":
            max_pos = get_dynamic_limit(equity)
        else:
            max_pos = mode

        if len(open_positions) >= max_pos:
            continue

        open_positions.append({
            "exit_time": row["exit_time"],
            "pnl": row["pnl"],
            "row": row.to_dict()
        })

    # 残り精算
    for pos in open_positions:
        equity += pos["pnl"]
        eq_timeline.append((pos["exit_time"], equity))
        trades.append({**pos["row"], "equity": equity})

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


# ── 3パターン実行 ─────────────────────────────────────────
print("\n" + "=" * 70)
print("3パターン比較（既存データ再利用）")
print("=" * 70)

patterns = [
    ("固定10件",          10),
    ("動的（10→15→20）", "dynamic"),
    ("固定20件",          20),
]

results = {}
for label, mode in patterns:
    trades, final_eq, eq_tl = simulate_dynamic_from_csv(df_all, INIT_CASH, mode)
    stats = calc_stats(trades, eq_tl, label)
    results[label] = stats
    print(f"[{label}] 件数:{stats['n']} 勝率:{stats['winrate']:.1f}% PF:{stats['pf']:.2f} "
          f"リターン:{stats['return_pct']:+.1f}% 最終資産:{stats['final_equity']/10000:.0f}万円 "
          f"MDD:{stats['mdd_pct']:.1f}% 月次+:{stats['monthly_plus']}")

# ── サマリー ──────────────────────────────────────────────
print("\n" + "=" * 70)
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

ax1.axhline(500,  color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax1.axhline(1000, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax1.text(pd.Timestamp("2025-09-05", tz="UTC"), 515,  "500万 → 15件", fontsize=8, color="gray")
ax1.text(pd.Timestamp("2025-09-05", tz="UTC"), 1015, "1000万 → 20件", fontsize=8, color="gray")
ax1.set_ylabel("資産（万円）"); ax1.set_title("エクイティカーブ")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))

# 右: 指標バー比較
ax2 = axes[1]
labels = list(results.keys())
x = np.arange(len(labels)); w = 0.25

pfs  = [results[l]["pf"] for l in labels]
mdds = [results[l]["mdd_pct"] for l in labels]
rets = [results[l]["return_pct"] for l in labels]
finals = [results[l]["final_equity"] / 10000 for l in labels]

b1 = ax2.bar(x - w, pfs,  w, label="PF", color="#3b82f6", alpha=0.85)
b2 = ax2.bar(x,     mdds, w, label="MDD (%)", color="#ef4444", alpha=0.85)

for bar, v in zip(b1, pfs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#3b82f6")
for bar, v in zip(b2, mdds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#ef4444")

ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
ax2.set_title("PF / MDD 比較")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")

# 最終資産をテキストで追記
for i, (label, s) in enumerate(results.items()):
    ax2.text(i, -1.5, f"最終資産\n{s['final_equity']/10000:.0f}万円",
             ha="center", va="top", fontsize=10, fontweight="bold",
             color=colors[label])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "v77_dynamic_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n結果保存: results/v77_dynamic_comparison.png")
