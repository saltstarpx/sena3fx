"""
backtest_longterm_v2_features.py
=================================
◎採用5銘柄 特徴量追加版バックテスト
期間: 2024-07-01 〜 2026-02-27
新規特徴量:
  F1: 時間帯フィルター（UTC 5〜15時のみ）
  F2: ストリーク: 前回負け後はスキップ
  F3: 1H足優先（4H足シグナルは1H足でも同方向確認）
比較: ベースライン vs F1 vs F1+F2 vs F1+F2+F3
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
START     = "2024-07-01"
END       = "2026-02-27"

# ◎採用5銘柄
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "EURGBP"]
SYM_MAP = {"EURUSD":"eurusd","GBPUSD":"gbpusd","AUDUSD":"audusd","USDCHF":"usdchf","EURGBP":"eurgbp"}

# 特徴量パラメータ
GOOD_HOURS = list(range(5, 16))   # UTC 5〜15時（ロンドン〜NYオーバーラップ）
BAD_HOURS  = [0,1,2,3,4,16,17,18,19,20,21]  # 期待値<-0.15の時間帯

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
    for main, sub in [(f"{sym}_is_{tf}.csv", f"{sym}_oos_{tf}.csv"), (f"{sym}_{tf}.csv", None)]:
        p1 = os.path.join(DATA_DIR, main)
        if os.path.exists(p1):
            df1 = load_csv(p1)
            if sub:
                p2 = os.path.join(DATA_DIR, sub)
                if os.path.exists(p2):
                    df2 = load_csv(p2)
                    combined = pd.concat([df1, df2]).sort_index()
                    return combined[~combined.index.duplicated(keep="first")]
            return df1
    return None

def slice_period(df, start, end):
    return df[(df.index >= start) & (df.index <= end)].copy()

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

KLOW_THR = 0.0015
def check_kmid(bar, direction):
    o = bar["open"]; c = bar["close"]
    return (direction == 1 and c > o) or (direction == -1 and c < o)

def check_klow(bar, direction):
    o = bar["open"]; c = bar["close"]; l = bar["low"]; h = bar["high"]
    ratio = (min(o,c) - l) / o if direction == 1 else (h - max(o,c)) / o
    return ratio < KLOW_THR if o > 0 else True

def generate_signals(data_1m, data_1h_raw, data_4h, spread_pips, pip_size,
                     use_hour_filter=False, use_streak=False, use_1h_priority=False):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_1h_raw.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    data_1h = add_indicators(data_1h)

    signals = []; used_times = set()

    # 4H足シグナル
    for i in range(2, len(data_4h)):
        t = data_4h.index[i]; cur = data_4h.iloc[i]
        p1 = data_4h.iloc[i-1]; p2 = data_4h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue

        # F1: 時間帯フィルター
        if use_hour_filter and t.hour not in GOOD_HOURS: continue

        trend = cur["trend"]; tol = atr * 0.3

        for direction in [1, -1]:
            if direction == 1:
                if not (trend == 1 and abs(p2["low"] - p1["low"]) <= tol and p1["close"] > p1["open"]): continue
                if not check_kmid(p1, 1) or not check_klow(p1, 1): continue
                sl = min(p2["low"], p1["low"]) - atr * 0.15
            else:
                if not (trend == -1 and abs(p2["high"] - p1["high"]) <= tol and p1["close"] < p1["open"]): continue
                if not check_kmid(p1, -1) or not check_klow(p1, -1): continue
                sl = max(p2["high"], p1["high"]) + atr * 0.15

            # F3: 1H足でも同方向確認
            if use_1h_priority:
                h1b = data_1h[data_1h.index <= t]
                if len(h1b) == 0: continue
                h1l = h1b.iloc[-1]
                if pd.isna(h1l["trend"]) or h1l["trend"] != direction: continue

            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue
            raw_ep = eb["open"]
            ep = raw_ep + spread if direction == 1 else raw_ep - spread
            risk = raw_ep - sl if direction == 1 else sl - raw_ep
            if risk <= 0 or risk > atr * 3: continue
            tp = raw_ep + risk * RR_RATIO if direction == 1 else raw_ep - risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk, "tf": "4h"})
            used_times.add(et)
            break

    # 1H足シグナル
    for i in range(2, len(data_1h)):
        t = data_1h.index[i]; cur = data_1h.iloc[i]
        p1 = data_1h.iloc[i-1]; p2 = data_1h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue

        # F1: 時間帯フィルター
        if use_hour_filter and t.hour not in GOOD_HOURS: continue

        h4b = data_4h[data_4h.index <= t]
        if len(h4b) == 0: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l["atr"]) or pd.isna(h4l["ema20"]): continue
        trend = h4l["trend"]; h4_atr = h4l["atr"]; tol = atr * 0.3

        for direction in [1, -1]:
            if direction == 1:
                if not (trend == 1 and abs(p2["low"] - p1["low"]) <= tol and p1["close"] > p1["open"]): continue
                if not check_kmid(h4l, 1) or not check_klow(h4l, 1): continue
                sl = min(p2["low"], p1["low"]) - atr * 0.15
            else:
                if not (trend == -1 and abs(p2["high"] - p1["high"]) <= tol and p1["close"] < p1["open"]): continue
                if not check_kmid(h4l, -1) or not check_klow(h4l, -1): continue
                sl = max(p2["high"], p1["high"]) + atr * 0.15

            m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
            if len(m1w) == 0: continue
            eb = m1w.iloc[0]; et = eb.name
            if et in used_times: continue
            raw_ep = eb["open"]
            ep = raw_ep + spread if direction == 1 else raw_ep - spread
            risk = raw_ep - sl if direction == 1 else sl - raw_ep
            if risk <= 0 or risk > h4_atr * 2: continue
            tp = raw_ep + risk * RR_RATIO if direction == 1 else raw_ep - risk * RR_RATIO
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk, "tf": "1h"})
            used_times.add(et)
            break

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()


def simulate_fast(sigs, data_1m, usdjpy_1m, rm, init_cash, use_streak=False):
    if len(sigs) == 0:
        return [], [(pd.Timestamp(START, tz="UTC"), init_cash)]

    usdjpy_init = usdjpy_1m.iloc[0]["close"] if usdjpy_1m is not None and len(usdjpy_1m) > 0 else 150.0
    m1_times = data_1m.index.values
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values
    uj_times = usdjpy_1m.index.values if usdjpy_1m is not None else None
    uj_closes = usdjpy_1m["close"].values if usdjpy_1m is not None else None

    equity = init_cash
    eq_timeline = [(sigs.iloc[0]["time"], equity)]
    trades = []
    last_result = None  # F2用

    for _, sig in sigs.iterrows():
        # F2: 前回負けはスキップ
        if use_streak and last_result == "SL_LOSS":
            last_result = None  # 1回だけスキップ
            continue

        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]
        entry_time = sig["time"]

        start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
        if start_idx >= len(m1_times): continue

        lot = rm.calc_lot(init_cash, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0: continue

        half_tp = (ep + (tp - ep) * (HALF_R / RR_RATIO) if direction == 1
                   else ep - (ep - tp) * (HALF_R / RR_RATIO))
        half_done = False; sl_current = sl
        result = None; exit_idx = None

        for i in range(start_idx, len(m1_times)):
            h = m1_highs[i]; l = m1_lows[i]
            if direction == 1:
                if l <= sl_current: result = "SL"; exit_idx = i; break
                if not half_done and h >= half_tp: half_done = True; sl_current = ep
                if h >= tp: result = "TP"; exit_idx = i; break
            else:
                if h >= sl_current: result = "SL"; exit_idx = i; break
                if not half_done and l <= half_tp: half_done = True; sl_current = ep
                if l <= tp: result = "TP"; exit_idx = i; break

        if result is None:
            result = "BE" if half_done else "OPEN"
            exit_idx = len(m1_times) - 1
        if result == "OPEN": continue

        exit_time = pd.Timestamp(m1_times[exit_idx])
        if exit_time.tzinfo is None: exit_time = exit_time.tz_localize("UTC")
        exit_price = sl_current if result == "SL" else tp

        usdjpy_at_exit = usdjpy_init
        if uj_times is not None:
            uj_idx = np.searchsorted(uj_times, m1_times[exit_idx], side="right") - 1
            if uj_idx >= 0: usdjpy_at_exit = uj_closes[uj_idx]

        if result == "TP":
            pnl = (rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=usdjpy_init, ref_price=ep)
                 + rm.calc_pnl_jpy(direction, ep, tp, lot*0.5, usdjpy_rate=usdjpy_at_exit, ref_price=ep)) if half_done else \
                   rm.calc_pnl_jpy(direction, ep, tp, lot, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        elif result == "SL":
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=usdjpy_init, ref_price=ep) if half_done else \
                  rm.calc_pnl_jpy(direction, ep, exit_price, lot, usdjpy_rate=usdjpy_at_exit, ref_price=ep)
        else:
            pnl = rm.calc_pnl_jpy(direction, ep, half_tp, lot*0.5, usdjpy_rate=usdjpy_init, ref_price=ep)

        # F2用: 純粋な損失（半利確なしのSL）のみ「前回負け」とする
        last_result = "SL_LOSS" if (result == "SL" and not half_done) else "OTHER"

        equity += pnl
        eq_timeline.append((exit_time, equity))
        trades.append({"entry_time": entry_time, "exit_time": exit_time,
                       "direction": direction, "result": result, "pnl": pnl,
                       "equity": equity, "tf": sig["tf"]})

    return trades, eq_timeline


def calc_stats(trades, eq_timeline, label):
    if not trades:
        return {"label": label, "n": 0, "wr": 0, "pf": 0,
                "sharpe": 0, "return_pct": 0, "mdd_pct": 0,
                "monthly_plus": "0/0", "eq_timeline": eq_timeline}
    df = pd.DataFrame(trades)
    n = len(df)
    wr = (df["result"] == "TP").sum() / n
    gp = df[df["pnl"] > 0]["pnl"].sum()
    gl = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    sharpe = df["pnl"].mean() / df["pnl"].std() * np.sqrt(252) if df["pnl"].std() > 0 else 0

    eq_df = pd.DataFrame(eq_timeline, columns=["time","equity"]).sort_values("time")
    eq_arr = eq_df["equity"].values
    peak = np.maximum.accumulate(eq_arr)
    mdd = abs(((eq_arr - peak) / peak).min()) * 100
    ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0] * 100

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    mp = f"{(monthly > 0).sum()}/{len(monthly)}"

    return {"label": label, "n": n, "wr": wr*100, "pf": pf, "sharpe": sharpe,
            "return_pct": ret, "mdd_pct": mdd, "monthly_plus": mp,
            "eq_timeline": eq_timeline, "final_equity": eq_arr[-1]}


# ── データ読み込み ────────────────────────────────────────
print(f"特徴量追加版バックテスト {START} 〜 {END}")
print("=" * 70)

uj_raw = try_load("usdjpy", "1m")
all_results = {
    "ベースライン": [],
    "F1: 時間帯フィルター": [],
    "F1+F2: +ストリーク": [],
    "F1+F2+F3: +1H優先": [],
}

for pair in PAIRS:
    sym = SYM_MAP[pair]
    d1m  = try_load(sym, "1m")
    d1h  = try_load(sym, "1h")
    d4h  = try_load(sym, "4h")
    if any(d is None for d in [d1m, d1h, d4h]):
        print(f"  [{pair}] SKIP"); continue

    d1m  = slice_period(d1m,  START, END)
    d1h  = slice_period(d1h,  START, END)
    d4h  = slice_period(d4h,  START, END)

    rm = RiskManager(pair, risk_pct=RISK_PCT)
    usdjpy_1m = slice_period(uj_raw, START, END) if rm.quote_type != "A" and uj_raw is not None else None

    print(f"  [{pair}] シグナル生成中...")

    configs = [
        ("ベースライン",         False, False, False),
        ("F1: 時間帯フィルター",  True,  False, False),
        ("F1+F2: +ストリーク",    True,  True,  False),
        ("F1+F2+F3: +1H優先",    True,  True,  True),
    ]

    for label, hf, streak, h1p in configs:
        sigs = generate_signals(d1m, d1h, d4h, rm.spread_pips, rm.pip_size,
                                use_hour_filter=hf, use_streak=streak, use_1h_priority=h1p)
        trades, eq_tl = simulate_fast(sigs, d1m, usdjpy_1m, rm, INIT_CASH, use_streak=streak)
        stats = calc_stats(trades, eq_tl, label)
        all_results[label].append(stats)
        print(f"    [{label}] n:{stats['n']} WR:{stats['wr']:.1f}% PF:{stats['pf']:.2f} "
              f"SR:{stats['sharpe']:.3f} Ret:{stats['return_pct']:+.0f}% MDD:{stats['mdd_pct']:.1f}%")

# ── 集計 ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5銘柄合算 特徴量比較サマリー")
print("=" * 70)

summary_rows = []
for label, stats_list in all_results.items():
    if not stats_list: continue
    # 全銘柄のトレードを合算してシャープを計算
    total_n = sum(s["n"] for s in stats_list)
    avg_wr  = np.mean([s["wr"] for s in stats_list])
    avg_pf  = np.mean([s["pf"] for s in stats_list if s["pf"] < 100])
    avg_sr  = np.mean([s["sharpe"] for s in stats_list])
    avg_ret = np.mean([s["return_pct"] for s in stats_list])
    avg_mdd = np.mean([s["mdd_pct"] for s in stats_list])
    summary_rows.append({"条件": label, "件数": total_n, "勝率%": f"{avg_wr:.1f}",
                         "PF": f"{avg_pf:.2f}", "シャープ": f"{avg_sr:.3f}",
                         "リターン%": f"{avg_ret:+.0f}", "MDD%": f"{avg_mdd:.1f}"})

df_sum = pd.DataFrame(summary_rows)
print(df_sum.to_string(index=False))

# ── チャート ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"特徴量追加版 シャープレシオ比較（5銘柄平均）\n{START} 〜 {END}", fontsize=13, fontweight="bold")

labels_list = list(all_results.keys())
colors_map = {"ベースライン": "#94a3b8", "F1: 時間帯フィルター": "#22c55e",
              "F1+F2: +ストリーク": "#3b82f6", "F1+F2+F3: +1H優先": "#f97316"}

# 1. シャープレシオ比較
ax = axes[0][0]
srs = [np.mean([s["sharpe"] for s in all_results[l]]) for l in labels_list]
bars = ax.bar(range(len(labels_list)), srs, color=[colors_map[l] for l in labels_list], alpha=0.85)
for bar, v in zip(bars, srs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_list)))
ax.set_xticklabels([l.replace(": ", "\n") for l in labels_list], fontsize=8)
ax.set_ylabel("シャープレシオ（年率）"); ax.set_title("シャープレシオ比較"); ax.grid(True, alpha=0.3, axis="y")

# 2. 勝率比較
ax = axes[0][1]
wrs = [np.mean([s["wr"] for s in all_results[l]]) for l in labels_list]
bars = ax.bar(range(len(labels_list)), wrs, color=[colors_map[l] for l in labels_list], alpha=0.85)
for bar, v in zip(bars, wrs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_list)))
ax.set_xticklabels([l.replace(": ", "\n") for l in labels_list], fontsize=8)
ax.set_ylabel("勝率(%)"); ax.set_title("勝率比較"); ax.grid(True, alpha=0.3, axis="y")

# 3. PF比較
ax = axes[0][2]
pfs = [np.mean([s["pf"] for s in all_results[l] if s["pf"] < 100]) for l in labels_list]
bars = ax.bar(range(len(labels_list)), pfs, color=[colors_map[l] for l in labels_list], alpha=0.85)
for bar, v in zip(bars, pfs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_list)))
ax.set_xticklabels([l.replace(": ", "\n") for l in labels_list], fontsize=8)
ax.set_ylabel("PF"); ax.set_title("プロフィットファクター比較"); ax.grid(True, alpha=0.3, axis="y")

# 4. MDD比較
ax = axes[1][0]
mdds = [np.mean([s["mdd_pct"] for s in all_results[l]]) for l in labels_list]
bars = ax.bar(range(len(labels_list)), mdds, color=[colors_map[l] for l in labels_list], alpha=0.85)
for bar, v in zip(bars, mdds):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_list)))
ax.set_xticklabels([l.replace(": ", "\n") for l in labels_list], fontsize=8)
ax.set_ylabel("MDD(%)"); ax.set_title("最大ドローダウン比較"); ax.grid(True, alpha=0.3, axis="y")

# 5. 件数比較
ax = axes[1][1]
ns = [sum(s["n"] for s in all_results[l]) for l in labels_list]
bars = ax.bar(range(len(labels_list)), ns, color=[colors_map[l] for l in labels_list], alpha=0.85)
for bar, v in zip(bars, ns):
    ax.text(bar.get_x() + bar.get_width()/2, v + 20, f"{v}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(labels_list)))
ax.set_xticklabels([l.replace(": ", "\n") for l in labels_list], fontsize=8)
ax.set_ylabel("件数"); ax.set_title("トレード件数（5銘柄合計）"); ax.grid(True, alpha=0.3, axis="y")

# 6. エクイティカーブ（GBPUSD代表）
ax = axes[1][2]
for label in labels_list:
    gbp_stats = [s for s in all_results[label] if "GBPUSD" in str(s.get("label",""))]
    # GBPUSDのeq_timelineを取得（インデックス1番目 = GBPUSD）
    pair_idx = PAIRS.index("GBPUSD")
    if pair_idx < len(all_results[label]):
        eq_tl = all_results[label][pair_idx]["eq_timeline"]
        eq_df = pd.DataFrame(eq_tl, columns=["time","equity"]).sort_values("time")
        ax.plot(eq_df["time"], eq_df["equity"]/10000, color=colors_map[label],
                linewidth=1.5, label=label, alpha=0.85)
ax.axhline(INIT_CASH/10000, color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("資産(万円)"); ax.set_title("GBPUSD エクイティカーブ比較")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
ax.tick_params(axis="x", labelsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n結果保存: results/feature_comparison.png")
df_sum.to_csv(os.path.join(OUT_DIR, "feature_comparison_summary.csv"), index=False)
print(f"結果保存: results/feature_comparison_summary.csv")
