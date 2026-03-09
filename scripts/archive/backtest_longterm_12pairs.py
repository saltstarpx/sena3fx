"""
backtest_longterm_12pairs.py
=============================
旧ロジック（トレンドフォロー型）12銘柄 長期バックテスト
期間: 2024-07-01 〜 2026-02-27（約20ヶ月）
ロット: 固定ロット（総資産×2%）
評価指標: PF / 勝率 / MDD / 月次黒字率 / 最終資産
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

INIT_CASH = 1_000_000
RISK_PCT  = 0.02
RR_RATIO  = 2.5
HALF_R    = 1.0
KLOW_THR  = 0.0015
START     = "2024-07-01"
END       = "2026-02-27"

PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "USDJPY", "EURJPY", "GBPJPY", "EURGBP", "SPX500", "US30"
]
SYM_MAP = {
    "EURUSD":"eurusd","GBPUSD":"gbpusd","AUDUSD":"audusd",
    "USDCAD":"usdcad","USDCHF":"usdchf","NZDUSD":"nzdusd",
    "USDJPY":"usdjpy","EURJPY":"eurjpy","GBPJPY":"gbpjpy",
    "EURGBP":"eurgbp","SPX500":"spx500","US30":"us30"
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
    # is + oos を結合して最長期間を確保
    candidates = [
        (f"{sym}_is_{tf}.csv", f"{sym}_oos_{tf}.csv"),
        (f"{sym}_{tf}.csv", None),
    ]
    for main, sub in candidates:
        p1 = os.path.join(DATA_DIR, main)
        if os.path.exists(p1):
            df1 = load_csv(p1)
            if sub:
                p2 = os.path.join(DATA_DIR, sub)
                if os.path.exists(p2):
                    df2 = load_csv(p2)
                    combined = pd.concat([df1, df2]).sort_index()
                    combined = combined[~combined.index.duplicated(keep="first")]
                    return combined
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


def simulate_single_fast(sigs, data_1m, usdjpy_1m, rm, init_cash, risk_pct, half_r):
    """numpy高速化版シミュレーション"""
    if len(sigs) == 0:
        return [], init_cash, [(pd.Timestamp(START, tz="UTC"), init_cash)]

    usdjpy_init = usdjpy_1m.iloc[0]["close"] if usdjpy_1m is not None and len(usdjpy_1m) > 0 else 150.0

    # 1分足をnumpy配列に変換（高速アクセス用）
    m1_times  = data_1m.index.values  # numpy datetime64
    m1_highs  = data_1m["high"].values
    m1_lows   = data_1m["low"].values

    if usdjpy_1m is not None:
        uj_times  = usdjpy_1m.index.values
        uj_closes = usdjpy_1m["close"].values
    else:
        uj_times = uj_closes = None

    equity = init_cash
    eq_timeline = [(sigs.iloc[0]["time"], equity)]
    trades = []

    for _, sig in sigs.iterrows():
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]
        entry_time = sig["time"]

        # エントリー以降のインデックスをバイナリサーチで取得
        start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
        if start_idx >= len(m1_times): continue

        lot = rm.calc_lot(init_cash, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0: continue

        half_tp = (ep + (tp - ep) * (half_r / RR_RATIO) if direction == 1
                   else ep - (ep - tp) * (half_r / RR_RATIO))

        # 1本ずつ走査（numpy配列アクセスで高速化）
        half_done = False; sl_current = sl
        result = None; exit_idx = None

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
        if result == "OPEN": continue

        exit_time = pd.Timestamp(m1_times[exit_idx], tz="UTC") if pd.Timestamp(m1_times[exit_idx]).tzinfo is None else pd.Timestamp(m1_times[exit_idx])
        exit_price = sl_current if result == "SL" else tp

        usdjpy_at_exit = usdjpy_init
        if uj_times is not None:
            uj_idx = np.searchsorted(uj_times, m1_times[exit_idx], side="right") - 1
            if uj_idx >= 0: usdjpy_at_exit = uj_closes[uj_idx]

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

        equity += pnl
        eq_timeline.append((exit_time, equity))
        trades.append({"entry_time": entry_time, "exit_time": exit_time,
                       "direction": direction, "ep": ep, "sl": sl, "tp": tp,
                       "result": result, "pnl": pnl, "equity": equity, "tf": sig["tf"]})

    return trades, equity, eq_timeline


def simulate_single(sigs, data_1m, usdjpy_1m, rm, init_cash, risk_pct, half_r):
    return simulate_single_fast(sigs, data_1m, usdjpy_1m, rm, init_cash, risk_pct, half_r)


def calc_stats(trades, eq_timeline, pair):
    if not trades:
        return {"pair": pair, "n": 0, "winrate": 0, "pf": 0,
                "return_pct": 0, "final_equity": INIT_CASH,
                "mdd_pct": 0, "monthly_plus": "0/0", "kelly": 0,
                "eq_timeline": eq_timeline}
    df = pd.DataFrame(trades)
    wins = df[df["result"] == "TP"]
    n = len(df); wr = len(wins) / n if n > 0 else 0
    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss   = abs(df[df["pnl"] < 0]["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win  = df[df["pnl"] > 0]["pnl"].mean() if len(df[df["pnl"] > 0]) > 0 else 0
    avg_loss = abs(df[df["pnl"] < 0]["pnl"].mean()) if len(df[df["pnl"] < 0]) > 0 else 1
    rr = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = wr - (1 - wr) / rr if rr > 0 else 0

    eq_df = pd.DataFrame(eq_timeline, columns=["time","equity"]).sort_values("time")
    eq_arr = eq_df["equity"].values
    peak = np.maximum.accumulate(eq_arr)
    mdd = ((eq_arr - peak) / peak).min()
    ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0]

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    monthly_plus = f"{(monthly > 0).sum()}/{len(monthly)}"

    return {"pair": pair, "n": n, "winrate": wr * 100, "pf": pf,
            "return_pct": ret * 100, "final_equity": eq_arr[-1],
            "mdd_pct": abs(mdd) * 100, "monthly_plus": monthly_plus,
            "kelly": kelly * 100, "eq_timeline": eq_timeline,
            "monthly_pnl": monthly}


# ── データ読み込み & シグナル生成 ─────────────────────────
print(f"長期バックテスト {START} 〜 {END}")
print("=" * 70)

uj_raw = try_load("usdjpy", "1m")
results = {}

for pair in PAIRS:
    sym = SYM_MAP[pair]
    d1m  = try_load(sym, "1m")
    d15m = try_load(sym, "1h")  # 1hをresampleせずそのまま使用（15mがない場合）
    d4h  = try_load(sym, "4h")

    if any(d is None for d in [d1m, d15m, d4h]):
        print(f"  [{pair}] SKIP: データ不足"); continue

    d1m  = slice_period(d1m,  START, END)
    d15m = slice_period(d15m, START, END)
    d4h  = slice_period(d4h,  START, END)
    if len(d1m) < 1000:
        print(f"  [{pair}] SKIP: データ不足（{len(d1m)}行）"); continue

    rm = RiskManager(pair, risk_pct=RISK_PCT)
    usdjpy_1m = slice_period(uj_raw, START, END) if rm.quote_type != "A" and uj_raw is not None else None

    print(f"  [{pair}] シグナル生成中... (1m:{len(d1m)}, 4h:{len(d4h)})")
    sigs = generate_signals(d1m, d15m, d4h, rm.spread_pips, rm.pip_size, rr_ratio=RR_RATIO)
    print(f"  [{pair}] シグナル:{len(sigs)}件 シミュレーション中...")

    trades, final_eq, eq_tl = simulate_single(sigs, d1m, usdjpy_1m, rm, INIT_CASH, RISK_PCT, HALF_R)
    stats = calc_stats(trades, eq_tl, pair)
    results[pair] = stats

    pd.DataFrame(trades).to_csv(
        os.path.join(OUT_DIR, f"longterm_{pair.lower()}.csv"), index=False)
    print(f"  [{pair}] 件数:{stats['n']} 勝率:{stats['winrate']:.1f}% PF:{stats['pf']:.2f} "
          f"リターン:{stats['return_pct']:+.1f}% MDD:{stats['mdd_pct']:.1f}% "
          f"月次+:{stats['monthly_plus']} ケリー:{stats['kelly']:.1f}%")

# ── サマリー ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("12銘柄 長期バックテスト サマリー（固定ロット2%）")
print("=" * 70)
rows = []
for pair, s in results.items():
    # 採用判定: PF≥1.5 かつ 月次黒字率≥60% かつ MDD≤25%
    mp_num, mp_den = map(int, s["monthly_plus"].split("/"))
    mp_rate = mp_num / mp_den if mp_den > 0 else 0
    if s["pf"] >= 1.5 and mp_rate >= 0.6 and s["mdd_pct"] <= 25:
        judge = "◎採用"
    elif s["pf"] >= 1.3 and mp_rate >= 0.5:
        judge = "○検討"
    elif s["pf"] >= 1.0:
        judge = "△保留"
    else:
        judge = "✕除外"
    rows.append({"銘柄": pair, "件数": s["n"], "勝率%": f"{s['winrate']:.1f}",
                 "PF": f"{s['pf']:.2f}", "リターン%": f"{s['return_pct']:+.1f}",
                 "最終資産(万円)": f"{s['final_equity']/10000:.0f}",
                 "MDD%": f"{s['mdd_pct']:.1f}", "月次+": s["monthly_plus"],
                 "ケリー%": f"{s['kelly']:.1f}", "判定": judge})

df_sum = pd.DataFrame(rows)
print(df_sum.to_string(index=False))
df_sum.to_csv(os.path.join(OUT_DIR, "longterm_summary.csv"), index=False)

# ── チャート ──────────────────────────────────────────────
n_pairs = len(results)
cols = 4; rows_chart = (n_pairs + cols - 1) // cols
fig, axes = plt.subplots(rows_chart, cols, figsize=(20, rows_chart * 4))
fig.suptitle(f"旧ロジック 12銘柄 長期バックテスト（固定ロット2%）\n{START} 〜 {END}",
             fontsize=14, fontweight="bold")

ax_list = axes.flatten() if rows_chart > 1 else axes.flatten()
for idx, (pair, s) in enumerate(results.items()):
    ax = ax_list[idx]
    eq_df = pd.DataFrame(s["eq_timeline"], columns=["time","equity"]).sort_values("time")
    color = "#22c55e" if s["pf"] >= 1.5 else ("#f97316" if s["pf"] >= 1.3 else "#ef4444")
    ax.plot(eq_df["time"], eq_df["equity"] / 10000, color=color, linewidth=1.5)
    ax.axhline(INIT_CASH / 10000, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{pair}  PF:{s['pf']:.2f}  MDD:{s['mdd_pct']:.1f}%\n"
                 f"月次+:{s['monthly_plus']}  {s['return_pct']:+.0f}%", fontsize=9)
    ax.set_ylabel("資産(万円)", fontsize=8); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax.tick_params(axis="x", labelsize=7)

for idx in range(len(results), len(ax_list)):
    ax_list[idx].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "longterm_12pairs_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 採用判定チャート ──────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.suptitle(f"12銘柄 採用判定（PF / MDD）  {START} 〜 {END}", fontsize=13, fontweight="bold")

pairs_sorted = sorted(results.keys(), key=lambda p: results[p]["pf"], reverse=True)
pfs  = [results[p]["pf"] for p in pairs_sorted]
mdds = [results[p]["mdd_pct"] for p in pairs_sorted]
bar_colors = []
for p in pairs_sorted:
    s = results[p]
    mp_num, mp_den = map(int, s["monthly_plus"].split("/"))
    mp_rate = mp_num / mp_den if mp_den > 0 else 0
    if s["pf"] >= 1.5 and mp_rate >= 0.6 and s["mdd_pct"] <= 25:
        bar_colors.append("#22c55e")
    elif s["pf"] >= 1.3 and mp_rate >= 0.5:
        bar_colors.append("#f97316")
    else:
        bar_colors.append("#ef4444")

ax21 = axes2[0]
bars = ax21.barh(pairs_sorted, pfs, color=bar_colors, alpha=0.85)
ax21.axvline(1.0, color="red", linestyle="--", linewidth=1)
ax21.axvline(1.5, color="orange", linestyle="--", linewidth=1)
for bar, v in zip(bars, pfs):
    ax21.text(v + 0.02, bar.get_y() + bar.get_height()/2,
              f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
ax21.set_xlabel("PF"); ax21.set_title("プロフィットファクター（緑=採用◎/橙=検討○/赤=除外）")
ax21.grid(True, alpha=0.3, axis="x")

ax22 = axes2[1]
bars2 = ax22.barh(pairs_sorted, mdds, color=bar_colors, alpha=0.85)
ax22.axvline(25, color="red", linestyle="--", linewidth=1)
for bar, v in zip(bars2, mdds):
    ax22.text(v + 0.3, bar.get_y() + bar.get_height()/2,
              f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
ax22.set_xlabel("MDD (%)"); ax22.set_title("最大ドローダウン（小さいほど良い）")
ax22.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "longterm_12pairs_selection.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n結果保存: results/longterm_12pairs_equity.png")
print(f"結果保存: results/longterm_12pairs_selection.png")
print(f"結果保存: results/longterm_summary.csv")
