"""
F1+F3（時間帯フィルター + 1H優先）のエクイティカーブと合計利益を描画
前回バックテスト結果（backtest_longterm_v2_features.py）のデータを再利用
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
START     = "2024-07-01"
END       = "2026-02-27"
PAIRS     = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "EURGBP"]
SYM_MAP   = {"EURUSD":"eurusd","GBPUSD":"gbpusd","AUDUSD":"audusd","USDCHF":"usdchf","EURGBP":"eurgbp"}
GOOD_HOURS = list(range(5, 16))

COLORS = {
    "EURUSD": "#3b82f6",
    "GBPUSD": "#22c55e",
    "AUDUSD": "#f97316",
    "USDCHF": "#8b5cf6",
    "EURGBP": "#ec4899",
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
    return (direction == 1 and bar["close"] > bar["open"]) or (direction == -1 and bar["close"] < bar["open"])
def check_klow(bar, direction):
    o = bar["open"]; c = bar["close"]; l = bar["low"]; h = bar["high"]
    ratio = (min(o,c) - l) / o if direction == 1 else (h - max(o,c)) / o
    return ratio < KLOW_THR if o > 0 else True

def generate_signals_f1f3(data_1m, data_1h_raw, data_4h, spread_pips, pip_size):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_1h_raw.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    data_1h = add_indicators(data_1h)
    signals = []; used_times = set()

    for i in range(2, len(data_4h)):
        t = data_4h.index[i]; cur = data_4h.iloc[i]
        p1 = data_4h.iloc[i-1]; p2 = data_4h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        if t.hour not in GOOD_HOURS: continue
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
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
            used_times.add(et); break

    for i in range(2, len(data_1h)):
        t = data_1h.index[i]; cur = data_1h.iloc[i]
        p1 = data_1h.iloc[i-1]; p2 = data_1h.iloc[i-2]
        atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        if t.hour not in GOOD_HOURS: continue
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
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
            used_times.add(et); break

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()

def simulate_fast(sigs, data_1m, usdjpy_1m, rm):
    if len(sigs) == 0:
        return [], [(pd.Timestamp(START, tz="UTC"), INIT_CASH)]
    usdjpy_init = usdjpy_1m.iloc[0]["close"] if usdjpy_1m is not None and len(usdjpy_1m) > 0 else 150.0
    m1_times = data_1m.index.values
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values
    uj_times  = usdjpy_1m.index.values if usdjpy_1m is not None else None
    uj_closes = usdjpy_1m["close"].values if usdjpy_1m is not None else None
    equity = INIT_CASH
    eq_timeline = [(sigs.iloc[0]["time"], equity)]
    trades = []
    for _, sig in sigs.iterrows():
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]; entry_time = sig["time"]
        start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
        if start_idx >= len(m1_times): continue
        lot = rm.calc_lot(INIT_CASH, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
        if lot <= 0: continue
        half_tp = (ep + (tp - ep) * (HALF_R / RR_RATIO) if direction == 1
                   else ep - (ep - tp) * (HALF_R / RR_RATIO))
        half_done = False; sl_current = sl; result = None; exit_idx = None
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
        equity += pnl
        eq_timeline.append((exit_time, equity))
        trades.append({"entry_time": entry_time, "exit_time": exit_time,
                       "result": result, "pnl": pnl, "equity": equity})
    return trades, eq_timeline


# ── 実行 ──────────────────────────────────────────────────
uj_raw = try_load("usdjpy", "1m")
pair_results = {}

for pair in PAIRS:
    sym = SYM_MAP[pair]
    d1m = try_load(sym, "1m")
    d1h = try_load(sym, "1h")
    d4h = try_load(sym, "4h")
    if any(d is None for d in [d1m, d1h, d4h]): continue
    d1m = slice_period(d1m, START, END)
    d1h = slice_period(d1h, START, END)
    d4h = slice_period(d4h, START, END)
    rm  = RiskManager(pair, risk_pct=RISK_PCT)
    usdjpy_1m = slice_period(uj_raw, START, END) if rm.quote_type != "A" and uj_raw is not None else None
    sigs = generate_signals_f1f3(d1m, d1h, d4h, rm.spread_pips, rm.pip_size)
    trades, eq_tl = simulate_fast(sigs, d1m, usdjpy_1m, rm)
    pair_results[pair] = {"trades": trades, "eq_tl": eq_tl}
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    wr = (df_t["result"] == "TP").mean() * 100 if n > 0 else 0
    gp = df_t[df_t["pnl"] > 0]["pnl"].sum()
    gl = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 0
    final = eq_tl[-1][1] if eq_tl else INIT_CASH
    profit = final - INIT_CASH
    sr = df_t["pnl"].mean() / df_t["pnl"].std() * np.sqrt(252) if n > 0 and df_t["pnl"].std() > 0 else 0
    print(f"[{pair}] n:{n} WR:{wr:.1f}% PF:{pf:.2f} SR:{sr:.3f} "
          f"最終:{final/10000:.0f}万円 利益:{profit/10000:.0f}万円")

# ── 合計利益 ──────────────────────────────────────────────
total_profit = sum(r["eq_tl"][-1][1] - INIT_CASH for r in pair_results.values() if r["eq_tl"])
total_final  = sum(r["eq_tl"][-1][1] for r in pair_results.values() if r["eq_tl"])
print(f"\n5銘柄合計利益（固定ロット2%・各100万円スタート）: {total_profit/10000:.0f}万円")
print(f"5銘柄合計最終資産: {total_final/10000:.0f}万円")

# ── チャート ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(f"F1+F3（時間帯フィルター+1H優先）エクイティカーブ\n"
             f"2024-07-01 〜 2026-02-27  固定ロット2%  初期資産100万円/銘柄",
             fontsize=13, fontweight="bold")

# 上段: 5銘柄個別エクイティカーブ
ax_main = fig.add_subplot(2, 1, 1)
for pair, res in pair_results.items():
    eq_df = pd.DataFrame(res["eq_tl"], columns=["time","equity"]).sort_values("time")
    final = eq_df["equity"].iloc[-1]
    profit = final - INIT_CASH
    label = f"{pair}  最終:{final/10000:.0f}万円（+{profit/10000:.0f}万円）"
    ax_main.plot(eq_df["time"], eq_df["equity"]/10000,
                 color=COLORS[pair], linewidth=1.8, label=label, alpha=0.9)

ax_main.axhline(INIT_CASH/10000, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax_main.set_ylabel("資産（万円）", fontsize=11)
ax_main.set_title("銘柄別エクイティカーブ", fontsize=11)
ax_main.legend(fontsize=9, loc="upper left")
ax_main.grid(True, alpha=0.3)
ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax_main.tick_params(axis="x", labelsize=9)
ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# 下段左: 5銘柄合算エクイティカーブ
ax_sum = fig.add_subplot(2, 2, 3)
# 全トレードを時系列でマージして累積PnL
all_trades = []
for pair, res in pair_results.items():
    for t in res["trades"]:
        all_trades.append({"exit_time": t["exit_time"], "pnl": t["pnl"], "pair": pair})
all_df = pd.DataFrame(all_trades).sort_values("exit_time")
all_df["cum_pnl"] = all_df["pnl"].cumsum()
all_df["total_equity"] = len(pair_results) * INIT_CASH + all_df["cum_pnl"]
ax_sum.plot(all_df["exit_time"], all_df["total_equity"]/10000,
            color="#1e40af", linewidth=2.0, alpha=0.9)
ax_sum.axhline(len(pair_results) * INIT_CASH / 10000, color="gray", linestyle="--", linewidth=0.8)
ax_sum.fill_between(all_df["exit_time"], len(pair_results) * INIT_CASH / 10000,
                    all_df["total_equity"]/10000, alpha=0.15, color="#3b82f6")
ax_sum.set_ylabel("合計資産（万円）", fontsize=10)
ax_sum.set_title(f"5銘柄合算  最終:{all_df['total_equity'].iloc[-1]/10000:.0f}万円", fontsize=10)
ax_sum.grid(True, alpha=0.3)
ax_sum.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
ax_sum.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax_sum.tick_params(axis="x", labelsize=8)

# 下段右: 月次PnL棒グラフ（合算）
ax_monthly = fig.add_subplot(2, 2, 4)
all_df["exit_time"] = pd.to_datetime(all_df["exit_time"])
all_df["month"] = all_df["exit_time"].dt.to_period("M")
monthly_pnl = all_df.groupby("month")["pnl"].sum()
colors_m = ["#22c55e" if v > 0 else "#ef4444" for v in monthly_pnl.values]
bars = ax_monthly.bar(range(len(monthly_pnl)), monthly_pnl.values/10000, color=colors_m, alpha=0.85)
ax_monthly.axhline(0, color="black", linewidth=0.8)
ax_monthly.set_xticks(range(len(monthly_pnl)))
ax_monthly.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45, fontsize=7)
ax_monthly.set_ylabel("月次PnL（万円）", fontsize=10)
ax_monthly.set_title(f"月次PnL（5銘柄合算）  月次黒字:{(monthly_pnl>0).sum()}/{len(monthly_pnl)}ヶ月", fontsize=10)
ax_monthly.grid(True, alpha=0.3, axis="y")

# 合計利益テキスト
fig.text(0.5, 0.01,
         f"5銘柄合計利益: {total_profit/10000:.0f}万円  "
         f"（初期投資{len(pair_results)*INIT_CASH/10000:.0f}万円 → 最終{total_final/10000:.0f}万円  "
         f"リターン+{total_profit/(len(pair_results)*INIT_CASH)*100:.0f}%）",
         ha="center", fontsize=11, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#3b82f6", alpha=0.9))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(OUT_DIR, "f1f3_equity_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nチャート保存: results/f1f3_equity_curve.png")
