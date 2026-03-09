"""
F2（ストリーク: 前回負け後スキップ）で除外された167件の内訳分析
F1+F3 vs F1+F2+F3 の差分を抽出する
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

def generate_signals_f1f3(data_1m, data_1h_raw, data_4h, spread_pips, pip_size):
    """F1+F3のみ（ストリークなし）"""
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
            # F3: 1H確認
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
            signals.append({"time": et, "dir": direction, "ep": ep, "sl": sl,
                            "tp": tp, "risk": risk, "tf": "1h"})
            used_times.add(et); break

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals) if signals else pd.DataFrame()


def simulate_with_skip_tracking(sigs, data_1m, usdjpy_1m, rm):
    """F2スキップされたシグナルを記録しながらシミュレーション"""
    if len(sigs) == 0:
        return [], []

    usdjpy_init = usdjpy_1m.iloc[0]["close"] if usdjpy_1m is not None and len(usdjpy_1m) > 0 else 150.0
    m1_times = data_1m.index.values
    m1_highs = data_1m["high"].values
    m1_lows  = data_1m["low"].values
    uj_times = usdjpy_1m.index.values if usdjpy_1m is not None else None
    uj_closes = usdjpy_1m["close"].values if usdjpy_1m is not None else None

    trades = []; skipped = []
    last_result = None

    for _, sig in sigs.iterrows():
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]
        risk = sig["risk"]; direction = sig["dir"]
        entry_time = sig["time"]

        # F2チェック: スキップ対象かどうか記録
        is_skipped = (last_result == "SL_LOSS")
        if is_skipped:
            last_result = None
            # このシグナルを「もし実行したら」どうなったかをシミュレート
            start_idx = np.searchsorted(m1_times, np.datetime64(entry_time), side="right")
            if start_idx < len(m1_times):
                lot = rm.calc_lot(INIT_CASH, risk, ref_price=ep, usdjpy_rate=usdjpy_init)
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
                if result is None: result = "BE" if half_done else "OPEN"
                if result != "OPEN":
                    exit_price = sl_current if result == "SL" else tp
                    usdjpy_at_exit = usdjpy_init
                    if uj_times is not None and exit_idx is not None:
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
                    skipped.append({"entry_time": entry_time, "result": result,
                                    "pnl": pnl, "half_done": half_done,
                                    "hour_utc": pd.Timestamp(entry_time).hour,
                                    "tf": sig["tf"]})
            continue

        # 通常シミュレーション
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

        exit_price = sl_current if result == "SL" else tp
        usdjpy_at_exit = usdjpy_init
        if uj_times is not None and exit_idx is not None:
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

        last_result = "SL_LOSS" if (result == "SL" and not half_done) else "OTHER"
        trades.append({"entry_time": entry_time, "result": result, "pnl": pnl,
                       "half_done": half_done, "hour_utc": pd.Timestamp(entry_time).hour,
                       "tf": sig["tf"]})

    return trades, skipped


# ── 実行 ──────────────────────────────────────────────────
uj_raw = try_load("usdjpy", "1m")
all_skipped = []

for pair in PAIRS:
    sym = SYM_MAP[pair]
    d1m  = try_load(sym, "1m")
    d1h  = try_load(sym, "1h")
    d4h  = try_load(sym, "4h")
    if any(d is None for d in [d1m, d1h, d4h]): continue

    d1m  = slice_period(d1m,  START, END)
    d1h  = slice_period(d1h,  START, END)
    d4h  = slice_period(d4h,  START, END)
    rm   = RiskManager(pair, risk_pct=RISK_PCT)
    usdjpy_1m = slice_period(uj_raw, START, END) if rm.quote_type != "A" and uj_raw is not None else None

    sigs = generate_signals_f1f3(d1m, d1h, d4h, rm.spread_pips, rm.pip_size)
    trades, skipped = simulate_with_skip_tracking(sigs, d1m, usdjpy_1m, rm)
    for s in skipped:
        s["pair"] = pair
    all_skipped.extend(skipped)
    print(f"[{pair}] F1+F3件数:{len(trades)+len(skipped)} スキップ:{len(skipped)}")

skip_df = pd.DataFrame(all_skipped)
if len(skip_df) == 0:
    print("スキップなし")
    exit()

skip_df["entry_time"] = pd.to_datetime(skip_df["entry_time"])
skip_df["month"] = skip_df["entry_time"].dt.to_period("M")

print(f"\n=== F2スキップ件数: {len(skip_df)} ===")
print(f"TP: {(skip_df['result']=='TP').sum()} / SL: {(skip_df['result']=='SL').sum()} / BE: {(skip_df['result']=='BE').sum()}")
print(f"勝率: {(skip_df['result']=='TP').mean()*100:.1f}%")
print(f"平均PnL: {skip_df['pnl'].mean():.0f}円")
print(f"PnL合計: {skip_df['pnl'].sum():.0f}円")
print()

# 銘柄別
print("=== 銘柄別 ===")
print(skip_df.groupby("pair").agg(
    n=("pnl","count"), wr=("result", lambda x: (x=="TP").mean()),
    avg_pnl=("pnl","mean"), total_pnl=("pnl","sum")
).round(3).to_string())
print()

# 時間帯別
print("=== 時間帯別 ===")
print(skip_df.groupby("hour_utc").agg(
    n=("pnl","count"), wr=("result", lambda x: (x=="TP").mean()), avg_pnl=("pnl","mean")
).round(3).to_string())
print()

# 時間足別
print("=== 時間足別 ===")
print(skip_df.groupby("tf").agg(
    n=("pnl","count"), wr=("result", lambda x: (x=="TP").mean()), avg_pnl=("pnl","mean")
).round(3).to_string())
print()

# 月別
print("=== 月別 ===")
print(skip_df.groupby("month").agg(
    n=("pnl","count"), wr=("result", lambda x: (x=="TP").mean()), avg_pnl=("pnl","mean")
).round(3).to_string())
print()

# ── チャート ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"F2スキップ {len(skip_df)}件の内訳分析", fontsize=12, fontweight="bold")

# 1. 銘柄別件数
ax = axes[0]
pair_cnt = skip_df.groupby("pair")["pnl"].agg(["count","mean"])
colors = ["#22c55e" if v > 0 else "#ef4444" for v in pair_cnt["mean"]]
ax.bar(pair_cnt.index, pair_cnt["count"], color=colors, alpha=0.8)
ax.set_title("銘柄別 スキップ件数\n（緑=平均PnL+）"); ax.set_ylabel("件数"); ax.grid(True, alpha=0.3, axis="y")

# 2. 時間帯別 スキップ件数と勝率
ax = axes[1]
h_stats = skip_df.groupby("hour_utc").agg(n=("pnl","count"), wr=("result", lambda x: (x=="TP").mean()))
ax2 = ax.twinx()
ax.bar(h_stats.index, h_stats["n"], color="#6366f1", alpha=0.6, label="件数")
ax2.plot(h_stats.index, h_stats["wr"]*100, color="orange", marker="o", linewidth=2, markersize=4, label="勝率")
ax2.axhline(29.2, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("時間帯（UTC）"); ax.set_ylabel("件数"); ax2.set_ylabel("勝率(%)")
ax.set_title("時間帯別 スキップ件数・勝率"); ax.grid(True, alpha=0.3, axis="y")

# 3. PnL分布
ax = axes[2]
colors_r = {"TP": "#22c55e", "SL": "#ef4444", "BE": "#f97316"}
for result, grp in skip_df.groupby("result"):
    ax.hist(grp["pnl"].clip(-80000, 80000), bins=30, alpha=0.6,
            color=colors_r.get(result, "gray"), label=result)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("PnL（円）"); ax.set_ylabel("頻度"); ax.set_title("スキップ件数のPnL分布")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f2_skipped_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート保存: results/f2_skipped_analysis.png")
