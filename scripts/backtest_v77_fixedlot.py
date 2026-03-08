"""
backtest_v77_fixedlot.py
========================
backtest_v77_4m_lite.py の固定ロット版
- ロットを期間中固定（初期資産100万円×2%リスクで計算した値を固定）
- 複利膨張の影響を排除して戦略の真の優位性を測定
- 複利版との比較チャートも出力

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

INIT_CASH    = 1_000_000
RISK_PCT     = 0.02
RR_RATIO     = 2.5
HALF_R       = 1.0
KLOW_THR     = 0.0015
BODY_MIN_ATR = 0.3
MOUNTAIN_MIN = 0.5
COOLDOWN_H   = 4
START        = "2025-11-01"
END          = "2026-02-27"

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

def detect_double_bottom(bars, atr_val, direction):
    n = len(bars)
    if n < 5: return False, None
    tolerance   = atr_val * 0.3
    body_min    = atr_val * BODY_MIN_ATR
    mountain_min = atr_val * MOUNTAIN_MIN
    if direction == 1:
        for i in range(n - 4, max(n - 8, 0), -1):
            b1 = bars.iloc[i]; mid = bars.iloc[i+1]
            b2 = bars.iloc[i+2]; conf = bars.iloc[i+3]
            if abs(b1["low"] - b2["low"]) > tolerance: continue
            if b2["low"] < b1["low"] - tolerance: continue
            if mid["high"] < b1["close"] + mountain_min: continue
            if conf["close"] <= conf["open"]: continue
            if abs(conf["close"] - conf["open"]) < body_min: continue
            if not check_kmid(conf, 1): continue
            if not check_klow(conf, 1): continue
            return True, min(b1["low"], b2["low"]) - atr_val * 0.15
    else:
        for i in range(n - 4, max(n - 8, 0), -1):
            t1 = bars.iloc[i]; mid = bars.iloc[i+1]
            t2 = bars.iloc[i+2]; conf = bars.iloc[i+3]
            if abs(t1["high"] - t2["high"]) > tolerance: continue
            if t2["high"] > t1["high"] + tolerance: continue
            if mid["low"] > t1["close"] - mountain_min: continue
            if conf["close"] >= conf["open"]: continue
            if abs(conf["close"] - conf["open"]) < body_min: continue
            if not check_kmid(conf, -1): continue
            if not check_klow(conf, -1): continue
            return True, max(t1["high"], t2["high"]) + atr_val * 0.15
    return False, None

def generate_signals_hybrid(data_1m, data_15m, data_4h, spread_pips, pip_size, rr_ratio=2.5):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna(subset=["open","close"])
    data_1h = add_indicators(data_1h)
    signals = []; used_times = set()
    last_4h = pd.Timestamp("2000-01-01", tz="UTC")
    last_1h = pd.Timestamp("2000-01-01", tz="UTC")

    for i in range(8, len(data_4h)):
        t = data_4h.index[i]
        if (t - last_4h).total_seconds() / 3600 < COOLDOWN_H: continue
        cur = data_4h.iloc[i]; atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        trend = cur["trend"]
        detected, sl = detect_double_bottom(data_4h.iloc[i-7:i], atr, trend)
        if not detected: continue
        m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
        if len(m1w) == 0: continue
        eb = m1w.iloc[0]; et = eb.name
        if et in used_times: continue
        raw_ep = eb["open"]
        ep = raw_ep + spread if trend == 1 else raw_ep - spread
        risk = (raw_ep - sl) if trend == 1 else (sl - raw_ep)
        tp = raw_ep + risk * rr_ratio if trend == 1 else raw_ep - risk * rr_ratio
        if risk <= 0 or risk > atr * 3: continue
        signals.append({"time": et, "dir": trend, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "4h"})
        used_times.add(et); last_4h = et

    for i in range(8, len(data_1h)):
        t = data_1h.index[i]
        if (t - last_1h).total_seconds() / 3600 < COOLDOWN_H: continue
        cur = data_1h.iloc[i]; atr = cur["atr"]
        if pd.isna(atr) or atr <= 0: continue
        h4b = data_4h[data_4h.index <= t]
        if len(h4b) == 0: continue
        h4l = h4b.iloc[-1]
        if pd.isna(h4l["atr"]) or pd.isna(h4l["ema20"]): continue
        trend = h4l["trend"]; h4_atr = h4l["atr"]
        detected, sl = detect_double_bottom(data_1h.iloc[i-7:i], atr, trend)
        if not detected: continue
        m1w = data_1m[(data_1m.index >= t) & (data_1m.index < t + pd.Timedelta(minutes=2))]
        if len(m1w) == 0: continue
        eb = m1w.iloc[0]; et = eb.name
        if et in used_times: continue
        raw_ep = eb["open"]
        ep = raw_ep + spread if trend == 1 else raw_ep - spread
        risk = (raw_ep - sl) if trend == 1 else (sl - raw_ep)
        tp = raw_ep + risk * rr_ratio if trend == 1 else raw_ep - risk * rr_ratio
        if risk <= 0 or risk > h4_atr * 2: continue
        signals.append({"time": et, "dir": trend, "ep": ep, "sl": sl, "tp": tp, "risk": risk, "tf": "1h"})
        used_times.add(et); last_1h = et

    signals.sort(key=lambda x: x["time"])
    return pd.DataFrame(signals)

def simulate_fixed(signals, data_1m, init_cash, risk_pct, half_r, symbol, usdjpy_1m=None):
    """固定ロット版: 初期資産×risk_pctで計算したロットを期間中固定"""
    rm = RiskManager(symbol, risk_pct=risk_pct)
    if signals is None or len(signals) == 0:
        return [], [init_cash]

    # 固定ロット計算用のUSDJPY（期間開始時点）
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

        # 固定ロット: 常に初期資産ベースで計算
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
        else:  # BE
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
    losses = df[df["pnl"] < 0]
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
print(f"v77 Hybrid 固定ロット版  {START} 〜 {END}")
print(f"初期資金: {INIT_CASH:,}円  リスク: {RISK_PCT*100:.0f}%（固定）  RR: {RR_RATIO}")
print("=" * 70)

fixed_results = []; fixed_trades = []; fixed_eq = {}

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

    sigs = generate_signals_hybrid(d1m, d15m, d4h, spread, pip, rr_ratio=RR_RATIO)
    trades, eq = simulate_fixed(sigs, d1m, INIT_CASH, RISK_PCT, HALF_R, pair, usdjpy_1m)
    stats = calc_stats(trades, eq, pair)
    stats["pair"] = pair
    fixed_results.append(stats); fixed_eq[pair] = eq
    for t in trades: fixed_trades.append(t)

    print(f"  [{pair}] {stats['n']}件 | 勝率{stats['winrate']:.1f}% | "
          f"PF{stats['pf']:.2f} | リターン{stats['return_pct']:+.1f}% | "
          f"MDD{stats['mdd_pct']:.1f}% | 月次+{stats['monthly_plus']}")

df_fixed = pd.DataFrame(fixed_results)
df_fixed.to_csv(os.path.join(OUT_DIR, "v77_fixedlot_results.csv"), index=False)
pd.DataFrame(fixed_trades).to_csv(os.path.join(OUT_DIR, "v77_fixedlot_trades.csv"), index=False)

# 複利版の結果を読み込んで比較
df_compound = pd.read_csv(os.path.join(OUT_DIR, "v77_4m_lite_results.csv"))

# ── 比較チャート ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f"v77 Hybrid: 複利 vs 固定ロット比較\n{START} 〜 {END}", fontsize=13, fontweight="bold")

# 左: PF比較
ax1 = axes[0]
pairs_list = df_fixed["pair"].tolist()
x = np.arange(len(pairs_list))
w = 0.35
pf_compound = [df_compound[df_compound["pair"]==p]["pf"].values[0] if p in df_compound["pair"].values else 0 for p in pairs_list]
pf_fixed    = df_fixed["pf"].tolist()
bars1 = ax1.bar(x - w/2, pf_compound, w, label="複利版", color="#3b82f6", alpha=0.8)
bars2 = ax1.bar(x + w/2, pf_fixed,    w, label="固定ロット版", color="#f97316", alpha=0.8)
ax1.axhline(1.0, color="red", linestyle="--", linewidth=1)
ax1.axhline(1.5, color="orange", linestyle="--", linewidth=1)
ax1.set_xticks(x); ax1.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("プロフィットファクター"); ax1.set_title("PF比較（複利 vs 固定ロット）")
ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

# 右: リターン比較
ax2 = axes[1]
ret_compound = [df_compound[df_compound["pair"]==p]["return_pct"].values[0] if p in df_compound["pair"].values else 0 for p in pairs_list]
ret_fixed    = df_fixed["return_pct"].tolist()
ax2.bar(x - w/2, ret_compound, w, label="複利版", color="#3b82f6", alpha=0.8)
ax2.bar(x + w/2, ret_fixed,    w, label="固定ロット版", color="#f97316", alpha=0.8)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(pairs_list, rotation=45, ha="right", fontsize=9)
ax2.set_ylabel("リターン (%)"); ax2.set_title("リターン比較（複利 vs 固定ロット）")
ax2.legend(); ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "v77_fixedlot_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── サマリー ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("固定ロット版 全銘柄サマリー")
print("=" * 70)
print(df_fixed[["pair","n","winrate","pf","return_pct","mdd_pct","monthly_plus"]].to_string(index=False))

print("\n" + "=" * 70)
print("複利 vs 固定ロット PF比較")
print("=" * 70)
merged = df_fixed[["pair","pf","return_pct"]].rename(columns={"pf":"pf_fixed","return_pct":"ret_fixed"})
comp   = df_compound[["pair","pf","return_pct"]].rename(columns={"pf":"pf_compound","return_pct":"ret_compound"})
cmp = merged.merge(comp, on="pair")
cmp["pf_ratio"] = cmp["pf_compound"] / cmp["pf_fixed"]
print(cmp.to_string(index=False))
print(f"\n平均PF乖離倍率: {cmp['pf_ratio'].mean():.2f}x")
print(f"結果保存: results/v77_fixedlot_comparison.png")
