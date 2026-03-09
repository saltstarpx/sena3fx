"""
backtest_7sym_v1.py
===================
7銘柄 3ヶ月バックテスト【yagami_position8_risk10 v1】
銘柄: USDJPY, AUDUSD, EURJPY, EURGBP, US30, SPX500, XAUUSD
期間: 2025-03-03 〜 2025-06-02
初期資金: 100万円
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from utils.risk_manager     import AdaptiveRiskManager
from utils.position_manager import PositionManager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INIT_CASH  = 1_000_000
BASE_RISK  = 0.02
RR_RATIO   = 2.5
HALF_R     = 1.0
START      = "2025-03-03"
END        = "2025-06-02"
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS  = ["USDJPY", "AUDUSD", "EURJPY", "EURGBP", "US30", "SPX500", "XAUUSD"]
KLOW_THR = 0.0015
USDJPY_RATE = 150.0

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
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)].copy()

def calculate_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df, span=20, atr_period=14):
    df = df.copy()
    df["atr"]   = calculate_atr(df, atr_period)
    df["ema20"] = df["close"].ewm(span=span, adjust=False).mean()
    df["trend"] = np.where(df["close"] > df["ema20"], 1, -1)
    return df

def check_kmid_klow(bar, direction):
    o, c, l = bar["open"], bar["close"], bar["low"]
    kmid_ok = (direction == 1 and c > o) or (direction == -1 and c < o)
    body_bottom = min(o, c)
    klow_ratio  = (body_bottom - l) / o if o > 0 else 0
    return kmid_ok and klow_ratio < KLOW_THR

def generate_signals(data_1m, data_15m, data_4h, spread_pips, pip_size):
    spread  = spread_pips * pip_size
    data_4h = add_indicators(data_4h)
    data_1h = data_15m.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna(subset=["open","close"])
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals = []; used_times = set()
    h1_times = data_1h.index.tolist()

    for i in range(2, len(h1_times)):
        h1_ct    = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        atr_val  = data_1h.iloc[i]["atr"]
        if pd.isna(atr_val) or atr_val <= 0: continue

        h4_before = data_4h[data_4h.index <= h1_ct]
        if len(h4_before) == 0: continue
        h4_latest = h4_before.iloc[-1]
        if pd.isna(h4_latest.get("atr", np.nan)) or pd.isna(h4_latest.get("ema20", np.nan)): continue

        trend = h4_latest["trend"]; h4_atr = h4_latest["atr"]
        tolerance = atr_val * 0.3

        for direction, low_or_high in [(1, "low"), (-1, "high")]:
            if trend != direction: continue
            v1 = h1_prev2[low_or_high]; v2 = h1_prev1[low_or_high]
            if abs(v1 - v2) > tolerance: continue
            if direction == 1 and h1_prev1["close"] <= h1_prev1["open"]: continue
            if direction == -1 and h1_prev1["close"] >= h1_prev1["open"]: continue
            if not check_kmid_klow(h4_latest, direction): continue   # 4H足
            if not check_kmid_klow(h1_prev1, direction): continue    # 1H足 AND条件

            entry_window_end = h1_ct + pd.Timedelta(minutes=2)
            m1_window = data_1m[(data_1m.index >= h1_ct) & (data_1m.index < entry_window_end)]
            if len(m1_window) == 0: continue
            entry_bar = m1_window.iloc[0]; entry_time = entry_bar.name
            if entry_time in used_times: continue

            raw_ep = entry_bar["open"]
            if direction == 1:
                sl = min(v1, v2) - atr_val * 0.15; ep = raw_ep + spread
                risk = raw_ep - sl; tp = raw_ep + risk * RR_RATIO
            else:
                sl = max(v1, v2) + atr_val * 0.15; ep = raw_ep - spread
                risk = sl - raw_ep; tp = raw_ep - risk * RR_RATIO

            if 0 < risk <= h4_atr * 2:
                signals.append({"time": entry_time, "dir": direction,
                                 "ep": ep, "sl": sl, "tp": tp, "risk": risk})
                used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals

def simulate_single(symbol, signals, data_1m, init_cash, base_risk):
    arm = AdaptiveRiskManager(symbol, base_risk_pct=base_risk)
    equity = init_cash; trades = []
    times = data_1m.index.values; highs = data_1m["high"].values; lows = data_1m["low"].values

    for sig in signals:
        entry_time = sig["time"]; direction = sig["dir"]
        ep = sig["ep"]; sl = sig["sl"]; tp = sig["tp"]; risk = sig["risk"]
        idx_arr = np.searchsorted(times, np.datetime64(entry_time))
        if idx_arr >= len(times): continue

        lot, eff_risk, reason = arm.calc_lot_adaptive(
            equity=equity, sl_distance=risk, ref_price=ep, usdjpy_rate=USDJPY_RATE)
        if lot <= 0: continue

        equity -= arm.calc_commission_jpy(lot, USDJPY_RATE)
        half_done = False; be_sl = None; half_pnl = 0.0
        result = None; exit_time = None; exit_price = None

        for j in range(idx_arr + 1, len(times)):
            bh = highs[j]; bl = lows[j]; bt = times[j]
            cur_sl = be_sl if half_done else sl

            if direction == 1:
                if bl <= cur_sl:
                    exit_price = cur_sl; exit_time = bt; rem = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price, lot*rem, USDJPY_RATE, ep)
                    equity += pnl; equity -= arm.calc_commission_jpy(lot*rem, USDJPY_RATE)
                    result = "BE" if (half_done and abs(exit_price-ep)<risk*0.01) else ("TP" if pnl>0 else "SL")
                    break
                if bh >= tp:
                    if not half_done and bh >= ep + risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep+risk*HALF_R, lot*0.5, USDJPY_RATE, ep)
                        equity += hp; half_pnl += hp; equity -= arm.calc_commission_jpy(lot*0.5, USDJPY_RATE)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bt
                    pnl = arm.calc_pnl_jpy(direction, ep, tp, lot*0.5, USDJPY_RATE, ep)
                    equity += pnl; equity -= arm.calc_commission_jpy(lot*0.5, USDJPY_RATE)
                    result = "TP"; break
                if not half_done and bh >= ep + risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep+risk*HALF_R, lot*0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp; equity -= arm.calc_commission_jpy(lot*0.5, USDJPY_RATE)
                    half_done = True; be_sl = ep
            else:
                if bh >= cur_sl:
                    exit_price = cur_sl; exit_time = bt; rem = 0.5 if half_done else 1.0
                    pnl = arm.calc_pnl_jpy(direction, ep, exit_price, lot*rem, USDJPY_RATE, ep)
                    equity += pnl; equity -= arm.calc_commission_jpy(lot*rem, USDJPY_RATE)
                    result = "BE" if (half_done and abs(exit_price-ep)<risk*0.01) else ("TP" if pnl>0 else "SL")
                    break
                if bl <= tp:
                    if not half_done and bl <= ep - risk * HALF_R:
                        hp = arm.calc_pnl_jpy(direction, ep, ep-risk*HALF_R, lot*0.5, USDJPY_RATE, ep)
                        equity += hp; half_pnl += hp; equity -= arm.calc_commission_jpy(lot*0.5, USDJPY_RATE)
                        half_done = True; be_sl = ep
                    exit_price = tp; exit_time = bt
                    pnl = arm.calc_pnl_jpy(direction, ep, tp, lot*0.5, USDJPY_RATE, ep)
                    equity += pnl; equity -= arm.calc_commission_jpy(lot*0.5, USDJPY_RATE)
                    result = "TP"; break
                if not half_done and bl <= ep - risk * HALF_R:
                    hp = arm.calc_pnl_jpy(direction, ep, ep-risk*HALF_R, lot*0.5, USDJPY_RATE, ep)
                    equity += hp; half_pnl += hp; equity -= arm.calc_commission_jpy(lot*0.5, USDJPY_RATE)
                    half_done = True; be_sl = ep

        if result is None: continue
        arm.update_peak(equity)
        trades.append({"symbol": symbol, "entry_time": entry_time, "exit_time": exit_time,
                        "dir": direction, "ep": ep, "sl": sl, "tp": tp, "exit_price": exit_price,
                        "result": result, "lot": lot, "eff_risk": eff_risk,
                        "half_pnl": half_pnl, "equity_after": equity})
    return trades, equity

def apply_position_filter_v1(all_trades_df, init_cash, base_risk):
    """v1: yagami_position8_risk10 フィルター（含み益枠外なし）"""
    pm = PositionManager()
    equity = init_cash; equity_curve = [init_cash]; accepted_trades = []
    df = all_trades_df.sort_values("entry_time").reset_index(drop=True)
    open_pos = {}

    for _, row in df.iterrows():
        sym = row["symbol"]; entry_time = row["entry_time"]; exit_time = row["exit_time"]

        closed = [s for s, p in open_pos.items() if p["exit_time"] <= entry_time]
        for s in closed:
            pos = open_pos.pop(s)
            equity += pos["pnl_delta"]; equity_curve.append(equity)
            pm.close_position(s)

        if sym in open_pos: continue

        eff_risk = row["eff_risk"]
        ok, reason = pm.can_enter(sym, eff_risk)
        if not ok: continue

        pm.open_position(sym, eff_risk, entry_price=row["ep"],
                         entry_time=pd.Timestamp(entry_time))
        open_pos[sym] = {"exit_time": exit_time, "pnl_delta": row["pnl_delta"]}
        accepted_trades.append(row.to_dict())

    for s, pos in open_pos.items():
        equity += pos["pnl_delta"]; equity_curve.append(equity)

    return pd.DataFrame(accepted_trades), pd.Series(equity_curve)

# ── メイン ────────────────────────────────────────────────
print("=" * 70)
print(f"7銘柄 3ヶ月バックテスト【yagami_position8_risk10 v1】")
print(f"期間: {START} 〜 {END}  初期資金: {INIT_CASH:,}円  基本リスク: {BASE_RISK*100:.0f}%")
print(f"銘柄: {SYMBOLS}")
print("=" * 70)

all_trades = []
for sym in SYMBOLS:
    sym_lower = sym.lower()
    # OOSデータを優先、なければ汎用データを使用
    d1m  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_1m.csv"))
    d15m = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_15m.csv"))
    d4h  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_oos_4h.csv"))
    # フォールバック
    if d1m  is None: d1m  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_1m.csv"))
    if d15m is None: d15m = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_15m.csv"))
    if d4h  is None: d4h  = load_csv(os.path.join(DATA_DIR, f"{sym_lower}_4h.csv"))

    if d1m is None or d15m is None or d4h is None:
        print(f"  {sym}: データなし → スキップ"); continue

    d1m  = slice_period(d1m,  START, END)
    d15m = slice_period(d15m, START, END)
    d4h  = slice_period(d4h,  START, END)
    if d1m is None or len(d1m) == 0:
        print(f"  {sym}: 期間データなし → スキップ"); continue

    arm_ref = AdaptiveRiskManager(sym, base_risk_pct=BASE_RISK)
    print(f"  {sym}: シグナル生成中... (spread={arm_ref.spread_pips}pips)")
    sigs = generate_signals(d1m, d15m, d4h, arm_ref.spread_pips, arm_ref.pip_size)
    print(f"    → シグナル{len(sigs)}件 / シミュレーション実行中...")
    trades, final_eq = simulate_single(sym, sigs, d1m, INIT_CASH, BASE_RISK)

    prev_eq = INIT_CASH
    for t in trades:
        t["pnl_delta"] = t["equity_after"] - prev_eq; prev_eq = t["equity_after"]

    n_tp = sum(1 for t in trades if t["result"] == "TP")
    n_sl = sum(1 for t in trades if t["result"] == "SL")
    n_be = sum(1 for t in trades if t["result"] == "BE")
    wr   = n_tp / (n_tp + n_sl) if (n_tp + n_sl) > 0 else 0
    print(f"    → 完了: {len(trades)}件 TP:{n_tp} SL:{n_sl} BE:{n_be} 勝率:{wr*100:.0f}%")
    all_trades.extend(trades)

if not all_trades:
    print("トレードなし。終了。"); sys.exit(0)

all_df = pd.DataFrame(all_trades)
all_df["entry_time"] = pd.to_datetime(all_df["entry_time"], utc=True)
all_df["exit_time"]  = pd.to_datetime(all_df["exit_time"],  utc=True)
print(f"\n全シグナル合計: {len(all_df)}件")
print("PositionManagerフィルタリング中（v1）...")
accepted_df, equity_curve = apply_position_filter_v1(all_df, INIT_CASH, BASE_RISK)

# 集計
n_total = len(accepted_df)
n_tp = (accepted_df["result"] == "TP").sum()
n_sl = (accepted_df["result"] == "SL").sum()
n_be = (accepted_df["result"] == "BE").sum()
win_rate = n_tp / (n_tp + n_sl) * 100 if (n_tp + n_sl) > 0 else 0

eq_arr = equity_curve.values
peak   = np.maximum.accumulate(eq_arr)
dd_arr = (eq_arr - peak) / peak * 100
max_dd = dd_arr.min()
final_equity = eq_arr[-1]
ret_pct = (final_equity - INIT_CASH) / INIT_CASH * 100

tp_pnl = accepted_df[accepted_df["result"]=="TP"]["pnl_delta"].sum()
sl_pnl = abs(accepted_df[accepted_df["result"]=="SL"]["pnl_delta"].sum())
pf = tp_pnl / sl_pnl if sl_pnl > 0 else float("inf")

accepted_df["month"] = pd.to_datetime(accepted_df["exit_time"], utc=True).dt.to_period("M")
monthly_pnl = accepted_df.groupby("month")["pnl_delta"].sum()
n_pos_months = (monthly_pnl > 0).sum()

print("\n" + "=" * 70)
print(f"【v1 結果サマリー（{START} 〜 {END}）】")
print("=" * 70)
print(f"  採用トレード数:        {n_total}件（TP:{n_tp} SL:{n_sl} BE:{n_be}）")
print(f"  勝率（BE除く）:        {win_rate:.1f}%")
print(f"  プロフィットファクター: {pf:.2f}")
print(f"  最大ドローダウン:      {max_dd:.1f}%")
print(f"  リターン（3ヶ月）:     {ret_pct:+.1f}%")
print(f"  月次プラス:            {n_pos_months}/{len(monthly_pnl)}")
print(f"  最終資産:              {final_equity:,.0f}円")
print("=" * 70)

# 銘柄別
sym_stats = accepted_df.groupby("symbol").agg(
    count=("result","count"),
    tp=("result", lambda x: (x=="TP").sum()),
    sl=("result", lambda x: (x=="SL").sum()),
    be=("result", lambda x: (x=="BE").sum()),
).reset_index()
print("\n【銘柄別採用トレード数】")
for _, r in sym_stats.iterrows():
    wr = r["tp"] / (r["tp"] + r["sl"]) * 100 if (r["tp"] + r["sl"]) > 0 else 0
    print(f"  {r['symbol']:8s}: {int(r['count']):3d}件 TP:{int(r['tp']):3d} SL:{int(r['sl']):3d} BE:{int(r['be']):3d} 勝率:{wr:.0f}%")

# チャート
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"7銘柄バックテスト【v1: yagami_position8_risk10】\n{START}〜{END}  初期資金: {INIT_CASH:,}円", fontsize=13, fontweight="bold")

ax = axes[0,0]
ax.plot(eq_arr / 1e6, color="#3b82f6", linewidth=1.5)
ax.axhline(INIT_CASH/1e6, color="gray", linestyle="--", alpha=0.5)
ax.text(0.04, 0.93, f"最終: {final_equity/1e6:.2f}M円\n{ret_pct:+.1f}%", transform=ax.transAxes,
        fontsize=10, color="#3b82f6", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax.set_title("資産曲線"); ax.set_ylabel("資産（百万円）"); ax.grid(alpha=0.3)

ax = axes[1,0]
ax.fill_between(range(len(dd_arr)), dd_arr, 0, color="#ef4444", alpha=0.5)
mdd_i = int(np.argmin(dd_arr))
ax.annotate(f"最大DD\n{dd_arr[mdd_i]:.1f}%", xy=(mdd_i, dd_arr[mdd_i]),
            xytext=(mdd_i+max(3,len(dd_arr)//10), dd_arr[mdd_i]+abs(dd_arr[mdd_i])*0.3),
            fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
ax.set_title("ドローダウン"); ax.set_ylabel("DD（%）"); ax.grid(alpha=0.3)

ax = axes[0,1]
months_str = [str(m) for m in monthly_pnl.index]
colors = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly_pnl.values]
bars = ax.bar(range(len(months_str)), monthly_pnl.values/1e4, color=colors, alpha=0.85)
for i, (bar, val) in enumerate(zip(bars, monthly_pnl.values)):
    ax.text(i, val/1e4 + (0.3 if val>=0 else -1.5), f"{val/1e4:+.0f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(months_str))); ax.set_xticklabels(months_str, rotation=30, fontsize=9)
ax.set_title(f"月次損益（{n_pos_months}/{len(monthly_pnl)}月プラス）"); ax.set_ylabel("損益（万円）"); ax.grid(alpha=0.3, axis="y")

ax = axes[1,1]
sym_order = sym_stats.sort_values("count", ascending=False)
colors_sym = ["#3b82f6"] * len(sym_order)
bars2 = ax.bar(range(len(sym_order)), sym_order["count"], color=colors_sym, alpha=0.8)
ax2 = ax.twinx()
wr_list = [r["tp"]/(r["tp"]+r["sl"])*100 if (r["tp"]+r["sl"])>0 else 0 for _, r in sym_order.iterrows()]
ax2.plot(range(len(sym_order)), wr_list, "o-", color="#f59e0b", linewidth=2)
ax.set_xticks(range(len(sym_order))); ax.set_xticklabels(sym_order["symbol"].tolist(), fontsize=9)
ax.set_ylabel("採用トレード数"); ax2.set_ylabel("勝率（%）"); ax2.set_ylim(0, 100)
ax.set_title(f"銘柄別採用トレード数・勝率\nPF:{pf:.2f}  最大DD:{max_dd:.1f}%  勝率:{win_rate:.1f}%")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "7sym_v1_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nチャート保存: {out_path}")

# CSV保存（v2比較用）
accepted_df.to_csv(os.path.join(OUT_DIR, "7sym_v1_trades.csv"), index=False)
np.save(os.path.join(OUT_DIR, "7sym_v1_equity.npy"), eq_arr)
print("完了。")
