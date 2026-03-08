"""
二番底・二番天井判定閾値（tolerance = ATR × N）の感度検証
N = 0.1 / 0.2 / 0.3（現行） / 0.4 / 0.5

方法: v76本体のコードをコピーし、tolerance係数だけを外部から注入する。
      それ以外のロジック（EMA20トレンド・1h resample・used_times管理）は完全に同一。
データ: USDJPY IS期間（2024年7月〜2025年2月）OANDAデータ
スプレッド: 0.4pips
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "/home/ubuntu/sena3fx/strategies")
import yagami_mtf_v76 as v76

DATA    = "/home/ubuntu/sena3fx/data"
RESULTS = "/home/ubuntu/sena3fx/results"
SPREAD  = 0.4

def load(p):
    df = pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df

is_1m  = load(f"{DATA}/usdjpy_is_1m.csv")
is_15m = load(f"{DATA}/usdjpy_is_15m.csv")
is_4h  = load(f"{DATA}/usdjpy_is_4h.csv")

# ---- tolerance係数だけ差し替えたシグナル生成関数 ----
def generate_signals_tol(data_1m, data_15m, data_4h, spread_pips, tol_mult, rr_ratio=2.5):
    """v76.generate_signals の tolerance = atr * 0.3 の係数だけを tol_mult に差し替えたもの。
    それ以外は v76 本体と完全に同一。"""
    spread = spread_pips * 0.01

    data_4h = data_4h.copy()
    data_4h["atr"] = v76.calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    data_1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    data_1h["atr"] = v76.calculate_atr(data_1h, period=14)

    signals = []
    used_times = set()

    # 4時間足ループ
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_prev1  = data_4h.iloc[i - 1]
        h4_prev2  = data_4h.iloc[i - 2]
        h4_current = data_4h.iloc[i]
        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        trend     = h4_current["trend"]
        tolerance = atr_val * tol_mult          # ← 唯一の変更点

        if trend == 1:
            low1, low2 = h4_prev2["low"], h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                sl  = min(low1, low2) - atr_val * 0.15
                end = h4_current_time + pd.Timedelta(minutes=2)
                win = data_1m[(data_1m.index >= h4_current_time) & (data_1m.index < end)]
                if len(win) > 0:
                    bar = win.iloc[0]; t = bar.name
                    if t not in used_times:
                        raw_ep = bar["open"]; ep = raw_ep + spread
                        risk = raw_ep - sl
                        if 0 < risk <= atr_val * 3:
                            signals.append({"time": t, "dir": 1, "ep": ep, "sl": sl,
                                            "tp": raw_ep + risk * rr_ratio,
                                            "risk": risk, "spread": spread, "tf": "4h"})
                            used_times.add(t)
        if trend == -1:
            high1, high2 = h4_prev2["high"], h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                sl  = max(high1, high2) + atr_val * 0.15
                end = h4_current_time + pd.Timedelta(minutes=2)
                win = data_1m[(data_1m.index >= h4_current_time) & (data_1m.index < end)]
                if len(win) > 0:
                    bar = win.iloc[0]; t = bar.name
                    if t not in used_times:
                        raw_ep = bar["open"]; ep = raw_ep - spread
                        risk = sl - raw_ep
                        if 0 < risk <= atr_val * 3:
                            signals.append({"time": t, "dir": -1, "ep": ep, "sl": sl,
                                            "tp": raw_ep - risk * rr_ratio,
                                            "risk": risk, "spread": spread, "tf": "4h"})
                            used_times.add(t)

    # 1時間足ループ
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1  = data_1h.iloc[i - 1]
        h1_prev2  = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]
        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]
        tolerance = atr_val * tol_mult          # ← 唯一の変更点

        if trend == 1:
            low1, low2 = h1_prev2["low"], h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                sl  = min(low1, low2) - atr_val * 0.15
                end = h1_current_time + pd.Timedelta(minutes=2)
                win = data_1m[(data_1m.index >= h1_current_time) & (data_1m.index < end)]
                if len(win) > 0:
                    bar = win.iloc[0]; t = bar.name
                    if t not in used_times:
                        raw_ep = bar["open"]; ep = raw_ep + spread
                        risk = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            signals.append({"time": t, "dir": 1, "ep": ep, "sl": sl,
                                            "tp": raw_ep + risk * rr_ratio,
                                            "risk": risk, "spread": spread, "tf": "1h"})
                            used_times.add(t)
        if trend == -1:
            high1, high2 = h1_prev2["high"], h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                sl  = max(high1, high2) + atr_val * 0.15
                end = h1_current_time + pd.Timedelta(minutes=2)
                win = data_1m[(data_1m.index >= h1_current_time) & (data_1m.index < end)]
                if len(win) > 0:
                    bar = win.iloc[0]; t = bar.name
                    if t not in used_times:
                        raw_ep = bar["open"]; ep = raw_ep - spread
                        risk = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            signals.append({"time": t, "dir": -1, "ep": ep, "sl": sl,
                                            "tp": raw_ep - risk * rr_ratio,
                                            "risk": risk, "spread": spread, "tf": "1h"})
                            used_times.add(t)

    signals.sort(key=lambda x: x["time"])
    return signals

# ---- バックテストエンジン（backtest_full_oanda.pyと完全に同一） ----
def run_bt(signals, data_1m):
    sig_map = {s["time"]: s for s in signals}
    trades = []
    pos = None
    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name
        closed_this_bar = False
        if pos is not None:
            d       = pos["dir"]
            raw_ep  = pos["ep"] - pos["spread"] * d
            half_tp = raw_ep + pos["risk"] * d
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"] = raw_ep
                    pos["half_closed"] = True
            if (d == 1 and bar["low"] <= pos["sl"]) or (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl = (pos["sl"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + sl_pnl
                trades.append({"pnl": total, "result": "win" if total > 0 else "loss"})
                pos = None; closed_this_bar = True
            elif (d == 1 and bar["high"] >= pos["tp"]) or (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl = (pos["tp"] - pos["ep"]) * 100 * d
                total  = pos.get("half_pnl", 0) + tp_pnl
                trades.append({"pnl": total, "result": "win" if total > 0 else "loss"})
                pos = None; closed_this_bar = True
        if pos is None and not closed_this_bar and t in sig_map:
            pos = {**sig_map[t], "entry_time": t, "half_closed": False}
    return pd.DataFrame(trades)

# ---- 感度検証 ----
tol_values = [0.1, 0.2, 0.3, 0.4, 0.5]
results = []

for tol in tol_values:
    label = f"ATR×{tol}"
    print(f"\ntolerance = {label}", flush=True)
    sigs = generate_signals_tol(is_1m, is_15m, is_4h, SPREAD, tol)
    print(f"  シグナル数: {len(sigs)}")
    df_t = run_bt(sigs, is_1m)
    if df_t.empty:
        print("  トレードなし"); continue
    wins   = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df_t) * 100
    avg_w  = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_l  = losses["pnl"].mean() if len(losses) > 0 else 0
    kelly  = wr/100 - (1 - wr/100) / (abs(avg_w) / abs(avg_l)) if avg_l != 0 else 0
    t_stat, p_val = stats.ttest_1samp(df_t["pnl"], 0)
    total_pnl = df_t["pnl"].sum()
    print(f"  トレード数: {len(df_t)}  勝率: {wr:.1f}%  PF: {pf:.2f}  "
          f"総損益: {total_pnl:+.1f}pips  p値: {p_val:.4f}")
    results.append(dict(tol=tol, label=label, trades=len(df_t), win_rate=wr, pf=pf,
                        total_pnl=total_pnl, kelly=kelly, t_stat=t_stat, p_value=p_val))

df_res = pd.DataFrame(results)
print("\n\n=== 感度検証サマリー ===")
print(df_res[["label","trades","win_rate","pf","total_pnl","kelly","p_value"]].to_string(index=False))

# ---- チャート ----
plt.rcParams["font.family"] = "Noto Sans CJK JP"
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#0d0d1a")
fig.suptitle("二番底・二番天井 判定閾値（ATR×N）感度検証\n"
             "USDJPY IS期間（2024年7月〜2025年2月）/ スプレッド0.4pips\n"
             "橙色 = 現行値（ATR×0.3）",
             fontsize=12, color="white", y=1.02)

colors = ["#f39c12" if t == 0.3 else "#3498db" for t in df_res["tol"]]
labels = df_res["label"].tolist()

def bar_ax(ax, y_vals, ylabel, title):
    bars = ax.bar(labels, y_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=10, color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ymax = max(y_vals) if max(y_vals) > 0 else 1
    for bar, val in zip(bars, y_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + ymax * 0.02,
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=9, color="white")

bar_ax(axes[0,0], df_res["trades"].tolist(), "回数", "トレード数")
bar_ax(axes[0,1], df_res["pf"].tolist(), "PF", "プロフィットファクター")
bar_ax(axes[1,0], df_res["win_rate"].tolist(), "%", "勝率")
bar_ax(axes[1,1], df_res["total_pnl"].tolist(), "pips", "総損益")

plt.tight_layout()
out = f"{RESULTS}/v76_tolerance_sensitivity.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out}")
