"""
sl_sensitivity_v76.py
=====================
v76（スプレッド正しく実装）を使った損切り幅感度分析
- スプレッド: 0.4pips（実運用環境）
- SL幅: ATR×0.10 / ×0.15 / ×0.25 / ×0.35
- RR: 2.5固定
- スプレッドはエントリーコストとして損益に正しく反映
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.append('/home/ubuntu/sena3fx/strategies')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
except:
    pass

SYMBOL  = "USDJPY"
START   = "2024-07-01"
END     = "2025-02-06"
SPREAD  = 0.4
RR      = 2.5
OUT     = "/home/ubuntu/sena3fx/results"
DATA    = "/home/ubuntu/sena3fx/data"
os.makedirs(OUT, exist_ok=True)

# ===== データ読み込み =====
def load_data(timeframe):
    path = f"{DATA}/{SYMBOL.lower()}_{timeframe}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df[(df.index >= START) & (df.index <= END)]

data_1m  = load_data("1m")
data_15m = load_data("15m")
data_4h  = load_data("4h")
print(f"データ読み込み完了: 1m={len(data_1m)}本, 15m={len(data_15m)}本, 4h={len(data_4h)}本")

# ===== ATR計算 =====
def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ===== v76ロジック（SL倍率パラメータ付き） =====
def generate_signals_v76(sl_buffer_mult, spread_pips, rr_ratio):
    """
    v76と同じロジック（スプレッドをエントリーコストとして正しく反映）
    SL/TPはチャートレベル（始値基準）で固定
    """
    spread = spread_pips * 0.01

    d4h = data_4h.copy()
    d4h["atr"]   = calculate_atr(d4h, period=14)
    d4h["ema20"] = d4h["close"].ewm(span=20, adjust=False).mean()
    d4h["trend"] = np.where(d4h["close"] > d4h["ema20"], 1, -1)

    d1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    d1h.index = pd.to_datetime(d1h.index, utc=True)
    d1h["atr"] = calculate_atr(d1h, period=14)

    signals    = []
    used_times = set()

    # 4時間足
    h4_times = d4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_prev1   = d4h.iloc[i - 1]
        h4_prev2   = d4h.iloc[i - 2]
        h4_current = d4h.iloc[i]

        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        trend     = h4_current["trend"]
        tolerance = atr_val * 0.3

        # ロング
        if trend == 1:
            low1 = h4_prev2["low"]
            low2 = h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                sl = min(low1, low2) - atr_val * sl_buffer_mult
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep + spread
                        risk   = raw_ep - sl
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": 1,
                                "ep": ep, "sl": sl, "tp": tp,
                                "risk": risk, "spread": spread,
                                "tf": "4h", "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート
        if trend == -1:
            high1 = h4_prev2["high"]
            high2 = h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                sl = max(high1, high2) + atr_val * sl_buffer_mult
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep - spread
                        risk   = sl - raw_ep
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": -1,
                                "ep": ep, "sl": sl, "tp": tp,
                                "risk": risk, "spread": spread,
                                "tf": "4h", "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    # 1時間足
    h1_times = d1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1   = d1h.iloc[i - 1]
        h1_prev2   = d1h.iloc[i - 2]
        h1_current = d1h.iloc[i]

        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        h4_before = d4h[d4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        trend  = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        tolerance = atr_val * 0.3

        # ロング
        if trend == 1:
            low1 = h1_prev2["low"]
            low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                sl = min(low1, low2) - atr_val * sl_buffer_mult
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep + spread
                        risk   = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": 1,
                                "ep": ep, "sl": sl, "tp": tp,
                                "risk": risk, "spread": spread,
                                "tf": "1h", "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート
        if trend == -1:
            high1 = h1_prev2["high"]
            high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                sl = max(high1, high2) + atr_val * sl_buffer_mult
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index <  entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar  = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep     = raw_ep - spread
                        risk   = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": -1,
                                "ep": ep, "sl": sl, "tp": tp,
                                "risk": risk, "spread": spread,
                                "tf": "1h", "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals


def run_backtest(signals):
    """1トレード=1エントリーとして正しく集計"""
    sig_map = {s["time"]: s for s in signals}
    trades  = []
    pos     = None

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name

        if pos is not None:
            d       = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] * d   # チャートレベル1R

            # 半利確
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or \
                   (d == -1 and bar["low"] <= half_tp):
                    pos["half_pnl"] = (half_tp - pos["ep"]) * 100 * d
                    pos["sl"]       = pos["ep"]
                    pos["half_closed"] = True

            # 損切り
            if (d == 1 and bar["low"] <= pos["sl"]) or \
               (d == -1 and bar["high"] >= pos["sl"]):
                sl_pnl    = (pos["sl"] - pos["ep"]) * 100 * d
                total_pnl = pos.get("half_pnl", 0) + sl_pnl
                exit_type = "HALF_TP+SL" if pos["half_closed"] else "SL"
                trades.append({**pos, "exit_time": t, "pnl": total_pnl, "type": exit_type})
                pos = None
                continue

            # 利確
            if (d == 1 and bar["high"] >= pos["tp"]) or \
               (d == -1 and bar["low"] <= pos["tp"]):
                tp_pnl    = (pos["tp"] - pos["ep"]) * 100 * d
                total_pnl = pos.get("half_pnl", 0) + tp_pnl
                exit_type = "HALF_TP+TP" if pos["half_closed"] else "TP"
                trades.append({**pos, "exit_time": t, "pnl": total_pnl, "type": exit_type})
                pos = None
                continue

        if pos is None and t in sig_map:
            s   = sig_map[t]
            pos = {**s, "entry_time": t, "half_closed": False}

    return pd.DataFrame(trades)


def calc_stats(df):
    if df.empty:
        return None
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    n      = len(df)
    total  = df["pnl"].sum()
    wr     = len(wins) / n * 100
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    avg_w  = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_l  = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
    avg_sl = df["risk"].mean() * 100
    kelly  = (wr/100) - (1 - wr/100) / (avg_w / avg_l) if avg_l > 0 else 0
    monthly = df.copy()
    monthly["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    monthly_pnl = monthly.groupby("month")["pnl"].sum()
    plus_months = (monthly_pnl > 0).sum()
    return {
        "n": n, "total": total, "wr": wr, "pf": pf,
        "avg_w": avg_w, "avg_l": avg_l, "avg_sl": avg_sl,
        "kelly": kelly, "monthly_pnl": monthly_pnl,
        "plus_months": plus_months, "total_months": len(monthly_pnl),
        "types": df["type"].value_counts().to_dict()
    }


# ===== 感度分析実行 =====
configs = [
    (0.10, "×0.10（超タイト）"),
    (0.15, "×0.15（現行v75/v76）"),
    (0.25, "×0.25（やや広め）"),
    (0.35, "×0.35（広め）"),
]

print("=" * 70)
print(f"  損切り幅感度分析  スプレッド: {SPREAD}pips  RR: {RR}")
print(f"  ※ v76ロジック（スプレッドをエントリーコストとして正しく反映）")
print("=" * 70)

results = {}
for mult, label in configs:
    print(f"  [{label}] シグナル生成中...")
    signals = generate_signals_v76(mult, SPREAD, RR)
    df_t    = run_backtest(signals)
    st      = calc_stats(df_t)
    results[label] = st
    if st:
        print(f"    シグナル数: {len(signals)}本")
        print(f"    エントリー: {st['n']}回 | 総損益: {st['total']:+.1f}pips | "
              f"勝率: {st['wr']:.1f}% | PF: {st['pf']:.2f} | ケリー: {st['kelly']:.4f}")

# ===== 比較表 =====
labels = [c[1] for c in configs]
metrics = ["総損益 (pips)", "エントリー数", "勝率", "PF", "ケリー基準",
           "平均利益 (pips)", "平均損失 (pips)", "平均SL幅 (pips)", "プラス月数"]

print("\n" + "=" * 70)
header = f"  {'指標':<22}" + "".join(f"{l[:10]:<14}" for l in labels)
print(header)
print("  " + "-" * 68)

for m in metrics:
    row = f"  {m:<22}"
    for label in labels:
        st = results[label]
        if st is None:
            row += f"{'N/A':<14}"
            continue
        if m == "総損益 (pips)":
            row += f"{st['total']:+.1f}".ljust(14)
        elif m == "エントリー数":
            row += f"{st['n']}".ljust(14)
        elif m == "勝率":
            row += f"{st['wr']:.1f}%".ljust(14)
        elif m == "PF":
            row += f"{st['pf']:.2f}".ljust(14)
        elif m == "ケリー基準":
            row += f"{st['kelly']:+.4f}".ljust(14)
        elif m == "平均利益 (pips)":
            row += f"+{st['avg_w']:.1f}".ljust(14)
        elif m == "平均損失 (pips)":
            row += f"{st['avg_l']:.1f}".ljust(14)
        elif m == "平均SL幅 (pips)":
            row += f"{st['avg_sl']:.1f}".ljust(14)
        elif m == "プラス月数":
            row += f"{st['plus_months']}/{st['total_months']}".ljust(14)
    print(row)

print("\n  【決済タイプ内訳】")
for label in labels:
    st = results[label]
    if st:
        print(f"  SL{label[:5]}: {st['types']}")

print("\n  【月別損益 (pips)】")
months = sorted(set(
    m for label in labels
    for m in results[label]["monthly_pnl"].index
    if results[label] is not None
))
header2 = f"  {'月':<12}" + "".join(f"{l[:10]:<14}" for l in labels)
print(header2)
for m in months:
    row = f"  {str(m):<12}"
    for label in labels:
        st = results[label]
        if st and m in st["monthly_pnl"].index:
            v = st["monthly_pnl"][m]
            row += f"{v:+.1f}".ljust(14)
        else:
            row += f"{'N/A':<14}"
    print(row)

# ===== グラフ =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"v76 損切り幅感度分析\nスプレッド {SPREAD}pips / RR {RR} / USDJPY",
             fontsize=14, fontweight="bold")

colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

# 1. 総損益比較
ax = axes[0, 0]
totals = [results[l]["total"] for l in labels]
bars = ax.bar([l[:8] for l in labels], totals, color=colors, alpha=0.8, edgecolor="white")
ax.set_title("総損益 (pips)", fontweight="bold")
ax.set_ylabel("pips")
ax.axhline(0, color="black", linewidth=0.8)
for bar, v in zip(bars, totals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{v:+.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# 2. PF・ケリー比較
ax = axes[0, 1]
pfs    = [results[l]["pf"] for l in labels]
kellys = [results[l]["kelly"] for l in labels]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, pfs, w, label="PF", color=colors, alpha=0.8, edgecolor="white")
ax2 = ax.twinx()
ax2.bar(x + w/2, kellys, w, label="ケリー基準", color=colors, alpha=0.4,
        edgecolor="white", hatch="//")
ax.set_title("PF・ケリー基準", fontweight="bold")
ax.set_ylabel("PF")
ax2.set_ylabel("ケリー基準")
ax.set_xticks(x)
ax.set_xticklabels([l[:8] for l in labels])
ax.axhline(1, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)

# 3. 月別損益推移
ax = axes[1, 0]
for i, (label, color) in enumerate(zip(labels, colors)):
    st = results[label]
    mp = st["monthly_pnl"]
    ax.plot([str(m) for m in mp.index], mp.values, marker="o", color=color,
            label=label[:8], linewidth=2, markersize=5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("月別損益推移 (pips)", fontweight="bold")
ax.set_ylabel("pips")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3)

# 4. 勝率・エントリー数
ax = axes[1, 1]
wrs = [results[l]["wr"] for l in labels]
ns  = [results[l]["n"] for l in labels]
x   = np.arange(len(labels))
ax.bar(x, wrs, color=colors, alpha=0.8, edgecolor="white")
ax.set_title("勝率 & エントリー数", fontweight="bold")
ax.set_ylabel("勝率 (%)")
ax.set_xticks(x)
ax.set_xticklabels([l[:8] for l in labels])
ax.set_ylim(0, 100)
ax2 = ax.twinx()
ax2.plot(x, ns, marker="s", color="black", linewidth=2, markersize=8, label="エントリー数")
ax2.set_ylabel("エントリー数")
for xi, n in zip(x, ns):
    ax2.text(xi, n + 0.5, str(n), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = f"{OUT}/v76_sl_sensitivity.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart: {out_path}")
print("完了")
