"""
sl_sensitivity_v2.py
====================
v75の generate_signals を流用した正しい損切り幅感度分析
- スプレッド: 0.4pips
- SL幅: ATR×0.15 / ×0.25 / ×0.35 / ×0.5（SL距離の倍率を変える）
- RR: 2.5固定
- エントリーロジック: v75と完全に同一（4h+1h 二番底・二番天井、1分足始値）

v75のSL設定: sl = min(low1, low2) - atr_val * 0.15
→ この 0.15 を 0.15 / 0.25 / 0.35 / 0.5 の4段階で比較する
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
SPREAD  = 0.4   # 0.4pips
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

# ===== v75のシグナル生成ロジックをSL倍率パラメータ付きで再実装 =====
def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close  = abs(df["low"]  - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_signals_with_sl(sl_buffer_mult, spread_pips, rr_ratio):
    """
    v75と同じロジックで、SLバッファ倍率のみを変えてシグナル生成
    sl_buffer_mult: SL = 直近二番底/二番天井 - atr * sl_buffer_mult
    """
    spread = spread_pips * 0.01

    d4h = data_4h.copy()
    d4h["atr"]   = calculate_atr(d4h, period=14)
    d4h["ema20"] = d4h["close"].ewm(span=20, adjust=False).mean()
    d4h["trend"] = np.where(d4h["close"] > d4h["ema20"], 1, -1)

    # 1時間足: 15分足から集約（v75と同じ）
    d1h = data_15m.resample("1h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    d1h.index = pd.to_datetime(d1h.index, utc=True)
    d1h["atr"] = calculate_atr(d1h, period=14)

    signals   = []
    used_times = set()

    # ── 4時間足の二番底・二番天井 ──
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

        # ロング: 二番底
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
                        ep   = entry_bar["open"] + spread
                        risk = ep - sl
                        if 0 < risk <= atr_val * 3:
                            tp = ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": 1,
                                "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                "tf": "4h", "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井
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
                        ep   = entry_bar["open"] - spread
                        risk = sl - ep
                        if 0 < risk <= atr_val * 3:
                            tp = ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": -1,
                                "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                "tf": "4h", "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    # ── 1時間足の二番底・二番天井 ──
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

        # ロング: 二番底
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
                        ep   = entry_bar["open"] + spread
                        risk = ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": 1,
                                "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                "tf": "1h", "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井
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
                        ep   = entry_bar["open"] - spread
                        risk = sl - ep
                        if 0 < risk <= h4_atr * 2:
                            tp = ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time, "dir": -1,
                                "ep": ep, "sl": sl, "tp": tp, "risk": risk,
                                "tf": "1h", "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals


def run_backtest(signals):
    """v75と同一のバックテストエンジン（1ポジション管理・半利確あり）"""
    sig_map = {s["time"]: s for s in signals}
    trades  = []
    pos     = None

    for i in range(len(data_1m)):
        bar = data_1m.iloc[i]
        t   = bar.name

        if pos is not None:
            d       = pos["dir"]
            half_tp = pos["ep"] + pos["risk"] if d == 1 else pos["ep"] - pos["risk"]

            # 半利確
            if not pos["half_closed"]:
                if (d == 1 and bar["high"] >= half_tp) or \
                   (d == -1 and bar["low"] <= half_tp):
                    pnl = pos["risk"] * 100
                    trades.append({**pos, "exit_time": t, "exit_price": half_tp,
                                   "pnl": pnl, "type": "HALF_TP"})
                    pos["sl"]          = pos["ep"]
                    pos["half_closed"] = True

            # 損切り
            if (d == 1 and bar["low"] <= pos["sl"]) or \
               (d == -1 and bar["high"] >= pos["sl"]):
                pnl = (pos["sl"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["sl"],
                               "pnl": pnl, "type": "SL"})
                pos = None
                continue

            # 利確
            if (d == 1 and bar["high"] >= pos["tp"]) or \
               (d == -1 and bar["low"] <= pos["tp"]):
                pnl = (pos["tp"] - pos["ep"]) * 100 * d
                trades.append({**pos, "exit_time": t, "exit_price": pos["tp"],
                               "pnl": pnl, "type": "TP"})
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
    n      = len(df[df["type"].isin(["SL", "TP"])])
    total  = df["pnl"].sum()
    nw     = len(wins)
    nl     = len(losses)
    avg_w  = wins["pnl"].mean()         if nw > 0 else 0
    avg_l  = abs(losses["pnl"].mean())  if nl > 0 else 0
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if nl > 0 and losses["pnl"].sum() != 0 else 999
    wr     = nw / (nw + nl) if (nw + nl) > 0 else 0
    kelly  = (pf * wr - (1 - wr)) / pf if pf > 0 and avg_l > 0 else 0

    df2 = df.copy()
    df2["exit_dt"] = pd.to_datetime(df2["exit_time"], utc=True)
    df2["month"]   = df2["exit_dt"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    plus_m  = (monthly > 0).sum()
    type_c  = df["type"].value_counts().to_dict()

    return {
        "n": n, "total": total, "wr": wr, "pf": pf, "kelly": kelly,
        "avg_w": avg_w, "avg_l": avg_l, "plus_months": plus_m,
        "monthly": monthly, "types": type_c,
        "avg_sl": df["risk"].mean() * 100 if "risk" in df.columns else 0
    }


# ===== 4段階バックテスト =====
sl_params = [
    (0.10, "×0.10（超タイト）"),
    (0.15, "×0.15（現行v75）"),
    (0.25, "×0.25（やや広め）"),
    (0.35, "×0.35（広め）"),
]

print(f"\n{'='*70}")
print(f"  損切り幅感度分析  スプレッド: {SPREAD}pips  RR: {RR}")
print(f"  ※ v75と完全同一ロジック（4h+1h 二番底・二番天井 / 1分足エントリー）")
print(f"{'='*70}")

all_stats = {}
all_trades = {}

for sl_mult, label in sl_params:
    print(f"\n  [{label}] シグナル生成中...")
    sigs = generate_signals_with_sl(sl_mult, SPREAD, RR)
    print(f"    シグナル数: {len(sigs)}本")
    df = run_backtest(sigs)
    df.to_csv(f"{OUT}/trades_sl_sens_{str(sl_mult).replace('.','')}.csv", index=False)
    s = calc_stats(df)
    all_stats[sl_mult]  = s
    all_trades[sl_mult] = df
    if s:
        print(f"    エントリー: {s['n']}回 | 総損益: {s['total']:+.1f}pips | 勝率: {s['wr']:.1%} | PF: {s['pf']:.2f} | ケリー: {s['kelly']:.4f}")

# ===== 結果表示 =====
print(f"\n{'='*70}")
print(f"  {'指標':<18}", end="")
for sl_mult, label in sl_params:
    print(f" {label[:8]:>14}", end="")
print()
print("  " + "-" * 70)

metrics = [
    ("総損益 (pips)",   "total",       "+.1f"),
    ("エントリー数",    "n",           "d"),
    ("勝率",           "wr",          ".1%"),
    ("PF",             "pf",          ".2f"),
    ("ケリー基準",     "kelly",       "+.4f"),
    ("平均利益 (pips)", "avg_w",      "+.1f"),
    ("平均損失 (pips)", "avg_l",      ".1f"),
    ("平均SL幅 (pips)", "avg_sl",     ".1f"),
    ("プラス月数",     "plus_months", "d"),
]

for label, key, fmt in metrics:
    print(f"  {label:<18}", end="")
    for sl_mult, _ in sl_params:
        s = all_stats.get(sl_mult)
        if s:
            val = s[key]
            formatted = format(val, fmt)
            print(f" {formatted:>14}", end="")
        else:
            print(f" {'N/A':>14}", end="")
    print()

print("  " + "-" * 70)

# 決済タイプ内訳
print("\n  【決済タイプ内訳】")
for sl_mult, label in sl_params:
    s = all_stats.get(sl_mult)
    if s:
        print(f"  SL×{sl_mult}: {s['types']}")

# 月別損益
print("\n  【月別損益 (pips)】")
all_months = set()
for sl_mult, _ in sl_params:
    s = all_stats.get(sl_mult)
    if s:
        all_months.update(s["monthly"].index.tolist())

print(f"  {'月':<12}", end="")
for sl_mult, label in sl_params:
    print(f" {label[:8]:>14}", end="")
print()

for m in sorted(all_months):
    print(f"  {str(m):<12}", end="")
    for sl_mult, _ in sl_params:
        s = all_stats.get(sl_mult)
        val = s["monthly"].get(m, 0) if s else 0
        print(f" {val:>+14.1f}", end="")
    print()

# ===== 可視化 =====
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f"損切り幅（SLバッファ倍率）別 バックテスト比較\nスプレッド {SPREAD}pips / RR {RR} / v75ロジック",
             fontsize=13, fontweight="bold")

colors = ["#E74C3C", "#2196F3", "#27AE60", "#FF9800"]
labels_short = [f"×{m}" for m, _ in sl_params]

# 1. 総損益
ax1 = axes[0, 0]
totals = [all_stats[m]["total"] if all_stats.get(m) else 0 for m, _ in sl_params]
bars = ax1.bar(labels_short, totals, color=colors, alpha=0.85)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("総損益 (pips)", fontweight="bold")
ax1.set_ylabel("pips")
for bar, val in zip(bars, totals):
    ypos = bar.get_height() + 20 if val >= 0 else bar.get_height() - 60
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:+.0f}", ha="center", fontsize=10, fontweight="bold")
ax1.grid(True, alpha=0.3)

# 2. 勝率・PF
ax2 = axes[0, 1]
wrs = [all_stats[m]["wr"] * 100 if all_stats.get(m) else 0 for m, _ in sl_params]
pfs = [all_stats[m]["pf"] if all_stats.get(m) else 0 for m, _ in sl_params]
x = np.arange(len(sl_params))
w = 0.35
ax2.bar(x - w/2, wrs, w, label="勝率 (%)", color=colors, alpha=0.8)
ax2t = ax2.twinx()
ax2t.bar(x + w/2, pfs, w, label="PF", color=colors, alpha=0.4, hatch="//")
ax2.set_title("勝率 vs PF", fontweight="bold")
ax2.set_ylabel("勝率 (%)")
ax2t.set_ylabel("PF")
ax2.set_xticks(x)
ax2.set_xticklabels(labels_short)
ax2.legend(loc="upper left", fontsize=9)
ax2t.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. 累積損益曲線
ax3 = axes[1, 0]
for (sl_mult, _), color in zip(sl_params, colors):
    df = all_trades.get(sl_mult)
    if df is not None and not df.empty:
        cum = df["pnl"].cumsum()
        ax3.plot(range(len(cum)), cum.values, color=color, lw=1.5, label=f"×{sl_mult}")
ax3.axhline(0, color="gray", lw=0.8, ls="--")
ax3.set_title("累積損益曲線", fontweight="bold")
ax3.set_ylabel("pips")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. 月別損益比較
ax4 = axes[1, 1]
months_list = sorted(all_months)
x = np.arange(len(months_list))
w = 0.2
for idx, ((sl_mult, _), color) in enumerate(zip(sl_params, colors)):
    s = all_stats.get(sl_mult)
    if s:
        vals = [s["monthly"].get(m, 0) for m in months_list]
        ax4.bar(x + idx * w - w * 1.5, vals, w, color=color, alpha=0.8, label=f"×{sl_mult}")
ax4.set_xticks(x)
ax4.set_xticklabels([str(m)[2:] for m in months_list], rotation=45, fontsize=8)
ax4.axhline(0, color="gray", lw=0.8)
ax4.set_title("月別損益比較", fontweight="bold")
ax4.set_ylabel("pips")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_path = f"{OUT}/v75_sl_sensitivity_v2.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Chart: {out_path}")
print("完了")
