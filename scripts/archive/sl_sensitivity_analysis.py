"""
損切り幅（ATR倍率）の段階別感度分析
スプレッド: 0.4pips
ATR倍率: 0.2 / 0.3 / 0.4 / 0.5
RR: 2.5固定
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")

# 日本語フォント設定
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
except:
    pass

# ===== データ読み込み =====
DATA = "/home/ubuntu/sena3fx/data"
OUT  = "/home/ubuntu/sena3fx/results"

d1m  = pd.read_csv(f"{DATA}/usdjpy_1m.csv",  parse_dates=["timestamp"], index_col="timestamp")
d15m = pd.read_csv(f"{DATA}/usdjpy_15m.csv", parse_dates=["timestamp"], index_col="timestamp")
d1h  = pd.read_csv(f"{DATA}/usdjpy_1h.csv",  parse_dates=["timestamp"], index_col="timestamp")
d4h  = pd.read_csv(f"{DATA}/usdjpy_4h.csv",  parse_dates=["timestamp"], index_col="timestamp")

for df in [d1m, d15m, d1h, d4h]:
    df.index = pd.to_datetime(df.index, utc=True)

# 4時間足EMA20
d4h["ema20"] = d4h["close"].ewm(span=20, adjust=False).mean()
# 4時間足ATR14
d4h["tr"] = np.maximum(d4h["high"] - d4h["low"],
            np.maximum(abs(d4h["high"] - d4h["close"].shift(1)),
                       abs(d4h["low"]  - d4h["close"].shift(1))))
d4h["atr14"] = d4h["tr"].ewm(span=14, adjust=False).mean()

# 1時間足ATR14
d1h["tr"] = np.maximum(d1h["high"] - d1h["low"],
            np.maximum(abs(d1h["high"] - d1h["close"].shift(1)),
                       abs(d1h["low"]  - d1h["close"].shift(1))))
d1h["atr14"] = d1h["tr"].ewm(span=14, adjust=False).mean()

SPREAD = 0.04  # 0.4pips（円換算）
RR     = 2.5
HALF_R = 1.0   # 半利確は1R

def find_swing_low(series, lookback=5):
    """直近lookback本の最安値"""
    return series.rolling(lookback).min().iloc[-1] if len(series) >= lookback else series.min()

def find_swing_high(series, lookback=5):
    """直近lookback本の最高値"""
    return series.rolling(lookback).max().iloc[-1] if len(series) >= lookback else series.max()

def run_backtest(sl_atr_mult):
    """指定ATR倍率でバックテスト実行"""
    trades = []
    
    # 4時間足・1時間足の更新タイミングを検出
    for tf_name, d_tf in [("4h", d4h), ("1h", d1h)]:
        for i in range(20, len(d_tf) - 1):
            bar     = d_tf.iloc[i]
            bar_time = d_tf.index[i]
            next_time = d_tf.index[i + 1]
            
            # 4時間足EMA20でトレンド判定
            try:
                h4_bar = d4h.asof(bar_time)
                trend_up   = h4_bar["close"] > h4_bar["ema20"]
                trend_down = h4_bar["close"] < h4_bar["ema20"]
                atr_4h     = h4_bar["atr14"]
            except:
                continue
            
            if atr_4h < 0.1:
                continue
            
            # 二番底・二番天井の検出（直近10本）
            if tf_name == "4h":
                lookback_bars = d4h.iloc[max(0, i-10):i]
            else:
                lookback_bars = d1h.iloc[max(0, i-10):i]
            
            if len(lookback_bars) < 5:
                continue
            
            # ロング：二番底（直近安値圏）
            if trend_up:
                recent_low = lookback_bars["low"].min()
                current_low = bar["low"]
                # 二番底の条件：現在の安値が直近安値の±ATR×0.5以内
                if abs(current_low - recent_low) > atr_4h * 0.5:
                    continue
                if bar["close"] <= bar["open"]:  # 陰線はスキップ
                    continue
                direction = "long"
                
            # ショート：二番天井（直近高値圏）
            elif trend_down:
                recent_high = lookback_bars["high"].max()
                current_high = bar["high"]
                if abs(current_high - recent_high) > atr_4h * 0.5:
                    continue
                if bar["close"] >= bar["open"]:  # 陽線はスキップ
                    continue
                direction = "short"
            else:
                continue
            
            # 更新から2分以内の1分足でエントリー
            entry_window = d1m.loc[
                (d1m.index >= next_time) &
                (d1m.index < next_time + pd.Timedelta(minutes=2))
            ]
            if len(entry_window) == 0:
                continue
            
            entry_bar   = entry_window.iloc[0]
            entry_time  = entry_window.index[0]
            
            if direction == "long":
                entry_price = entry_bar["open"] + SPREAD
            else:
                entry_price = entry_bar["open"] - SPREAD
            
            # SL/TP設定（ATR倍率で変化）
            sl_dist = atr_4h * sl_atr_mult
            half_tp_dist = sl_dist * HALF_R
            tp_dist      = sl_dist * RR
            
            if direction == "long":
                sl_price      = entry_price - sl_dist
                half_tp_price = entry_price + half_tp_dist
                tp_price      = entry_price + tp_dist
            else:
                sl_price      = entry_price + sl_dist
                half_tp_price = entry_price - half_tp_dist
                tp_price      = entry_price - tp_dist
            
            # 決済シミュレーション（1分足で追跡）
            future_1m = d1m.loc[d1m.index > entry_time]
            # 次の4時間足更新まで
            next_4h_times = d4h.index[d4h.index > entry_time]
            if len(next_4h_times) == 0:
                continue
            expire_time = next_4h_times[0]
            future_1m = future_1m.loc[future_1m.index <= expire_time]
            
            half_done = False
            result_type = None
            exit_price  = None
            exit_time   = None
            
            for j, (t, m) in enumerate(future_1m.iterrows()):
                if direction == "long":
                    # SL判定
                    if m["low"] <= sl_price:
                        if not half_done:
                            result_type = "SL"
                            exit_price  = sl_price
                        else:
                            result_type = "SL_AFTER_HALF"
                            exit_price  = entry_price  # 建値SL
                        exit_time = t
                        break
                    # 半利確
                    if not half_done and m["high"] >= half_tp_price:
                        half_done = True
                    # 全利確
                    if half_done and m["high"] >= tp_price:
                        result_type = "TP"
                        exit_price  = tp_price
                        exit_time   = t
                        break
                else:
                    if m["high"] >= sl_price:
                        if not half_done:
                            result_type = "SL"
                            exit_price  = sl_price
                        else:
                            result_type = "SL_AFTER_HALF"
                            exit_price  = entry_price
                        exit_time = t
                        break
                    if not half_done and m["low"] <= half_tp_price:
                        half_done = True
                    if half_done and m["low"] <= tp_price:
                        result_type = "TP"
                        exit_price  = tp_price
                        exit_time   = t
                        break
            
            if result_type is None:
                # 有効期限切れ（4時間足更新）
                if len(future_1m) > 0:
                    last_bar = future_1m.iloc[-1]
                    if half_done:
                        result_type = "EXPIRE_HALF"
                        exit_price  = last_bar["close"]
                    else:
                        result_type = "EXPIRE"
                        exit_price  = last_bar["close"]
                    exit_time = future_1m.index[-1]
                else:
                    continue
            
            # P&L計算（pips単位）
            if direction == "long":
                raw_pnl = (exit_price - entry_price) * 100
            else:
                raw_pnl = (entry_price - exit_price) * 100
            
            # 半利確分の処理
            if half_done:
                half_pnl = half_tp_dist * 100 * 0.5  # 半分を1Rで利確
                if result_type in ("TP",):
                    remain_pnl = tp_dist * 100 * 0.5
                elif result_type == "SL_AFTER_HALF":
                    remain_pnl = 0  # 建値SL
                else:
                    if direction == "long":
                        remain_pnl = (exit_price - entry_price) * 100 * 0.5
                    else:
                        remain_pnl = (entry_price - exit_price) * 100 * 0.5
                total_pnl = half_pnl + remain_pnl
                dec_type  = "HALF_TP" if result_type == "TP" else result_type
            else:
                if result_type == "SL":
                    total_pnl = -sl_dist * 100
                elif result_type == "TP":
                    total_pnl = tp_dist * 100
                else:
                    total_pnl = raw_pnl
                dec_type = result_type
            
            trades.append({
                "entry_time":  entry_time,
                "exit_time":   exit_time,
                "direction":   direction,
                "tf":          tf_name,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "sl_dist":     sl_dist * 100,
                "tp_dist":     tp_dist * 100,
                "pnl":         total_pnl,
                "type":        dec_type,
                "half_done":   half_done,
            })
    
    return pd.DataFrame(trades)

# ===== 4段階バックテスト実行 =====
sl_multiples = [0.2, 0.3, 0.4, 0.5]
results = {}

print("=" * 70)
print(f"  損切り幅感度分析  スプレッド: 0.4pips  RR: {RR}")
print("=" * 70)
print(f"  {'指標':<20} {'×0.2':>10} {'×0.3':>10} {'×0.4':>10} {'×0.5':>10}")
print("  " + "-" * 60)

for mult in sl_multiples:
    df = run_backtest(mult)
    df.to_csv(f"{OUT}/trades_v75_sl{str(mult).replace('.','')}.csv", index=False)
    results[mult] = df

# 統計集計
stats = {}
for mult, df in results.items():
    if len(df) == 0:
        continue
    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    total  = df["pnl"].sum()
    n      = len(df)
    wr     = len(wins) / n if n > 0 else 0
    avg_w  = wins["pnl"].mean()  if len(wins)   > 0 else 0
    avg_l  = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
    pf     = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) > 0 else 999
    kelly  = wr - (1 - wr) / (avg_w / avg_l) if avg_l > 0 else 0
    avg_sl = df["sl_dist"].mean()
    
    # 月別
    df2 = df.copy()
    df2["exit_dt"] = pd.to_datetime(df2["exit_time"], utc=True)
    df2["month"]   = df2["exit_dt"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    plus_months = (monthly > 0).sum()
    
    # 決済タイプ
    type_counts = df["type"].value_counts().to_dict()
    
    stats[mult] = {
        "n": n, "total": total, "wr": wr, "pf": pf, "kelly": kelly,
        "avg_w": avg_w, "avg_l": avg_l, "avg_sl": avg_sl,
        "plus_months": plus_months, "types": type_counts
    }

# 結果表示
metrics = [
    ("総損益 (pips)", "total", "+.1f"),
    ("エントリー数", "n", "d"),
    ("勝率", "wr", ".1%"),
    ("PF", "pf", ".2f"),
    ("ケリー基準", "kelly", ".4f"),
    ("平均利益 (pips)", "avg_w", "+.1f"),
    ("平均損失 (pips)", "avg_l", ".1f"),
    ("平均SL幅 (pips)", "avg_sl", ".1f"),
    ("プラス月数", "plus_months", "d"),
]

for label, key, fmt in metrics:
    row = f"  {label:<20}"
    for mult in sl_multiples:
        if mult in stats:
            val = stats[mult][key]
            formatted = format(val, fmt)
            row += f" {formatted:>10}"
        else:
            row += f" {'N/A':>10}"
    print(row)

print("  " + "-" * 60)
print("\n  【決済タイプ内訳】")
for mult in sl_multiples:
    if mult in stats:
        print(f"  ATR×{mult}: {stats[mult]['types']}")

# ===== 月別損益詳細 =====
print("\n  【月別損益（pips）】")
print(f"  {'月':<12}", end="")
for mult in sl_multiples:
    print(f" {'ATR×'+str(mult):>12}", end="")
print()

all_months = set()
monthly_data = {}
for mult, df in results.items():
    df2 = df.copy()
    df2["exit_dt"] = pd.to_datetime(df2["exit_time"], utc=True)
    df2["month"]   = df2["exit_dt"].dt.to_period("M")
    monthly = df2.groupby("month")["pnl"].sum()
    monthly_data[mult] = monthly
    all_months.update(monthly.index.tolist())

for m in sorted(all_months):
    print(f"  {str(m):<12}", end="")
    for mult in sl_multiples:
        val = monthly_data[mult].get(m, 0)
        print(f" {val:>+12.1f}", end="")
    print()

# ===== 可視化 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("損切り幅（ATR倍率）別 バックテスト比較\nスプレッド 0.4pips / RR 2.5", fontsize=14, fontweight="bold")

colors = ["#E74C3C", "#F39C12", "#27AE60", "#2980B9"]

# 1. 総損益比較
ax1 = axes[0, 0]
totals = [stats[m]["total"] for m in sl_multiples if m in stats]
bars = ax1.bar([f"ATR×{m}" for m in sl_multiples], totals, color=colors)
ax1.set_title("総損益 (pips)", fontweight="bold")
ax1.set_ylabel("pips")
for bar, val in zip(bars, totals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"+{val:.0f}", ha="center", fontsize=10, fontweight="bold")
ax1.axhline(0, color="black", linewidth=0.5)

# 2. 勝率・PF比較
ax2 = axes[0, 1]
wrs = [stats[m]["wr"] * 100 for m in sl_multiples if m in stats]
pfs = [stats[m]["pf"] for m in sl_multiples if m in stats]
x = np.arange(len(sl_multiples))
w = 0.35
bars1 = ax2.bar(x - w/2, wrs, w, label="勝率 (%)", color=colors, alpha=0.8)
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + w/2, pfs, w, label="PF", color=colors, alpha=0.5, hatch="//")
ax2.set_title("勝率 vs プロフィットファクター", fontweight="bold")
ax2.set_ylabel("勝率 (%)")
ax2_twin.set_ylabel("PF")
ax2.set_xticks(x)
ax2.set_xticklabels([f"ATR×{m}" for m in sl_multiples])
ax2.legend(loc="upper left")
ax2_twin.legend(loc="upper right")

# 3. 月別損益ヒートマップ的な棒グラフ
ax3 = axes[1, 0]
months_list = sorted(all_months)
x = np.arange(len(months_list))
w = 0.2
for i, mult in enumerate(sl_multiples):
    vals = [monthly_data[mult].get(m, 0) for m in months_list]
    ax3.bar(x + i*w - 0.3, vals, w, label=f"ATR×{mult}", color=colors[i], alpha=0.8)
ax3.set_title("月別損益比較 (pips)", fontweight="bold")
ax3.set_ylabel("pips")
ax3.set_xticks(x)
ax3.set_xticklabels([str(m)[2:] for m in months_list], rotation=45, fontsize=8)
ax3.axhline(0, color="black", linewidth=0.5)
ax3.legend(fontsize=8)

# 4. ケリー基準・平均SL幅
ax4 = axes[1, 1]
kellys = [stats[m]["kelly"] for m in sl_multiples if m in stats]
avg_sls = [stats[m]["avg_sl"] for m in sl_multiples if m in stats]
ax4.bar([f"ATR×{m}" for m in sl_multiples], kellys, color=colors, alpha=0.8)
ax4.set_title("ケリー基準", fontweight="bold")
ax4.set_ylabel("ケリー係数")
for i, (bar_x, val, sl) in enumerate(zip(sl_multiples, kellys, avg_sls)):
    ax4.text(i, val + 0.005, f"SL≈{sl:.0f}p", ha="center", fontsize=9)
ax4.axhline(0.3, color="red", linewidth=1, linestyle="--", label="ハーフケリー基準(0.3)")
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/v75_sl_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart: {OUT}/v75_sl_sensitivity.png")
print("完了")
