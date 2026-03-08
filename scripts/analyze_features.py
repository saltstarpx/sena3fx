"""
analyze_features.py
====================
◎採用5銘柄（EURUSD/GBPUSD/AUDUSD/USDCHF/EURGBP）の取引データを
定量・計量分析し、シャープレシオ改善につながる特徴量を発見する

分析項目:
1. 時間帯別 勝率・期待値
2. 曜日別 勝率・期待値
3. ATR比率（SL距離/ATR）別 勝率
4. EMA乖離率別 勝率
5. 直前N連勝/連敗後の勝率（ストリーク効果）
6. 時間足別（4H vs 1H）勝率
7. ボラティリティレジーム（高/中/低ATR）別 勝率
8. 連続エントリー間隔（前回シグナルからの経過時間）別 勝率
9. 相関分析（特徴量 × pnl）
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
OUT_DIR  = os.path.join(BASE_DIR, "results")

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

PAIRS = ["eurusd", "gbpusd", "audusd", "usdchf", "eurgbp"]

# ── データ読み込み ────────────────────────────────────────
dfs = []
for p in PAIRS:
    path = os.path.join(OUT_DIR, f"longterm_{p}.csv")
    if not os.path.exists(path):
        print(f"  SKIP: {path}")
        continue
    df = pd.read_csv(path)
    df["pair"] = p.upper()
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df["entry_time"] = pd.to_datetime(all_df["entry_time"], utc=True)
all_df["exit_time"]  = pd.to_datetime(all_df["exit_time"],  utc=True)

# 基本フィルタ: OPENを除外
all_df = all_df[all_df["result"].isin(["TP","SL","BE"])].copy()

# 勝敗バイナリ（TP=1, SL/BE=0）
all_df["win"] = (all_df["result"] == "TP").astype(int)
# pnl_sign（正=1, 負=0）
all_df["pnl_pos"] = (all_df["pnl"] > 0).astype(int)

print(f"総トレード数: {len(all_df)}")
print(f"TP: {(all_df['result']=='TP').sum()} / SL: {(all_df['result']=='SL').sum()} / BE: {(all_df['result']=='BE').sum()}")
print(f"全体勝率(TP): {all_df['win'].mean()*100:.1f}%")
print(f"全体pnl+率: {all_df['pnl_pos'].mean()*100:.1f}%")
print(f"全体平均PnL: {all_df['pnl'].mean():.0f}円")
print()

# ── 特徴量生成 ────────────────────────────────────────────
# 1. 時間帯（UTC）
all_df["hour_utc"] = all_df["entry_time"].dt.hour
# 2. 曜日
all_df["weekday"] = all_df["entry_time"].dt.dayofweek  # 0=月
# 3. SL距離/ATR比（risk = SL距離）
all_df["risk_pips"] = abs(all_df["ep"] - all_df["sl"])
# 4. 時間足
all_df["tf"] = all_df.get("tf", "4h")
# 5. 保有時間（分）
all_df["hold_min"] = (all_df["exit_time"] - all_df["entry_time"]).dt.total_seconds() / 60
# 6. 直前シグナルからの間隔（銘柄ごと）
all_df = all_df.sort_values(["pair","entry_time"])
all_df["prev_entry"] = all_df.groupby("pair")["entry_time"].shift(1)
all_df["gap_min"] = (all_df["entry_time"] - all_df["prev_entry"]).dt.total_seconds() / 60
# 7. ストリーク（直前の結果）
all_df["prev_win"] = all_df.groupby("pair")["win"].shift(1)
all_df["prev2_win"] = all_df.groupby("pair")["win"].shift(2)
all_df["streak"] = all_df["prev_win"].fillna(0.5)

# ── 分析 ──────────────────────────────────────────────────
results = {}

# 1. 時間帯別
hourly = all_df.groupby("hour_utc").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean"), pnl_pos=("pnl_pos","mean")
).reset_index()
hourly["expected_r"] = hourly["wr"] * 1.75 + (1 - hourly["wr"]) * (-1.0)
results["hourly"] = hourly
print("=== 時間帯別 勝率・期待値 ===")
print(hourly[["hour_utc","n","wr","avg_pnl","expected_r"]].to_string(index=False))
print()

# 2. 曜日別
weekday_names = ["月","火","水","木","金","土","日"]
weekly = all_df.groupby("weekday").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean"), pnl_pos=("pnl_pos","mean")
).reset_index()
weekly["day_name"] = weekly["weekday"].map(lambda x: weekday_names[x])
weekly["expected_r"] = weekly["wr"] * 1.75 + (1 - weekly["wr"]) * (-1.0)
results["weekly"] = weekly
print("=== 曜日別 勝率・期待値 ===")
print(weekly[["day_name","n","wr","avg_pnl","expected_r"]].to_string(index=False))
print()

# 3. 時間足別
tf_stats = all_df.groupby("tf").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean")
).reset_index()
print("=== 時間足別 ===")
print(tf_stats.to_string(index=False))
print()

# 4. 保有時間別（分位数）
all_df["hold_q"] = pd.qcut(all_df["hold_min"].clip(0, 10000), q=5, labels=["Q1短","Q2","Q3","Q4","Q5長"])
hold_stats = all_df.groupby("hold_q").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean"),
    hold_med=("hold_min","median")
).reset_index()
print("=== 保有時間別（分位数） ===")
print(hold_stats.to_string(index=False))
print()

# 5. ギャップ（前回エントリーからの間隔）別
all_df["gap_q"] = pd.cut(all_df["gap_min"].clip(0, 2000),
    bins=[0, 60, 240, 480, 1440, 99999],
    labels=["<1H","1-4H","4-8H","8-24H",">24H"])
gap_stats = all_df.groupby("gap_q").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean")
).reset_index()
print("=== 前回エントリーからの間隔別 ===")
print(gap_stats.to_string(index=False))
print()

# 6. ストリーク別（直前勝敗）
streak_stats = all_df.groupby("prev_win").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean")
).reset_index()
streak_stats["prev_result"] = streak_stats["prev_win"].map({0:"前回負け", 1:"前回勝ち", 0.5:"初回"})
print("=== 直前勝敗別 ===")
print(streak_stats[["prev_result","n","wr","avg_pnl"]].to_string(index=False))
print()

# 7. 銘柄別詳細
pair_stats = all_df.groupby("pair").agg(
    n=("win","count"), wr=("win","mean"), avg_pnl=("pnl","mean"),
    pnl_std=("pnl","std"), sharpe=("pnl", lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0)
).reset_index()
print("=== 銘柄別シャープレシオ ===")
print(pair_stats.to_string(index=False))
print()

# 8. 時間帯フィルター効果: 期待値が負の時間帯を除外した場合
bad_hours = hourly[hourly["expected_r"] < -0.15]["hour_utc"].tolist()
good_hours = hourly[hourly["expected_r"] > 0.0]["hour_utc"].tolist()
print(f"期待値が低い時間帯（除外候補）: {bad_hours}")
print(f"期待値が高い時間帯（採用候補）: {good_hours}")

filtered = all_df[~all_df["hour_utc"].isin(bad_hours)]
print(f"\n時間帯フィルター適用後:")
print(f"  件数: {len(filtered)} (-{len(all_df)-len(filtered)}件)")
print(f"  勝率: {filtered['win'].mean()*100:.1f}% (元: {all_df['win'].mean()*100:.1f}%)")
print(f"  平均PnL: {filtered['pnl'].mean():.0f}円 (元: {all_df['pnl'].mean():.0f}円)")
sharpe_before = all_df["pnl"].mean() / all_df["pnl"].std() * np.sqrt(252)
sharpe_after  = filtered["pnl"].mean() / filtered["pnl"].std() * np.sqrt(252)
print(f"  シャープレシオ: {sharpe_after:.3f} (元: {sharpe_before:.3f})")
print()

# 9. 曜日フィルター効果
bad_days = weekly[weekly["expected_r"] < -0.1]["weekday"].tolist()
print(f"期待値が低い曜日（除外候補）: {[weekday_names[d] for d in bad_days]}")
filtered_day = all_df[~all_df["weekday"].isin(bad_days)]
print(f"曜日フィルター適用後:")
print(f"  件数: {len(filtered_day)} (-{len(all_df)-len(filtered_day)}件)")
sharpe_day = filtered_day["pnl"].mean() / filtered_day["pnl"].std() * np.sqrt(252)
print(f"  シャープレシオ: {sharpe_day:.3f}")
print()

# 10. ギャップフィルター（<1H除外）
filtered_gap = all_df[all_df["gap_min"].isna() | (all_df["gap_min"] >= 60)]
print(f"60分未満ギャップ除外後:")
print(f"  件数: {len(filtered_gap)} (-{len(all_df)-len(filtered_gap)}件)")
sharpe_gap = filtered_gap["pnl"].mean() / filtered_gap["pnl"].std() * np.sqrt(252)
print(f"  シャープレシオ: {sharpe_gap:.3f}")
print()

# 11. 組み合わせフィルター
combined = all_df[
    (~all_df["hour_utc"].isin(bad_hours)) &
    (~all_df["weekday"].isin(bad_days)) &
    (all_df["gap_min"].isna() | (all_df["gap_min"] >= 60))
]
print(f"組み合わせフィルター（時間帯+曜日+ギャップ）:")
print(f"  件数: {len(combined)} (-{len(all_df)-len(combined)}件, -{(len(all_df)-len(combined))/len(all_df)*100:.1f}%)")
print(f"  勝率: {combined['win'].mean()*100:.1f}%")
print(f"  平均PnL: {combined['pnl'].mean():.0f}円")
sharpe_combined = combined["pnl"].mean() / combined["pnl"].std() * np.sqrt(252)
print(f"  シャープレシオ: {sharpe_combined:.3f} (元: {sharpe_before:.3f})")
print()

# ── チャート作成 ──────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle("◎採用5銘柄 定量分析レポート（2024-07〜2026-02）", fontsize=14, fontweight="bold")

# 1. 時間帯別 勝率
ax = axes[0][0]
colors = ["#22c55e" if e > 0 else "#ef4444" for e in hourly["expected_r"]]
ax.bar(hourly["hour_utc"], hourly["wr"] * 100, color=colors, alpha=0.8)
ax.axhline(all_df["win"].mean() * 100, color="blue", linestyle="--", linewidth=1, label=f"全体平均 {all_df['win'].mean()*100:.1f}%")
ax.set_xlabel("時間帯（UTC）"); ax.set_ylabel("勝率(%)"); ax.set_title("時間帯別 勝率（緑=期待値+）")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2. 時間帯別 平均PnL
ax = axes[0][1]
colors2 = ["#22c55e" if v > 0 else "#ef4444" for v in hourly["avg_pnl"]]
ax.bar(hourly["hour_utc"], hourly["avg_pnl"], color=colors2, alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("時間帯（UTC）"); ax.set_ylabel("平均PnL（円）"); ax.set_title("時間帯別 平均PnL")
ax.grid(True, alpha=0.3)

# 3. 曜日別 勝率
ax = axes[0][2]
colors3 = ["#22c55e" if e > 0 else "#ef4444" for e in weekly["expected_r"]]
ax.bar(weekly["day_name"], weekly["wr"] * 100, color=colors3, alpha=0.8)
ax.axhline(all_df["win"].mean() * 100, color="blue", linestyle="--", linewidth=1)
ax.set_xlabel("曜日"); ax.set_ylabel("勝率(%)"); ax.set_title("曜日別 勝率")
ax.grid(True, alpha=0.3)

# 4. 保有時間別 勝率
ax = axes[1][0]
ax.bar(hold_stats["hold_q"].astype(str), hold_stats["wr"] * 100, color="#6366f1", alpha=0.8)
ax.axhline(all_df["win"].mean() * 100, color="blue", linestyle="--", linewidth=1)
ax.set_xlabel("保有時間"); ax.set_ylabel("勝率(%)"); ax.set_title("保有時間別 勝率（分位数）")
ax.grid(True, alpha=0.3)

# 5. ギャップ別 勝率
ax = axes[1][1]
gap_valid = gap_stats.dropna(subset=["gap_q"])
ax.bar(gap_valid["gap_q"].astype(str), gap_valid["wr"] * 100, color="#f97316", alpha=0.8)
ax.axhline(all_df["win"].mean() * 100, color="blue", linestyle="--", linewidth=1)
ax.set_xlabel("前回エントリーからの間隔"); ax.set_ylabel("勝率(%)"); ax.set_title("エントリー間隔別 勝率")
ax.grid(True, alpha=0.3)

# 6. 銘柄別 シャープレシオ
ax = axes[1][2]
ax.barh(pair_stats["pair"], pair_stats["sharpe"], color="#8b5cf6", alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("シャープレシオ（年率換算）"); ax.set_title("銘柄別 シャープレシオ")
ax.grid(True, alpha=0.3, axis="x")

# 7. フィルター効果比較
ax = axes[2][0]
labels = ["フィルターなし", "時間帯除外", "曜日除外", "ギャップ≥60m", "組み合わせ"]
sharpes = [sharpe_before, sharpe_after, sharpe_day, sharpe_gap, sharpe_combined]
bar_colors = ["#94a3b8"] + ["#22c55e" if s > sharpe_before else "#ef4444" for s in sharpes[1:]]
bars = ax.bar(labels, sharpes, color=bar_colors, alpha=0.85)
for bar, v in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("シャープレシオ"); ax.set_title("フィルター別 シャープレシオ比較")
ax.tick_params(axis="x", labelsize=8); ax.grid(True, alpha=0.3, axis="y")

# 8. PnL分布
ax = axes[2][1]
ax.hist(all_df["pnl"].clip(-100000, 200000), bins=80, color="#6366f1", alpha=0.7, edgecolor="none")
ax.axvline(0, color="red", linewidth=1.5)
ax.axvline(all_df["pnl"].mean(), color="green", linewidth=1.5, linestyle="--",
           label=f"平均 {all_df['pnl'].mean():.0f}円")
ax.set_xlabel("PnL（円）"); ax.set_ylabel("頻度"); ax.set_title("PnL分布")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 9. 直前勝敗別 勝率
ax = axes[2][2]
streak_valid = streak_stats[streak_stats["prev_win"].notna()]
colors_s = ["#22c55e" if w > all_df["win"].mean() else "#ef4444" for w in streak_valid["wr"]]
ax.bar(streak_valid["prev_result"], streak_valid["wr"] * 100, color=colors_s, alpha=0.8)
ax.axhline(all_df["win"].mean() * 100, color="blue", linestyle="--", linewidth=1)
ax.set_xlabel("直前結果"); ax.set_ylabel("勝率(%)"); ax.set_title("直前勝敗後の勝率（ストリーク効果）")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート保存: results/feature_analysis.png")

# ── 特徴量サマリー ────────────────────────────────────────
print("\n" + "=" * 60)
print("特徴量候補サマリー（シャープレシオ改善効果）")
print("=" * 60)
candidates = [
    ("時間帯フィルター", sharpe_after, len(filtered), sharpe_after - sharpe_before),
    ("曜日フィルター",   sharpe_day,   len(filtered_day), sharpe_day - sharpe_before),
    ("ギャップ≥60m",    sharpe_gap,   len(filtered_gap), sharpe_gap - sharpe_before),
    ("組み合わせ",      sharpe_combined, len(combined), sharpe_combined - sharpe_before),
]
for name, sharpe, n, delta in sorted(candidates, key=lambda x: -x[3]):
    print(f"  {name:20s}: SR={sharpe:.3f} ({delta:+.3f})  件数={n}")

# 有効な特徴量を保存
summary = {
    "bad_hours": bad_hours,
    "bad_days": bad_days,
    "gap_min_threshold": 60,
    "sharpe_before": sharpe_before,
    "sharpe_combined": sharpe_combined,
    "hourly": hourly.to_dict("records"),
    "weekly": weekly.to_dict("records"),
}
import json
with open(os.path.join(OUT_DIR, "feature_summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n特徴量サマリー保存: results/feature_summary.json")
