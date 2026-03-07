"""
detailed_analysis_v76.py
========================
v76全期間バックテスト結果の詳細統計分析
- 時間帯別・曜日別・パターン別・連続勝敗・RR分析等
"""
import sys, os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

# 日本語フォント
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

RESULTS = "/home/ubuntu/sena3fx/results"

# ============================================================
# データ読み込み
# ============================================================
df = pd.read_csv(f"{RESULTS}/v76_full_period_trades.csv",
                 parse_dates=["entry_time", "exit_time"])
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)

# 時間帯（UTC→JST: +9h）
df["entry_jst"]  = df["entry_time"] + pd.Timedelta(hours=9)
df["entry_hour"] = df["entry_jst"].dt.hour
df["entry_dow"]  = df["entry_jst"].dt.day_name()
df["hold_hours"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
df["rr_actual"]  = df["pnl"] / df["pnl"].abs().where(df["pnl"] < 0, other=np.nan)

# 実際のRR計算（勝ちトレードのみ）
wins   = df[df["pnl"] > 0]
losses = df[df["pnl"] < 0]

print(f"総トレード数: {len(df)}")
print(f"勝ち: {len(wins)}  負け: {len(losses)}")

# ============================================================
# 1. 時間帯別分析（JST）
# ============================================================
hourly = df.groupby("entry_hour").agg(
    trades=("pnl", "count"),
    wins=("result", lambda x: (x=="win").sum()),
    total_pnl=("pnl", "sum"),
    avg_pnl=("pnl", "mean"),
).reset_index()
hourly["win_rate"] = (hourly["wins"] / hourly["trades"] * 100).round(1)
hourly["pf"] = df[df["pnl"]>0].groupby(df[df["pnl"]>0]["entry_hour"])["pnl"].sum() / \
               df[df["pnl"]<0].groupby(df[df["pnl"]<0]["entry_hour"])["pnl"].apply(lambda x: abs(x.sum()))
hourly["pf"] = hourly["pf"].fillna(0).round(2)

print("\n=== 時間帯別（JST）===")
print(hourly.to_string(index=False))

# ============================================================
# 2. 曜日別分析
# ============================================================
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_jp    = {"Monday":"月","Tuesday":"火","Wednesday":"水","Thursday":"木","Friday":"金","Saturday":"土","Sunday":"日"}
weekly = df.groupby("entry_dow").agg(
    trades=("pnl","count"),
    wins=("result", lambda x: (x=="win").sum()),
    total_pnl=("pnl","sum"),
    avg_pnl=("pnl","mean"),
).reindex(dow_order).dropna()
weekly["win_rate"] = (weekly["wins"] / weekly["trades"] * 100).round(1)
weekly.index = [dow_jp.get(d, d) for d in weekly.index]

print("\n=== 曜日別 ===")
print(weekly.to_string())

# ============================================================
# 3. パターン別分析
# ============================================================
pattern = df.groupby("pattern").agg(
    trades=("pnl","count"),
    wins=("result", lambda x: (x=="win").sum()),
    total_pnl=("pnl","sum"),
    avg_win=("pnl", lambda x: x[x>0].mean()),
    avg_loss=("pnl", lambda x: x[x<0].mean()),
).reset_index()
pattern["win_rate"] = (pattern["wins"] / pattern["trades"] * 100).round(1)

print("\n=== パターン別 ===")
print(pattern.to_string(index=False))

# ============================================================
# 4. 方向別（ロング/ショート）分析
# ============================================================
dir_map = {1: "ロング", -1: "ショート"}
df["dir_label"] = df["dir"].map(dir_map)
direction = df.groupby("dir_label").agg(
    trades=("pnl","count"),
    wins=("result", lambda x: (x=="win").sum()),
    total_pnl=("pnl","sum"),
    avg_pnl=("pnl","mean"),
).reset_index()
direction["win_rate"] = (direction["wins"] / direction["trades"] * 100).round(1)

print("\n=== 方向別 ===")
print(direction.to_string(index=False))

# ============================================================
# 5. 連続勝敗分析
# ============================================================
results_seq = df.sort_values("entry_time")["result"].tolist()
max_consec_win = max_consec_loss = cur_w = cur_l = 0
streaks_w, streaks_l = [], []
for r in results_seq:
    if r == "win":
        cur_w += 1
        if cur_l > 0: streaks_l.append(cur_l)
        cur_l = 0
        max_consec_win = max(max_consec_win, cur_w)
    else:
        cur_l += 1
        if cur_w > 0: streaks_w.append(cur_w)
        cur_w = 0
        max_consec_loss = max(max_consec_loss, cur_l)
if cur_w > 0: streaks_w.append(cur_w)
if cur_l > 0: streaks_l.append(cur_l)

avg_consec_win  = np.mean(streaks_w) if streaks_w else 0
avg_consec_loss = np.mean(streaks_l) if streaks_l else 0

print(f"\n=== 連続勝敗 ===")
print(f"最大連続勝ち: {max_consec_win}回  平均連続勝ち: {avg_consec_win:.1f}回")
print(f"最大連続負け: {max_consec_loss}回  平均連続負け: {avg_consec_loss:.1f}回")

# ============================================================
# 6. 保有時間分析
# ============================================================
print(f"\n=== 保有時間（時間）===")
print(f"平均: {df['hold_hours'].mean():.1f}h  中央値: {df['hold_hours'].median():.1f}h")
print(f"最短: {df['hold_hours'].min():.1f}h  最長: {df['hold_hours'].max():.1f}h")
print(f"勝ち平均: {wins['hold_hours'].mean():.1f}h  負け平均: {losses['hold_hours'].mean():.1f}h")

# ============================================================
# 7. 月次詳細
# ============================================================
monthly = df.groupby(["month","period"]).agg(
    trades=("pnl","count"),
    wins=("result", lambda x: (x=="win").sum()),
    total_pnl=("pnl","sum"),
    avg_pnl=("pnl","mean"),
    max_win=("pnl","max"),
    max_loss=("pnl","min"),
).reset_index()
monthly["win_rate"] = (monthly["wins"] / monthly["trades"] * 100).round(1)

print("\n=== 月次詳細 ===")
print(monthly[["month","period","trades","win_rate","total_pnl","avg_pnl"]].to_string(index=False))

# ============================================================
# 8. 半利確の効果分析
# ============================================================
half_sl = df[df["exit_type"] == "HALF+SL"]
half_tp = df[df["exit_type"] == "HALF+TP"]
full_sl = df[df["exit_type"] == "SL"]
full_tp = df[df["exit_type"] == "TP"]

print(f"\n=== 決済タイプ別 ===")
for label, sub in [("HALF+SL", half_sl), ("HALF+TP", half_tp), ("SL", full_sl), ("TP", full_tp)]:
    if len(sub) > 0:
        print(f"{label}: {len(sub)}回  平均損益: {sub['pnl'].mean():.1f}pips  合計: {sub['pnl'].sum():.1f}pips")

# ============================================================
# 9. 結果をJSONで保存（レポート用）
# ============================================================
report_data = {
    "summary": {
        "total_trades": len(df),
        "win_rate": round(len(wins)/len(df)*100, 1),
        "pf": round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2),
        "total_pnl": round(df["pnl"].sum(), 1),
        "avg_win": round(wins["pnl"].mean(), 2),
        "avg_loss": round(losses["pnl"].mean(), 2),
        "max_consec_win": max_consec_win,
        "max_consec_loss": max_consec_loss,
        "avg_hold_hours": round(df["hold_hours"].mean(), 1),
        "avg_hold_win": round(wins["hold_hours"].mean(), 1),
        "avg_hold_loss": round(losses["hold_hours"].mean(), 1),
    },
    "hourly": hourly.to_dict(orient="records"),
    "weekly": weekly.reset_index().rename(columns={"index":"dow"}).to_dict(orient="records"),
    "pattern": pattern.to_dict(orient="records"),
    "direction": direction.to_dict(orient="records"),
    "exit_type": {
        "HALF+SL": {"count": len(half_sl), "avg_pnl": round(half_sl["pnl"].mean(), 1) if len(half_sl)>0 else 0, "total": round(half_sl["pnl"].sum(), 1)},
        "HALF+TP": {"count": len(half_tp), "avg_pnl": round(half_tp["pnl"].mean(), 1) if len(half_tp)>0 else 0, "total": round(half_tp["pnl"].sum(), 1)},
        "SL":      {"count": len(full_sl), "avg_pnl": round(full_sl["pnl"].mean(), 1) if len(full_sl)>0 else 0, "total": round(full_sl["pnl"].sum(), 1)},
        "TP":      {"count": len(full_tp), "avg_pnl": round(full_tp["pnl"].mean(), 1) if len(full_tp)>0 else 0, "total": round(full_tp["pnl"].sum(), 1)},
    },
    "monthly": monthly.to_dict(orient="records"),
}

with open(f"{RESULTS}/v76_detailed_analysis.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

# ============================================================
# 10. 詳細可視化チャート
# ============================================================
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#0d0d1a")

def styled_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    if xlabel: ax.set_xlabel(xlabel, color="white", fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color="white", fontsize=8)

# ── 1. 時間帯別トレード数・勝率 ──
ax1 = fig.add_subplot(4, 3, 1)
hours = hourly["entry_hour"]
colors_h = ["#e74c3c" if (h < 7 or h >= 22) else  # 深夜〜早朝
            "#f39c12" if (7 <= h < 9) else           # 東京開場
            "#2ecc71" if (9 <= h < 17) else           # 東京時間
            "#3498db" if (17 <= h < 22) else           # ロンドン〜NY
            "#95a5a6" for h in hours]
ax1.bar(hours, hourly["trades"], color=colors_h, alpha=0.85, edgecolor="white", linewidth=0.3)
ax1_twin = ax1.twinx()
ax1_twin.plot(hours, hourly["win_rate"], color="#f1c40f", marker="o", markersize=4, linewidth=1.5)
ax1_twin.set_ylabel("勝率(%)", color="#f1c40f", fontsize=8)
ax1_twin.tick_params(colors="#f1c40f", labelsize=7)
ax1_twin.set_ylim(0, 100)
styled_ax(ax1, "時間帯別トレード数・勝率(JST)", xlabel="時刻(JST)", ylabel="トレード数")

# ── 2. 時間帯別損益 ──
ax2 = fig.add_subplot(4, 3, 2)
bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in hourly["total_pnl"]]
ax2.bar(hours, hourly["total_pnl"], color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.3)
ax2.axhline(0, color="white", linewidth=0.5)
styled_ax(ax2, "時間帯別累積損益(JST)", xlabel="時刻(JST)", ylabel="損益(pips)")

# ── 3. 曜日別分析 ──
ax3 = fig.add_subplot(4, 3, 3)
dow_labels = list(weekly.index)
dow_pnl    = weekly["total_pnl"].values
dow_wr     = weekly["win_rate"].values
bar_colors3 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in dow_pnl]
ax3.bar(dow_labels, dow_pnl, color=bar_colors3, alpha=0.85, edgecolor="white", linewidth=0.3)
ax3_twin = ax3.twinx()
ax3_twin.plot(dow_labels, dow_wr, color="#f1c40f", marker="o", markersize=5, linewidth=1.5)
ax3_twin.set_ylabel("勝率(%)", color="#f1c40f", fontsize=8)
ax3_twin.tick_params(colors="#f1c40f", labelsize=7)
ax3_twin.set_ylim(0, 100)
ax3.axhline(0, color="white", linewidth=0.5)
styled_ax(ax3, "曜日別損益・勝率", ylabel="損益(pips)")

# ── 4. 決済タイプ別 ──
ax4 = fig.add_subplot(4, 3, 4)
exit_labels = ["HALF+SL", "HALF+TP", "SL", "TP"]
exit_counts = [len(half_sl), len(half_tp), len(full_sl), len(full_tp)]
exit_colors = ["#f39c12", "#2ecc71", "#e74c3c", "#3498db"]
wedges, texts, autotexts = ax4.pie(
    exit_counts, labels=exit_labels, colors=exit_colors,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": "white", "fontsize": 8},
    pctdistance=0.75,
)
for at in autotexts: at.set_color("white")
styled_ax(ax4, "決済タイプ別構成比")

# ── 5. 決済タイプ別平均損益 ──
ax5 = fig.add_subplot(4, 3, 5)
exit_avg = [half_sl["pnl"].mean() if len(half_sl)>0 else 0,
            half_tp["pnl"].mean() if len(half_tp)>0 else 0,
            full_sl["pnl"].mean() if len(full_sl)>0 else 0,
            full_tp["pnl"].mean() if len(full_tp)>0 else 0]
bar_colors5 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in exit_avg]
ax5.bar(exit_labels, exit_avg, color=bar_colors5, alpha=0.85, edgecolor="white", linewidth=0.3)
ax5.axhline(0, color="white", linewidth=0.5)
for i, v in enumerate(exit_avg):
    ax5.text(i, v + (3 if v >= 0 else -8), f"{v:.1f}", ha="center", color="white", fontsize=8)
styled_ax(ax5, "決済タイプ別平均損益(pips)", ylabel="平均損益(pips)")

# ── 6. 保有時間分布 ──
ax6 = fig.add_subplot(4, 3, 6)
ax6.hist(df["hold_hours"].clip(0, 72), bins=40, color="#9b59b6", alpha=0.85,
         edgecolor="#0d0d1a", linewidth=0.3)
ax6.axvline(df["hold_hours"].mean(), color="#f1c40f", linewidth=1.5,
            label=f"平均 {df['hold_hours'].mean():.1f}h")
ax6.axvline(df["hold_hours"].median(), color="#2ecc71", linewidth=1.5, linestyle="--",
            label=f"中央値 {df['hold_hours'].median():.1f}h")
ax6.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7)
styled_ax(ax6, "保有時間分布", xlabel="保有時間(h)", ylabel="頻度")

# ── 7. 月次勝率推移 ──
ax7 = fig.add_subplot(4, 3, (7, 8))
monthly_wr = df.groupby("month").apply(lambda x: (x["result"]=="win").sum()/len(x)*100).reset_index()
monthly_wr.columns = ["month", "win_rate"]
colors_m = ["#3498db" if m < "2025-03" else "#e74c3c" for m in monthly_wr["month"]]
ax7.bar(range(len(monthly_wr)), monthly_wr["win_rate"], color=colors_m, alpha=0.85,
        edgecolor="white", linewidth=0.3)
ax7.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5, label="50%")
ax7.axhline(df["result"].eq("win").mean()*100, color="#f1c40f", linewidth=1.2,
            linestyle="-", label=f"全期間平均 {df['result'].eq('win').mean()*100:.1f}%")
ax7.set_xticks(range(len(monthly_wr)))
ax7.set_xticklabels(monthly_wr["month"], rotation=45, fontsize=7)
ax7.set_ylim(0, 100)
ax7.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax7, "月次勝率推移（青=IS / 赤=OOS）", ylabel="勝率(%)")

# ── 8. 連続勝敗分布 ──
ax8 = fig.add_subplot(4, 3, 9)
max_s = max(max(streaks_w) if streaks_w else 1, max(streaks_l) if streaks_l else 1)
bins_s = range(1, max_s + 2)
ax8.hist(streaks_w, bins=bins_s, alpha=0.7, color="#2ecc71", label=f"連続勝ち (最大{max_consec_win}回)", edgecolor="white", linewidth=0.3)
ax8.hist(streaks_l, bins=bins_s, alpha=0.7, color="#e74c3c", label=f"連続負け (最大{max_consec_loss}回)", edgecolor="white", linewidth=0.3)
ax8.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax8, "連続勝敗分布", xlabel="連続回数", ylabel="頻度")

# ── 9. IS/OOS月次PnL比較 ──
ax9 = fig.add_subplot(4, 3, (10, 11))
is_monthly  = df[df["period"]=="IS"].groupby("month")["pnl"].sum()
oos_monthly = df[df["period"]=="OOS"].groupby("month")["pnl"].sum()
all_months  = sorted(set(list(is_monthly.index) + list(oos_monthly.index)))
x_pos = range(len(all_months))
is_vals_m  = [is_monthly.get(m, 0) for m in all_months]
oos_vals_m = [oos_monthly.get(m, 0) for m in all_months]
w = 0.4
ax9.bar([x - w/2 for x in x_pos], is_vals_m,  w, color="#3498db", alpha=0.85, label="IS期間",  edgecolor="white", linewidth=0.3)
ax9.bar([x + w/2 for x in x_pos], oos_vals_m, w, color="#e74c3c", alpha=0.85, label="OOS期間", edgecolor="white", linewidth=0.3)
ax9.axhline(0, color="white", linewidth=0.5)
ax9.set_xticks(list(x_pos))
ax9.set_xticklabels(all_months, rotation=45, fontsize=7)
ax9.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
styled_ax(ax9, "月次損益 IS vs OOS", ylabel="損益(pips)")

# ── 10. パターン別損益 ──
ax10 = fig.add_subplot(4, 3, 12)
pat_labels = pattern["pattern"].tolist()
pat_pnl    = pattern["total_pnl"].tolist()
pat_wr     = pattern["win_rate"].tolist()
bar_colors10 = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pat_pnl]
ax10.bar(pat_labels, pat_pnl, color=bar_colors10, alpha=0.85, edgecolor="white", linewidth=0.3)
ax10_twin = ax10.twinx()
ax10_twin.plot(pat_labels, pat_wr, color="#f1c40f", marker="o", markersize=6, linewidth=1.5)
ax10_twin.set_ylabel("勝率(%)", color="#f1c40f", fontsize=8)
ax10_twin.tick_params(colors="#f1c40f", labelsize=7)
ax10_twin.set_ylim(0, 100)
styled_ax(ax10, "パターン別損益・勝率", ylabel="損益(pips)")

plt.tight_layout(pad=2.5)
out_img = f"{RESULTS}/v76_detailed_analysis.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out_img}")
print("完了")
