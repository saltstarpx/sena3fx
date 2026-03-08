"""
compare_v1_v2.py
================
v1 vs v2 比較チャート生成
"""
import os, sys
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

INIT_CASH = 1_000_000

# データ読み込み
eq_v1 = np.load(os.path.join(OUT_DIR, "7sym_v1_equity.npy"))
eq_v2 = np.load(os.path.join(OUT_DIR, "7sym_v2_equity.npy"))
df_v1 = pd.read_csv(os.path.join(OUT_DIR, "7sym_v1_trades.csv"))
df_v2 = pd.read_csv(os.path.join(OUT_DIR, "7sym_v2_trades.csv"))

def calc_stats(eq_arr, df):
    peak   = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - peak) / peak * 100
    max_dd = dd_arr.min()
    final  = eq_arr[-1]
    ret    = (final - INIT_CASH) / INIT_CASH * 100
    n_tp   = (df["result"] == "TP").sum()
    n_sl   = (df["result"] == "SL").sum()
    n_be   = (df["result"] == "BE").sum()
    wr     = n_tp / (n_tp + n_sl) * 100 if (n_tp + n_sl) > 0 else 0
    tp_pnl = df[df["result"]=="TP"]["pnl_delta"].sum()
    sl_pnl = abs(df[df["result"]=="SL"]["pnl_delta"].sum())
    pf     = tp_pnl / sl_pnl if sl_pnl > 0 else float("inf")
    df["month"] = pd.to_datetime(df["exit_time"], utc=True).dt.to_period("M")
    monthly = df.groupby("month")["pnl_delta"].sum()
    n_pos   = (monthly > 0).sum()
    return {
        "eq_arr": eq_arr, "dd_arr": dd_arr, "final": final, "ret": ret,
        "max_dd": max_dd, "n_total": len(df), "n_tp": n_tp, "n_sl": n_sl, "n_be": n_be,
        "wr": wr, "pf": pf, "monthly": monthly, "n_pos": n_pos,
    }

s1 = calc_stats(eq_v1, df_v1)
s2 = calc_stats(eq_v2, df_v2)

# ── 比較チャート ──────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "7銘柄 3ヶ月バックテスト比較\nv1: yagami_position8_risk10  vs  v2: yagami_position8_risk10_v2\n"
    "期間: 2025-03-03〜2025-06-02  初期資金: 100万円",
    fontsize=13, fontweight="bold"
)

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# ── 資産曲線（並列） ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
x1 = np.linspace(0, 100, len(s1["eq_arr"]))
x2 = np.linspace(0, 100, len(s2["eq_arr"]))
ax1.plot(x1, s1["eq_arr"]/1e4, color="#3b82f6", linewidth=1.8, label=f"v1  最終:{s1['final']/1e4:.0f}万円 ({s1['ret']:+.0f}%)")
ax1.plot(x2, s2["eq_arr"]/1e4, color="#8b5cf6", linewidth=1.8, label=f"v2  最終:{s2['final']/1e4:.0f}万円 ({s2['ret']:+.0f}%)")
ax1.axhline(INIT_CASH/1e4, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_title("資産曲線比較（100万円スタート）"); ax1.set_ylabel("資産（万円）"); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

# ── 達成度メーター ───────────────────────────────────────
ax_meter = fig.add_subplot(gs[0, 2])
labels = ["v1", "v2"]
values = [s1["final"]/1e4, s2["final"]/1e4]
colors = ["#3b82f6", "#8b5cf6"]
bars = ax_meter.barh(labels, values, color=colors, alpha=0.85, height=0.5)
ax_meter.axvline(INIT_CASH/1e4, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
for bar, val in zip(bars, values):
    ax_meter.text(val + 5, bar.get_y() + bar.get_height()/2,
                  f"{val:.0f}万円", va="center", fontsize=11, fontweight="bold")
ax_meter.set_xlabel("最終資産（万円）"); ax_meter.set_title("達成度（100万円→）"); ax_meter.grid(alpha=0.3, axis="x")
ax_meter.set_xlim(0, max(values)*1.2)

# ── DD比較 ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.fill_between(x1, s1["dd_arr"], 0, color="#3b82f6", alpha=0.35, label=f"v1  最大DD:{s1['max_dd']:.1f}%")
ax2.fill_between(x2, s2["dd_arr"], 0, color="#8b5cf6", alpha=0.35, label=f"v2  最大DD:{s2['max_dd']:.1f}%")
ax2.set_title("ドローダウン比較"); ax2.set_ylabel("DD（%）"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

# ── 指標比較テーブル ─────────────────────────────────────
ax_tbl = fig.add_subplot(gs[1, 2])
ax_tbl.axis("off")
table_data = [
    ["指標", "v1", "v2"],
    ["採用トレード数", f"{s1['n_total']}件", f"{s2['n_total']}件"],
    ["TP / SL / BE", f"{s1['n_tp']}/{s1['n_sl']}/{s1['n_be']}", f"{s2['n_tp']}/{s2['n_sl']}/{s2['n_be']}"],
    ["勝率（BE除く）", f"{s1['wr']:.1f}%", f"{s2['wr']:.1f}%"],
    ["PF", f"{s1['pf']:.2f}", f"{s2['pf']:.2f}"],
    ["最大DD", f"{s1['max_dd']:.1f}%", f"{s2['max_dd']:.1f}%"],
    ["リターン", f"{s1['ret']:+.0f}%", f"{s2['ret']:+.0f}%"],
    ["最終資産", f"{s1['final']/1e4:.0f}万円", f"{s2['final']/1e4:.0f}万円"],
    ["月次プラス", f"{s1['n_pos']}/{len(s1['monthly'])}月", f"{s2['n_pos']}/{len(s2['monthly'])}月"],
]
tbl = ax_tbl.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e3a5f"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#dbeafe")
    elif c == 2:
        cell.set_facecolor("#ede9fe")
ax_tbl.set_title("指標比較", fontsize=10, fontweight="bold")

# ── 月次損益比較 ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])
months_v1 = [str(m) for m in s1["monthly"].index]
months_v2 = [str(m) for m in s2["monthly"].index]
all_months = sorted(set(months_v1 + months_v2))
x = np.arange(len(all_months)); w = 0.35

v1_vals = [s1["monthly"].get(m, 0)/1e4 for m in all_months]
v2_vals = [s2["monthly"].get(m, 0)/1e4 for m in all_months]

b1 = ax3.bar(x - w/2, v1_vals, w, color=["#3b82f6" if v>=0 else "#93c5fd" for v in v1_vals], alpha=0.85, label="v1")
b2 = ax3.bar(x + w/2, v2_vals, w, color=["#8b5cf6" if v>=0 else "#c4b5fd" for v in v2_vals], alpha=0.85, label="v2")
for bar, val in zip(b1, v1_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, val+(0.2 if val>=0 else -1.5),
             f"{val:+.0f}", ha="center", fontsize=8, color="#1d4ed8")
for bar, val in zip(b2, v2_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, val+(0.2 if val>=0 else -1.5),
             f"{val:+.0f}", ha="center", fontsize=8, color="#6d28d9")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(all_months, fontsize=10)
ax3.set_title("月次損益比較（万円）"); ax3.set_ylabel("損益（万円）"); ax3.legend(fontsize=10); ax3.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "7sym_v1_v2_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
print(f"比較チャート保存: {out_path}")
