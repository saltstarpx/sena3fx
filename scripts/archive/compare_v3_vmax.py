"""
compare_v3_vmax.py
==================
v3 vs v_max 比較チャート生成
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(BASE_DIR, "results")
INIT_CASH = 1_000_000

eq_v3   = np.load(os.path.join(OUT_DIR, "7sym_v3_equity.npy"))
eq_vmax = np.load(os.path.join(OUT_DIR, "7sym_vmax_equity.npy"))
df_v3   = pd.read_csv(os.path.join(OUT_DIR, "7sym_v3_trades.csv"))
df_vmax = pd.read_csv(os.path.join(OUT_DIR, "7sym_vmax_trades.csv"))

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
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["exit_time"], utc=True).dt.to_period("M")
    monthly = df2.groupby("month")["pnl_delta"].sum()
    n_pos   = (monthly > 0).sum()
    n_extra = int(df["is_extra"].sum()) if "is_extra" in df.columns else 0
    return {
        "eq_arr": eq_arr, "dd_arr": dd_arr, "final": final, "ret": ret,
        "max_dd": max_dd, "n_total": len(df), "n_tp": n_tp, "n_sl": n_sl, "n_be": n_be,
        "wr": wr, "pf": pf, "monthly": monthly, "n_pos": n_pos, "n_extra": n_extra,
    }

s3   = calc_stats(eq_v3,   df_v3)
smax = calc_stats(eq_vmax, df_vmax)

# ── 比較チャート ────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "7銘柄 3ヶ月バックテスト比較（v3 vs v_max）\n"
    "v3: リスク制御あり（8ポジ・10%上限・強制カット-0.5R）\n"
    "v_max: リスク制御なし（20ポジ・上限撤廃・強制カットなし）\n"
    "期間: 2025-03-03〜2025-06-02  初期資金: 100万円",
    fontsize=12, fontweight="bold"
)

gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.35)

# 資産曲線
ax1 = fig.add_subplot(gs[0, :2])
x3   = np.linspace(0, 100, len(s3["eq_arr"]))
xmax = np.linspace(0, 100, len(smax["eq_arr"]))
ax1.plot(x3,   s3["eq_arr"]/1e4,   color="#10b981", linewidth=1.8,
         label=f"v3    最終:{s3['final']/1e4:.0f}万円 ({s3['ret']:+.0f}%)")
ax1.plot(xmax, smax["eq_arr"]/1e4, color="#f59e0b", linewidth=2.2,
         label=f"v_max 最終:{smax['final']/1e4:.0f}万円 ({smax['ret']:+.0f}%)")
ax1.axhline(INIT_CASH/1e4, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_title("資産曲線比較（100万円スタート）")
ax1.set_ylabel("資産（万円）"); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

# 達成度メーター
ax_meter = fig.add_subplot(gs[0, 2])
labels = ["v3", "v_max"]
values = [s3["final"]/1e4, smax["final"]/1e4]
colors = ["#10b981", "#f59e0b"]
bars = ax_meter.barh(labels, values, color=colors, alpha=0.85, height=0.5)
ax_meter.axvline(INIT_CASH/1e4, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
for bar, val in zip(bars, values):
    ax_meter.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                  f"{val:.0f}万円", va="center", fontsize=11, fontweight="bold")
ax_meter.set_xlabel("最終資産（万円）")
ax_meter.set_title("達成度（100万円→）"); ax_meter.grid(alpha=0.3, axis="x")
ax_meter.set_xlim(0, max(values)*1.25)

# DD比較
ax2 = fig.add_subplot(gs[1, :2])
ax2.fill_between(x3,   s3["dd_arr"],   0, color="#10b981", alpha=0.35,
                 label=f"v3    最大DD:{s3['max_dd']:.1f}%")
ax2.fill_between(xmax, smax["dd_arr"], 0, color="#f59e0b", alpha=0.35,
                 label=f"v_max 最大DD:{smax['max_dd']:.1f}%")
ax2.set_title("ドローダウン比較")
ax2.set_ylabel("DD（%）"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

# 指標比較テーブル
ax_tbl = fig.add_subplot(gs[1, 2])
ax_tbl.axis("off")
table_data = [
    ["指標", "v3", "v_max"],
    ["採用トレード数", f"{s3['n_total']}件", f"{smax['n_total']}件"],
    ["追加エントリー", f"{s3['n_extra']}件", f"{smax['n_extra']}件"],
    ["強制カット", "-0.5R", "なし"],
    ["ポジション上限", "8", "20"],
    ["リスク上限", "10%", "なし"],
    ["勝率（BE除く）", f"{s3['wr']:.1f}%", f"{smax['wr']:.1f}%"],
    ["PF", f"{s3['pf']:.2f}", f"{smax['pf']:.2f}"],
    ["最大DD", f"{s3['max_dd']:.1f}%", f"{smax['max_dd']:.1f}%"],
    ["リターン", f"{s3['ret']:+.0f}%", f"{smax['ret']:+.0f}%"],
    ["最終資産", f"{s3['final']/1e4:.0f}万円", f"{smax['final']/1e4:.0f}万円"],
    ["月次プラス", f"{s3['n_pos']}/{len(s3['monthly'])}月",
                  f"{smax['n_pos']}/{len(smax['monthly'])}月"],
]
tbl = ax_tbl.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e3a5f"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#d1fae5")
    elif c == 2:
        cell.set_facecolor("#fef3c7")
ax_tbl.set_title("指標比較", fontsize=10, fontweight="bold")

# 月次損益比較
ax3 = fig.add_subplot(gs[2, :])
months_all = sorted(set(
    [str(m) for m in s3["monthly"].index] +
    [str(m) for m in smax["monthly"].index]
))
x = np.arange(len(months_all)); w = 0.35

v3_vals   = [s3["monthly"].get(m, 0)/1e4 for m in months_all]
vmax_vals = [smax["monthly"].get(m, 0)/1e4 for m in months_all]

b1 = ax3.bar(x - w/2, v3_vals, w,
             color=["#10b981" if v>=0 else "#6ee7b7" for v in v3_vals], alpha=0.85, label="v3")
b2 = ax3.bar(x + w/2, vmax_vals, w,
             color=["#f59e0b" if v>=0 else "#fcd34d" for v in vmax_vals], alpha=0.85, label="v_max")
for bar, val in zip(b1, v3_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, val+(0.5 if val>=0 else -3),
             f"{val:+.0f}", ha="center", fontsize=9, color="#065f46", fontweight="bold")
for bar, val in zip(b2, vmax_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, val+(0.5 if val>=0 else -3),
             f"{val:+.0f}", ha="center", fontsize=9, color="#92400e", fontweight="bold")
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(months_all, fontsize=11)
ax3.set_title("月次損益比較（万円）")
ax3.set_ylabel("損益（万円）"); ax3.legend(fontsize=10); ax3.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "7sym_v3_vmax_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
print(f"比較チャート保存: {out_path}")
