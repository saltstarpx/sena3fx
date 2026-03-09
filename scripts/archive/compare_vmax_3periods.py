"""
compare_vmax_3periods.py
========================
vMAX 3期間比較チャート
  IS期間    : 2024-12-01〜2025-02-28（3ヶ月）
  OOS期間   : 2025-03-03〜2025-06-02（3ヶ月）
  直近期間  : 2026-01-01〜2026-02-28（2ヶ月）
"""
import os
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "results")
INIT_CASH = 1_000_000

# ── データ読み込み ─────────────────────────────────────────
periods = {
    "IS期間\n(2024-12〜2025-02)": {
        "eq_file":    "7sym_vmax_is_equity.npy",
        "trade_file": "7sym_vmax_is_trades.csv",
        "color":      "#6366f1",
        "label":      "IS期間（12〜2月）",
    },
    "OOS期間\n(2025-03〜2025-06)": {
        "eq_file":    "7sym_vmax_equity.npy",
        "trade_file": "7sym_vmax_trades.csv",
        "color":      "#f59e0b",
        "label":      "OOS期間（3〜6月）",
    },
    "直近期間\n(2026-01〜2026-02)": {
        "eq_file":    "7sym_vmax_2026_equity.npy",
        "trade_file": "7sym_vmax_2026_trades.csv",
        "color":      "#10b981",
        "label":      "直近期間（2026年1〜2月）",
    },
}

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
    n_extra = int(df["is_extra"].sum()) if "is_extra" in df.columns else 0
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["exit_time"], utc=True).dt.to_period("M")
    monthly = df2.groupby("month")["pnl_delta"].sum()
    n_pos   = (monthly > 0).sum()
    return {
        "eq_arr": eq_arr, "dd_arr": dd_arr, "final": final, "ret": ret,
        "max_dd": max_dd, "n_total": len(df), "n_tp": n_tp, "n_sl": n_sl, "n_be": n_be,
        "wr": wr, "pf": pf, "monthly": monthly, "n_pos": n_pos, "n_extra": n_extra,
    }

stats = {}
for name, cfg in periods.items():
    eq  = np.load(os.path.join(OUT_DIR, cfg["eq_file"]))
    df  = pd.read_csv(os.path.join(OUT_DIR, cfg["trade_file"]))
    stats[name] = {**calc_stats(eq, df), **cfg}

# ── チャート作成 ───────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    "vMAX 3期間バックテスト比較\n"
    "IS期間（2024年12月〜2025年2月）/ OOS期間（2025年3〜6月）/ 直近期間（2026年1〜2月）\n"
    "初期資金: 100万円  ポジション上限: 20  リスク制御: なし  強制カット: なし",
    fontsize=12, fontweight="bold"
)

gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

# ── 資産曲線（正規化：%リターン） ──────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for name, s in stats.items():
    x = np.linspace(0, 100, len(s["eq_arr"]))
    ret_curve = (s["eq_arr"] - INIT_CASH) / INIT_CASH * 100
    ax1.plot(x, ret_curve, color=s["color"], linewidth=2.0,
             label=f"{s['label']}  最終:{s['final']/1e4:.0f}万円 ({s['ret']:+.0f}%)")
ax1.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_title("資産曲線比較（リターン%、100万円スタート）")
ax1.set_ylabel("リターン（%）"); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

# ── 達成度バー ──────────────────────────────────────────────
ax_meter = fig.add_subplot(gs[0, 2])
labels = [s["label"] for s in stats.values()]
values = [s["final"]/1e4 for s in stats.values()]
colors = [s["color"] for s in stats.values()]
bars = ax_meter.barh(labels, values, color=colors, alpha=0.85, height=0.5)
ax_meter.axvline(INIT_CASH/1e4, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
for bar, val in zip(bars, values):
    ax_meter.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                  f"{val:.0f}万円", va="center", fontsize=10, fontweight="bold")
ax_meter.set_xlabel("最終資産（万円）")
ax_meter.set_title("達成度（100万円→）"); ax_meter.grid(alpha=0.3, axis="x")
ax_meter.set_xlim(0, max(values)*1.3)

# ── DD比較 ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
for name, s in stats.items():
    x = np.linspace(0, 100, len(s["dd_arr"]))
    ax2.fill_between(x, s["dd_arr"], 0, color=s["color"], alpha=0.3,
                     label=f"{s['label']}  最大DD:{s['max_dd']:.1f}%")
ax2.set_title("ドローダウン比較")
ax2.set_ylabel("DD（%）"); ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

# ── 指標比較テーブル ────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[1, 2])
ax_tbl.axis("off")
s_list = list(stats.values())
table_data = [
    ["指標", "IS期間", "OOS期間", "直近期間"],
    ["期間", "24-12〜25-02", "25-03〜25-06", "26-01〜26-02"],
    ["採用トレード数", f"{s_list[0]['n_total']}件", f"{s_list[1]['n_total']}件", f"{s_list[2]['n_total']}件"],
    ["追加エントリー", f"{s_list[0]['n_extra']}件", f"{s_list[1]['n_extra']}件", f"{s_list[2]['n_extra']}件"],
    ["勝率（BE除く）", f"{s_list[0]['wr']:.1f}%", f"{s_list[1]['wr']:.1f}%", f"{s_list[2]['wr']:.1f}%"],
    ["PF", f"{s_list[0]['pf']:.2f}", f"{s_list[1]['pf']:.2f}", f"{s_list[2]['pf']:.2f}"],
    ["最大DD", f"{s_list[0]['max_dd']:.1f}%", f"{s_list[1]['max_dd']:.1f}%", f"{s_list[2]['max_dd']:.1f}%"],
    ["リターン", f"{s_list[0]['ret']:+.0f}%", f"{s_list[1]['ret']:+.0f}%", f"{s_list[2]['ret']:+.0f}%"],
    ["最終資産", f"{s_list[0]['final']/1e4:.0f}万円", f"{s_list[1]['final']/1e4:.0f}万円", f"{s_list[2]['final']/1e4:.0f}万円"],
    ["月次プラス", f"{s_list[0]['n_pos']}/{len(s_list[0]['monthly'])}月",
                  f"{s_list[1]['n_pos']}/{len(s_list[1]['monthly'])}月",
                  f"{s_list[2]['n_pos']}/{len(s_list[2]['monthly'])}月"],
]
tbl = ax_tbl.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
col_colors = ["#1e3a5f", "#ede9fe", "#fef3c7", "#d1fae5"]
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1e3a5f"); cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#ede9fe")
    elif c == 2:
        cell.set_facecolor("#fef3c7")
    elif c == 3:
        cell.set_facecolor("#d1fae5")
ax_tbl.set_title("3期間指標比較", fontsize=10, fontweight="bold")

# ── 月次損益（全期間） ─────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])
all_months = []
for s in stats.values():
    all_months.extend([str(m) for m in s["monthly"].index])
months_all = sorted(set(all_months))

x = np.arange(len(months_all)); w = 0.28
for i, (name, s) in enumerate(stats.items()):
    vals = [s["monthly"].get(m, 0)/1e4 for m in months_all]
    offset = (i - 1) * w
    bars = ax3.bar(x + offset, vals, w, color=s["color"], alpha=0.85, label=s["label"])
    for bar, val in zip(bars, vals):
        if abs(val) > 0.1:
            ax3.text(bar.get_x()+bar.get_width()/2, val+(0.5 if val>=0 else -3),
                     f"{val:+.0f}", ha="center", fontsize=8, color="black", fontweight="bold")

ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(months_all, fontsize=10)
ax3.set_title("月次損益比較（万円）")
ax3.set_ylabel("損益（万円）"); ax3.legend(fontsize=10); ax3.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "vmax_3periods_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
print(f"3期間比較チャート保存: {out_path}")
