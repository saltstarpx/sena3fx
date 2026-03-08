"""
全12銘柄 v77（1Hベース）NYプロコーススプレッド バックテスト結果の可視化
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 日本語フォント設定
for font in fm.findSystemFonts():
    if "NotoSansCJK" in font or "NotoSansJP" in font or "ipag" in font.lower() or "ipaex" in font.lower():
        fm.fontManager.addfont(font)
try:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
except:
    try:
        plt.rcParams["font.family"] = "IPAexGothic"
    except:
        pass

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 全12銘柄の結果データ（ログから手動集計）
results = {
    # 前半4銘柄（bt_ny2_log.txtより）
    "USDJPY":  {"spread": 0.8,  "IS": {"n":476,  "wr":22.1, "pf":0.99, "ret":21.5,   "mdd":38.7, "kelly":-0.566, "mp":"3/8"},
                                "OOS":{"n":769,  "wr":20.3, "pf":0.99, "ret":58.0,   "mdd":48.6, "kelly":-0.604, "mp":"7/12"}},
    "EURUSD":  {"spread": 0.8,  "IS": {"n":546,  "wr":24.5, "pf":1.39, "ret":254.0,  "mdd":12.2, "kelly":-0.298, "mp":"6/8"},
                                "OOS":{"n":926,  "wr":23.4, "pf":1.18, "ret":242.5,  "mdd":30.4, "kelly":-0.416, "mp":"11/12"}},
    "GBPUSD":  {"spread": 1.3,  "IS": {"n":610,  "wr":26.2, "pf":1.35, "ret":270.0,  "mdd":16.9, "kelly":-0.285, "mp":"6/8"},
                                "OOS":{"n":918,  "wr":23.1, "pf":1.25, "ret":221.0,  "mdd":29.2, "kelly":-0.383, "mp":"10/12"}},
    "AUDUSD":  {"spread": 1.4,  "IS": {"n":555,  "wr":25.8, "pf":1.61, "ret":328.5,  "mdd":20.3, "kelly":-0.203, "mp":"8/8"},
                                "OOS":{"n":871,  "wr":21.9, "pf":1.24, "ret":202.5,  "mdd":46.9, "kelly":-0.411, "mp":"8/12"}},
    # 後半8銘柄（bt_remaining2より）
    "USDCAD":  {"spread": 2.2,  "IS": {"n":1336, "wr":72.8, "pf":2.92, "ret":1394.4, "mdd":32.7, "kelly":0.634,  "mp":"8/8"},
                                "OOS":{"n":1902, "wr":76.1, "pf":3.74, "ret":2487.3, "mdd":9.1,  "kelly":0.697,  "mp":"12/12"}},
    "USDCHF":  {"spread": 1.8,  "IS": {"n":1259, "wr":74.8, "pf":3.24, "ret":1417.4, "mdd":31.9, "kelly":0.670,  "mp":"8/8"},
                                "OOS":{"n":1825, "wr":70.3, "pf":2.52, "ret":1645.0, "mdd":18.6, "kelly":0.585,  "mp":"11/12"}},
    "NZDUSD":  {"spread": 1.7,  "IS": {"n":1213, "wr":74.9, "pf":4.05, "ret":1862.5, "mdd":19.8, "kelly":0.687,  "mp":"8/8"},
                                "OOS":{"n":1725, "wr":73.5, "pf":3.18, "ret":1989.0, "mdd":22.0, "kelly":0.652,  "mp":"12/12"}},
    "EURJPY":  {"spread": 2.0,  "IS": {"n":1112, "wr":77.2, "pf":4.61, "ret":1828.6, "mdd":14.3, "kelly":0.723,  "mp":"8/8"},
                                "OOS":{"n":1834, "wr":76.3, "pf":3.70, "ret":2347.0, "mdd":10.0, "kelly":0.699,  "mp":"12/12"}},
    "GBPJPY":  {"spread": 3.0,  "IS": {"n":1080, "wr":77.0, "pf":4.48, "ret":1726.6, "mdd":6.2,  "kelly":0.719,  "mp":"8/8"},
                                "OOS":{"n":1829, "wr":74.5, "pf":3.17, "ret":2023.0, "mdd":10.0, "kelly":0.664,  "mp":"12/12"}},
    "EURGBP":  {"spread": 1.7,  "IS": {"n":1312, "wr":64.1, "pf":1.85, "ret":801.1,  "mdd":34.9, "kelly":0.447,  "mp":"7/8"},
                                "OOS":{"n":1853, "wr":67.7, "pf":2.07, "ret":1279.1, "mdd":16.7, "kelly":0.520,  "mp":"11/12"}},
    "US30":    {"spread": 3.0,  "IS": {"n":1062, "wr":80.7, "pf":6.10, "ret":2089.7, "mdd":8.9,  "kelly":0.775,  "mp":"8/8"},
                                "OOS":{"n":1416, "wr":80.2, "pf":5.13, "ret":2310.1, "mdd":3.5,  "kelly":0.764,  "mp":"11/11"}},
    "SPX500":  {"spread": 0.5,  "IS": {"n":1059, "wr":84.3, "pf":8.67, "ret":2546.6, "mdd":10.0, "kelly":0.825,  "mp":"8/8"},
                                "OOS":{"n":1467, "wr":81.9, "pf":5.76, "ret":2529.8, "mdd":2.7,  "kelly":0.787,  "mp":"12/12"}},
}

pairs = list(results.keys())
n_pairs = len(pairs)

# ── 図1: PF比較バーチャート ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("v77（1Hベース）全12銘柄 NYプロコーススプレッド バックテスト結果", fontsize=14, fontweight="bold", y=0.98)

colors_is  = "#4C72B0"
colors_oos = "#DD8452"

# PF比較
ax = axes[0][0]
x = np.arange(n_pairs)
w = 0.35
pf_is  = [results[p]["IS"]["pf"]  for p in pairs]
pf_oos = [results[p]["OOS"]["pf"] for p in pairs]
ax.bar(x - w/2, pf_is,  w, label="IS",  color=colors_is,  alpha=0.85)
ax.bar(x + w/2, pf_oos, w, label="OOS", color=colors_oos, alpha=0.85)
ax.axhline(1.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="PF=1.5基準")
ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("プロフィットファクター"); ax.set_title("PF比較（IS vs OOS）")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# MDD比較
ax = axes[0][1]
mdd_is  = [results[p]["IS"]["mdd"]  for p in pairs]
mdd_oos = [results[p]["OOS"]["mdd"] for p in pairs]
ax.bar(x - w/2, mdd_is,  w, label="IS",  color=colors_is,  alpha=0.85)
ax.bar(x + w/2, mdd_oos, w, label="OOS", color=colors_oos, alpha=0.85)
ax.axhline(20, color="red", linestyle="--", linewidth=1, alpha=0.6, label="MDD=20%基準")
ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("最大ドローダウン（%）"); ax.set_title("MDD比較（IS vs OOS）")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# 勝率比較
ax = axes[1][0]
wr_is  = [results[p]["IS"]["wr"]  for p in pairs]
wr_oos = [results[p]["OOS"]["wr"] for p in pairs]
ax.bar(x - w/2, wr_is,  w, label="IS",  color=colors_is,  alpha=0.85)
ax.bar(x + w/2, wr_oos, w, label="OOS", color=colors_oos, alpha=0.85)
ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="勝率50%")
ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("勝率（%）"); ax.set_title("勝率比較（IS vs OOS）")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# ケリー係数比較
ax = axes[1][1]
kelly_is  = [results[p]["IS"]["kelly"]  for p in pairs]
kelly_oos = [results[p]["OOS"]["kelly"] for p in pairs]
ax.bar(x - w/2, kelly_is,  w, label="IS",  color=colors_is,  alpha=0.85)
ax.bar(x + w/2, kelly_oos, w, label="OOS", color=colors_oos, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(0.3, color="green", linestyle="--", linewidth=1, alpha=0.6, label="ケリー0.3基準")
ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("ケリー係数"); ax.set_title("ケリー係数比較（IS vs OOS）")
ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "v77_nypro_all_pairs_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"チャート保存: {out_path}")

# ── 図2: スコアカード（ランキング表） ─────────────────────
fig2, ax2 = plt.subplots(figsize=(16, 8))
ax2.axis("off")

col_labels = ["銘柄", "スプレッド", "IS勝率", "IS PF", "IS MDD", "IS月次+",
              "OOS勝率", "OOS PF", "OOS MDD", "OOS月次+", "OOSケリー", "評価"]

def grade(p):
    d = results[p]["OOS"]
    if d["pf"] >= 3.0 and d["mdd"] <= 15 and d["kelly"] >= 0.5:
        return "S"
    elif d["pf"] >= 2.0 and d["mdd"] <= 25 and d["kelly"] >= 0.3:
        return "A"
    elif d["pf"] >= 1.5 and d["mdd"] <= 35:
        return "B"
    elif d["pf"] >= 1.0:
        return "C"
    else:
        return "D"

grade_colors = {"S": "#2ecc71", "A": "#3498db", "B": "#f39c12", "C": "#e67e22", "D": "#e74c3c"}

table_data = []
for p in pairs:
    d_is  = results[p]["IS"]
    d_oos = results[p]["OOS"]
    g = grade(p)
    table_data.append([
        p,
        f'{results[p]["spread"]}pips',
        f'{d_is["wr"]:.1f}%',
        f'{d_is["pf"]:.2f}',
        f'{d_is["mdd"]:.1f}%',
        d_is["mp"],
        f'{d_oos["wr"]:.1f}%',
        f'{d_oos["pf"]:.2f}',
        f'{d_oos["mdd"]:.1f}%',
        d_oos["mp"],
        f'{d_oos["kelly"]:.3f}',
        g,
    ])

# OOS PFでソート
table_data.sort(key=lambda r: float(r[7]), reverse=True)

tbl = ax2.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)

# ヘッダー色
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2c3e50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# 評価列に色付け
for i, row in enumerate(table_data):
    g = row[-1]
    tbl[(i+1, len(col_labels)-1)].set_facecolor(grade_colors.get(g, "white"))
    tbl[(i+1, len(col_labels)-1)].set_text_props(color="white", fontweight="bold")
    # 行の背景（交互）
    bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
    for j in range(len(col_labels)-1):
        tbl[(i+1, j)].set_facecolor(bg)

ax2.set_title("v77（1Hベース）全12銘柄 スコアカード（OOS PF順）\n初期資金100万円・リスク2%・RR2.5 / NYプロコーススプレッド",
              fontsize=12, fontweight="bold", pad=20)

out_path2 = os.path.join(OUT_DIR, "v77_nypro_scorecard.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"スコアカード保存: {out_path2}")
