"""
quantitative_analysis_v75.py
============================
v75取引履歴の定量分析・計量分析

【分析内容】
1. 基本統計量（平均・中央値・標準偏差・歪度・尖度）
2. ケリー基準の詳細分析（分数ケリー含む）
3. モンテカルロシミュレーション（1000回）
4. 時間帯別パフォーマンス分析
5. 連勝・連敗の分布分析
6. t検定（期待値がゼロと有意に異なるか）
7. シャープレシオ・カルマーレシオ
8. ドローダウン分析
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

OUT = "/home/ubuntu/sena3fx/results"
os.makedirs(OUT, exist_ok=True)

# ── データ読み込み ──────────────────────────────────────────
df_raw = pd.read_csv(f"{OUT}/trades_v75.csv")
df_raw["exit_time"] = pd.to_datetime(df_raw["exit_time"])
df_raw["entry_time"] = pd.to_datetime(df_raw["time"])

# 決済済みトレードのみ（HALF_TPはSL/TPの補助）
df_closed = df_raw[df_raw["type"].isin(["SL", "TP"])].copy()
df_all = df_raw.copy()  # HALF_TP含む全損益

pnl = df_closed["pnl"].values
n = len(pnl)

print("=" * 65)
print("  v75 定量分析・計量分析レポート")
print("=" * 65)

# ── 1. 基本統計量 ──────────────────────────────────────────
wins   = pnl[pnl > 0]
losses = pnl[pnl < 0]
nw, nl = len(wins), len(losses)
wr = nw / n * 100
avg_w = wins.mean() if nw > 0 else 0
avg_l = losses.mean() if nl > 0 else 0
pf = wins.sum() / abs(losses.sum()) if nl > 0 else float("inf")
expectancy = pnl.mean()
std_pnl = pnl.std()
skew = stats.skew(pnl)
kurt = stats.kurtosis(pnl)

print(f"\n【1. 基本統計量】")
print(f"  トレード数    : {n}回 (勝{nw} 負{nl})")
print(f"  勝率          : {wr:.1f}%")
print(f"  平均利益      : {avg_w:+.2f}pips")
print(f"  平均損失      : {avg_l:+.2f}pips")
print(f"  プロフィットF : {pf:.2f}")
print(f"  1回あたり期待値: {expectancy:+.2f}pips")
print(f"  標準偏差      : {std_pnl:.2f}pips")
print(f"  歪度 (Skew)   : {skew:.3f}  {'（右歪み＝大勝ちが多い）' if skew > 0 else '（左歪み＝大負けが多い）'}")
print(f"  尖度 (Kurt)   : {kurt:.3f}  {'（ファットテール）' if kurt > 0 else '（薄いテール）'}")

# ── 2. ケリー基準 ──────────────────────────────────────────
b = abs(avg_w / avg_l) if avg_l != 0 else 0
p = nw / n
q = 1 - p
kelly_full = (b * p - q) / b if b != 0 else 0
kelly_half = kelly_full / 2
kelly_quarter = kelly_full / 4

print(f"\n【2. ケリー基準】")
print(f"  フルケリー    : {kelly_full:.4f} ({kelly_full*100:.1f}%)")
print(f"  ハーフケリー  : {kelly_half:.4f} ({kelly_half*100:.1f}%)  ← 推奨")
print(f"  クォーターK   : {kelly_quarter:.4f} ({kelly_quarter*100:.1f}%)  ← 保守的")
print(f"  ※ 証拠金の{kelly_half*100:.1f}%を1トレードのリスクとして使用するのが最適")

# ── 3. t検定（期待値の有意性） ──────────────────────────────
t_stat, p_value = stats.ttest_1samp(pnl, 0)
print(f"\n【3. 統計的有意性検定（t検定）】")
print(f"  帰無仮説: 期待値 = 0（ランダムと変わらない）")
print(f"  t統計量  : {t_stat:.4f}")
print(f"  p値      : {p_value:.4f}")
if p_value < 0.01:
    print(f"  結論     : *** 1%水準で有意 → ロジックに統計的優位性あり")
elif p_value < 0.05:
    print(f"  結論     : ** 5%水準で有意 → ロジックに統計的優位性あり")
elif p_value < 0.10:
    print(f"  結論     : * 10%水準で有意 → 弱い優位性")
else:
    print(f"  結論     : 有意差なし → サンプル数不足またはロジックに優位性なし")

# ── 4. シャープレシオ・カルマーレシオ ──────────────────────
# 1日あたりのリターンに換算（月平均6回 → 年72回）
daily_pnl = df_all.groupby(df_all["exit_time"].dt.date)["pnl"].sum()
sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

# ドローダウン
cum_pnl = df_all["pnl"].cumsum()
roll_max = cum_pnl.cummax()
drawdown = cum_pnl - roll_max
max_dd = drawdown.min()
total_return = cum_pnl.iloc[-1]
calmar = (total_return / abs(max_dd)) if max_dd != 0 else float("inf")

print(f"\n【4. リスク調整後リターン】")
print(f"  シャープレシオ (年換算): {sharpe:.3f}")
print(f"  最大ドローダウン      : {max_dd:.2f}pips")
print(f"  カルマーレシオ        : {calmar:.3f}")
print(f"  総損益 / 最大DD       : {total_return:.1f} / {abs(max_dd):.1f}")

# ── 5. 連勝・連敗分析 ──────────────────────────────────────
results = (pnl > 0).astype(int)  # 1=勝, 0=負
streaks_win = []; streaks_lose = []
cur_w = cur_l = 0
for r in results:
    if r == 1:
        cur_w += 1
        if cur_l > 0: streaks_lose.append(cur_l); cur_l = 0
    else:
        cur_l += 1
        if cur_w > 0: streaks_win.append(cur_w); cur_w = 0
if cur_w > 0: streaks_win.append(cur_w)
if cur_l > 0: streaks_lose.append(cur_l)

print(f"\n【5. 連勝・連敗分析】")
print(f"  最大連勝: {max(streaks_win) if streaks_win else 0}連勝")
print(f"  最大連敗: {max(streaks_lose) if streaks_lose else 0}連敗")
print(f"  平均連勝: {np.mean(streaks_win):.1f}回" if streaks_win else "  平均連勝: -")
print(f"  平均連敗: {np.mean(streaks_lose):.1f}回" if streaks_lose else "  平均連敗: -")

# ── 6. 時間帯別分析 ──────────────────────────────────────
df_closed2 = df_closed.copy()
df_closed2["hour"] = df_closed2["exit_time"].dt.hour
hourly = df_closed2.groupby("hour").agg(
    count=("pnl","count"),
    total=("pnl","sum"),
    wr=("pnl", lambda x: (x > 0).mean() * 100)
).reset_index()

print(f"\n【6. 時間帯別パフォーマンス（UTC）】")
print(f"  {'時間帯':<8} {'回数':<6} {'損益':<12} {'勝率'}")
for _, row in hourly.iterrows():
    print(f"  {int(row['hour']):02d}:00    {int(row['count']):<6} {row['total']:+8.1f}pips  {row['wr']:.0f}%")

# ── 7. モンテカルロシミュレーション ──────────────────────
np.random.seed(42)
N_SIM = 2000
N_TRADES = n
sim_finals = []
sim_max_dds = []

for _ in range(N_SIM):
    sim_pnl = np.random.choice(pnl, size=N_TRADES, replace=True)
    cum = np.cumsum(sim_pnl)
    sim_finals.append(cum[-1])
    roll_max_s = np.maximum.accumulate(cum)
    dd_s = cum - roll_max_s
    sim_max_dds.append(dd_s.min())

sim_finals = np.array(sim_finals)
sim_max_dds = np.array(sim_max_dds)

print(f"\n【7. モンテカルロシミュレーション（{N_SIM}回）】")
print(f"  最終損益の分布:")
print(f"    中央値  : {np.median(sim_finals):+.1f}pips")
print(f"    5%ile   : {np.percentile(sim_finals, 5):+.1f}pips  ← 最悪ケース")
print(f"    25%ile  : {np.percentile(sim_finals, 25):+.1f}pips")
print(f"    75%ile  : {np.percentile(sim_finals, 75):+.1f}pips")
print(f"    95%ile  : {np.percentile(sim_finals, 95):+.1f}pips  ← 最良ケース")
print(f"    プラス確率: {(sim_finals > 0).mean()*100:.1f}%")
print(f"  最大DDの分布:")
print(f"    中央値  : {np.median(sim_max_dds):.1f}pips")
print(f"    95%ile  : {np.percentile(sim_max_dds, 95):.1f}pips  ← 想定最大DD")

# ── 可視化 ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# (1) 累積損益
ax1 = fig.add_subplot(gs[0, :2])
cum_all = df_all["pnl"].cumsum()
ax1.plot(range(len(cum_all)), cum_all.values, color="#2196F3", lw=1.5)
ax1.axhline(0, color="gray", lw=0.8, ls="--")
ax1.fill_between(range(len(cum_all)), cum_all.values, 0,
                 where=(cum_all.values >= 0), alpha=0.15, color="#2196F3")
ax1.fill_between(range(len(cum_all)), cum_all.values, 0,
                 where=(cum_all.values < 0), alpha=0.15, color="#F44336")
ax1.set_title("累積損益曲線 (pips)", fontsize=12)
ax1.set_ylabel("pips"); ax1.grid(True, alpha=0.4)

# (2) ドローダウン
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.6, color="#F44336")
ax2.set_title("ドローダウン (pips)", fontsize=12)
ax2.set_ylabel("pips"); ax2.grid(True, alpha=0.4)

# (3) 損益分布ヒストグラム
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(pnl, bins=20, color="#2196F3", alpha=0.7, edgecolor="white")
ax3.axvline(0, color="red", lw=1, ls="--")
ax3.axvline(expectancy, color="orange", lw=1.5, ls="-", label=f"期待値={expectancy:.1f}")
ax3.set_title("損益分布", fontsize=12)
ax3.set_xlabel("pips"); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)

# (4) モンテカルロ最終損益分布
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(sim_finals, bins=50, color="#4CAF50", alpha=0.7, edgecolor="white")
ax4.axvline(np.percentile(sim_finals, 5), color="red", lw=1.5, ls="--", label=f"5%ile={np.percentile(sim_finals,5):.0f}")
ax4.axvline(np.median(sim_finals), color="orange", lw=1.5, ls="-", label=f"中央値={np.median(sim_finals):.0f}")
ax4.set_title(f"モンテカルロ最終損益分布 ({N_SIM}回)", fontsize=12)
ax4.set_xlabel("pips"); ax4.legend(fontsize=9); ax4.grid(True, alpha=0.4)

# (5) モンテカルロ最大DD分布
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(sim_max_dds, bins=50, color="#FF9800", alpha=0.7, edgecolor="white")
ax5.axvline(np.percentile(sim_max_dds, 95), color="red", lw=1.5, ls="--",
            label=f"95%ile={np.percentile(sim_max_dds,95):.0f}")
ax5.set_title("モンテカルロ最大DD分布", fontsize=12)
ax5.set_xlabel("pips"); ax5.legend(fontsize=9); ax5.grid(True, alpha=0.4)

# (6) 時間帯別損益
ax6 = fig.add_subplot(gs[2, :2])
colors_h = ["#2196F3" if v >= 0 else "#F44336" for v in hourly["total"]]
bars = ax6.bar(hourly["hour"], hourly["total"], color=colors_h, alpha=0.8)
ax6.axhline(0, color="gray", lw=0.8)
ax6.set_title("時間帯別損益 (UTC)", fontsize=12)
ax6.set_xlabel("時間 (UTC)"); ax6.set_ylabel("pips")
ax6.set_xticks(range(0, 24, 2)); ax6.grid(True, alpha=0.4)

# (7) 連勝・連敗分布
ax7 = fig.add_subplot(gs[2, 2])
max_streak = max(max(streaks_win) if streaks_win else 0, max(streaks_lose) if streaks_lose else 0)
bins_s = range(1, max_streak + 2)
ax7.hist(streaks_win,  bins=bins_s, alpha=0.7, color="#2196F3", label="連勝", align="left")
ax7.hist(streaks_lose, bins=bins_s, alpha=0.7, color="#F44336",  label="連敗", align="left")
ax7.set_title("連勝・連敗分布", fontsize=12)
ax7.set_xlabel("連続回数"); ax7.set_ylabel("頻度")
ax7.legend(fontsize=9); ax7.grid(True, alpha=0.4)

plt.suptitle("v75 定量分析・計量分析レポート", fontsize=15, y=1.01)
out_path = f"{OUT}/v75_quantitative_analysis.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  分析チャート: {out_path}")

# ── レポートをMarkdownで保存 ──────────────────────────────
report = f"""# v75 定量分析・計量分析レポート

## 1. 基本統計量

| 指標 | 値 |
| :--- | :--- |
| トレード数 | {n}回（勝{nw} 負{nl}） |
| 勝率 | {wr:.1f}% |
| 平均利益 | {avg_w:+.2f}pips |
| 平均損失 | {avg_l:+.2f}pips |
| プロフィットファクター | {pf:.2f} |
| 1回あたり期待値 | {expectancy:+.2f}pips |
| 標準偏差 | {std_pnl:.2f}pips |
| 歪度 | {skew:.3f}（{'右歪み＝大勝ちが多い' if skew > 0 else '左歪み＝大負けが多い'}） |
| 尖度 | {kurt:.3f}（{'ファットテール' if kurt > 0 else '薄いテール'}） |

## 2. ケリー基準

| 種別 | 値 | 推奨 |
| :--- | :--- | :--- |
| フルケリー | {kelly_full:.4f}（{kelly_full*100:.1f}%） | 理論値（実用には過大） |
| **ハーフケリー** | **{kelly_half:.4f}（{kelly_half*100:.1f}%）** | **推奨** |
| クォーターケリー | {kelly_quarter:.4f}（{kelly_quarter*100:.1f}%） | 保守的 |

## 3. 統計的有意性検定（t検定）

- t統計量: {t_stat:.4f}
- p値: {p_value:.4f}
- 結論: {'*** 1%水準で有意（ロジックに統計的優位性あり）' if p_value < 0.01 else ('** 5%水準で有意' if p_value < 0.05 else '有意差なし（サンプル数不足の可能性）')}

## 4. リスク調整後リターン

| 指標 | 値 |
| :--- | :--- |
| シャープレシオ（年換算） | {sharpe:.3f} |
| 最大ドローダウン | {max_dd:.2f}pips |
| カルマーレシオ | {calmar:.3f} |

## 5. 連勝・連敗分析

| 指標 | 値 |
| :--- | :--- |
| 最大連勝 | {max(streaks_win) if streaks_win else 0}連勝 |
| 最大連敗 | {max(streaks_lose) if streaks_lose else 0}連敗 |
| 平均連勝 | {np.mean(streaks_win):.1f}回 |
| 平均連敗 | {np.mean(streaks_lose):.1f}回 |

## 6. モンテカルロシミュレーション（{N_SIM}回）

| パーセンタイル | 最終損益 | 最大DD |
| :--- | :--- | :--- |
| 5%ile（最悪ケース） | {np.percentile(sim_finals, 5):+.1f}pips | {np.percentile(sim_max_dds, 95):.1f}pips |
| 25%ile | {np.percentile(sim_finals, 25):+.1f}pips | - |
| 中央値 | {np.median(sim_finals):+.1f}pips | {np.median(sim_max_dds):.1f}pips |
| 75%ile | {np.percentile(sim_finals, 75):+.1f}pips | - |
| 95%ile（最良ケース） | {np.percentile(sim_finals, 95):+.1f}pips | - |
| プラス確率 | {(sim_finals > 0).mean()*100:.1f}% | - |
"""

with open(f"{OUT}/v75_quantitative_report.md", "w") as f:
    f.write(report)
print(f"  レポート: {OUT}/v75_quantitative_report.md")
