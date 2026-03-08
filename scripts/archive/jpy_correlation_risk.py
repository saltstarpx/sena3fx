"""
JPY相関リスク対策検証スクリプト
問題: USDJPY・EURJPY・GBPJPYを同時保有すると「円の逆張り3倍」になる局面がある
対策: 同方向（円買い or 円売り）のポジションを最大N本に制限する
検証: 制限なし vs 同方向1本まで vs 同方向2本まで の損益・最大DD比較
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA    = "/home/ubuntu/sena3fx/data"
RESULTS = "/home/ubuntu/sena3fx/results"

def load_trades(path, pair=None):
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    if pair:
        df["pair"] = pair
    return df

# 各ペアのトレードログを読み込む
df_usd = load_trades(f"{RESULTS}/v76_all_oanda_trades.csv", pair="USDJPY")
df_eur = load_trades(f"{RESULTS}/v76_eurjpy_trades.csv")
df_gbp = load_trades(f"{RESULTS}/v76_gbpjpy_trades.csv")

# 全トレードを統合してentry_time順にソート
df_all = pd.concat([df_usd, df_eur, df_gbp], ignore_index=True)
df_all = df_all.sort_values("entry_time").reset_index(drop=True)

print(f"全トレード数: {len(df_all)}")
print(f"期間: {df_all['entry_time'].min()} 〜 {df_all['entry_time'].max()}")

# ---- 同時保有状況の分析 ----
def analyze_concurrent(df):
    """各トレードのエントリー時点で何本のポジションが同時保有されているかを分析"""
    concurrent_counts = []
    same_dir_counts = []

    for idx, row in df.iterrows():
        entry = row["entry_time"]
        direction = row["dir"]
        # このトレードのエントリー時点で保有中のポジション
        active = df[(df["entry_time"] < entry) & (df["exit_time"] > entry)]
        concurrent_counts.append(len(active))
        # 同方向（円売り or 円買い）のポジション数
        # dir=1はロング（円売り）、dir=-1はショート（円買い）
        same_dir = active[active["dir"] == direction]
        same_dir_counts.append(len(same_dir))

    df = df.copy()
    df["concurrent"] = concurrent_counts
    df["same_dir_count"] = same_dir_counts
    return df

print("\n同時保有状況を分析中...", flush=True)
df_analyzed = analyze_concurrent(df_all)

print(f"\n同時保有数の分布:")
print(df_analyzed["concurrent"].value_counts().sort_index())
print(f"\n同方向同時保有数の分布:")
print(df_analyzed["same_dir_count"].value_counts().sort_index())

# 同時保有が2本以上の場合の損益分析
df_high_conc = df_analyzed[df_analyzed["concurrent"] >= 2]
df_low_conc  = df_analyzed[df_analyzed["concurrent"] < 2]
print(f"\n同時保有2本以上のトレード: {len(df_high_conc)}回")
print(f"  勝率: {(df_high_conc['pnl'] > 0).mean()*100:.1f}%")
print(f"  平均損益: {df_high_conc['pnl'].mean():+.1f}pips")
print(f"同時保有1本以下のトレード: {len(df_low_conc)}回")
print(f"  勝率: {(df_low_conc['pnl'] > 0).mean()*100:.1f}%")
print(f"  平均損益: {df_low_conc['pnl'].mean():+.1f}pips")

# ---- 同時保有制限シミュレーション ----
def simulate_with_limit(df, max_same_dir=None, max_total=None):
    """
    同方向最大N本 or 総保有最大N本の制限を設けたシミュレーション
    制限に引っかかったトレードはスキップ
    """
    active_positions = []  # (entry_time, exit_time, dir, pair)
    accepted_trades = []
    skipped = 0

    for idx, row in df.iterrows():
        entry = row["entry_time"]
        direction = row["dir"]

        # 期限切れポジションを除去
        active_positions = [p for p in active_positions if p["exit_time"] > entry]

        # 制限チェック
        skip = False
        if max_same_dir is not None:
            same_dir_active = sum(1 for p in active_positions if p["dir"] == direction)
            if same_dir_active >= max_same_dir:
                skip = True
        if max_total is not None and not skip:
            if len(active_positions) >= max_total:
                skip = True

        if skip:
            skipped += 1
            continue

        # エントリー受け入れ
        active_positions.append({
            "entry_time": entry,
            "exit_time": row["exit_time"],
            "dir": direction,
            "pair": row["pair"]
        })
        accepted_trades.append(row)

    df_accepted = pd.DataFrame(accepted_trades)
    return df_accepted, skipped

# 3パターンを比較
scenarios = [
    ("制限なし",           None, None),
    ("同方向最大1本",      1,    None),
    ("同方向最大2本",      2,    None),
    ("総保有最大2本",      None, 2),
    ("総保有最大3本",      None, 3),
]

print("\n\n=== 同時保有制限シミュレーション ===")
scenario_results = []

for name, max_sd, max_tot in scenarios:
    df_acc, skipped = simulate_with_limit(df_all.copy(), max_same_dir=max_sd, max_total=max_tot)
    if df_acc.empty:
        continue
    wins   = df_acc[df_acc["pnl"] > 0]
    losses = df_acc[df_acc["pnl"] < 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else float("inf")
    wr     = len(wins) / len(df_acc) * 100
    total  = df_acc["pnl"].sum()

    # 最大ドローダウン計算
    cumulative = df_acc.sort_values("entry_time")["pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_dd = drawdown.min()

    t_stat, p_val = stats.ttest_1samp(df_acc["pnl"], 0)
    print(f"\n{name}")
    print(f"  採用: {len(df_acc)}回 / スキップ: {skipped}回")
    print(f"  勝率: {wr:.1f}%  PF: {pf:.2f}  総損益: {total:+.1f}pips")
    print(f"  最大DD: {max_dd:+.1f}pips  p値: {p_val:.4f}")

    scenario_results.append(dict(
        name=name, trades=len(df_acc), skipped=skipped,
        win_rate=wr, pf=pf, total_pnl=total, max_dd=max_dd,
        p_value=p_val, cumulative=cumulative.sort_index()
    ))

# ---- チャート ----
plt.rcParams["font.family"] = "Noto Sans CJK JP"
fig, axes = plt.subplots(2, 1, figsize=(14, 12))
fig.patch.set_facecolor("#0d0d1a")

# エクイティカーブ比較
ax = axes[0]
colors_map = {
    "制限なし":      "#3498db",
    "同方向最大1本": "#e74c3c",
    "同方向最大2本": "#2ecc71",
    "総保有最大2本": "#f39c12",
    "総保有最大3本": "#9b59b6",
}
for sr in scenario_results:
    cum = sr["cumulative"].reset_index(drop=True)
    ax.plot(range(len(cum)), cum.values,
            color=colors_map.get(sr["name"], "#ffffff"),
            linewidth=1.5 if sr["name"] == "制限なし" else 1.0,
            label=f"{sr['name']} ({sr['trades']}回 / PF{sr['pf']:.2f} / DD{sr['max_dd']:+.0f}pips)",
            alpha=0.9)
ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
ax.set_title("同時保有制限シナリオ別 エクイティカーブ比較", fontsize=11, color="white")
ax.set_ylabel("累積損益 (pips)", color="white")
ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
ax.set_facecolor("#1a1a2e")
ax.tick_params(colors="white")

# 最大DD比較
ax2 = axes[1]
names  = [sr["name"] for sr in scenario_results]
dds    = [abs(sr["max_dd"]) for sr in scenario_results]
totals = [sr["total_pnl"] for sr in scenario_results]
x = np.arange(len(names))
width = 0.35
bars1 = ax2.bar(x - width/2, totals, width, color="#2ecc71", label="総損益 (pips)", alpha=0.8)
ax2_r = ax2.twinx()
bars2 = ax2_r.bar(x + width/2, dds, width, color="#e74c3c", label="最大DD (pips)", alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=15, fontsize=9)
ax2.set_ylabel("総損益 (pips)", color="#2ecc71")
ax2_r.set_ylabel("最大DD (pips)", color="#e74c3c")
ax2.set_title("シナリオ別 総損益 vs 最大ドローダウン", fontsize=11, color="white")
ax2.set_facecolor("#1a1a2e")
ax2.tick_params(colors="white")
ax2_r.tick_params(colors="white")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, facecolor="#1a1a2e", labelcolor="white", fontsize=9)

plt.tight_layout(pad=2.0)
out = f"{RESULTS}/v76_jpy_correlation_risk.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.close()
print(f"\nチャート保存: {out}")

# 同時保有の同方向ペア組み合わせ分析
print("\n\n=== 同方向同時保有の組み合わせ分析 ===")
df_analyzed2 = df_analyzed[df_analyzed["same_dir_count"] >= 2].copy()
if len(df_analyzed2) > 0:
    print(f"同方向2本以上同時保有のトレード: {len(df_analyzed2)}回")
    print(f"  うち損失: {(df_analyzed2['pnl'] < 0).sum()}回")
    print(f"  平均損益: {df_analyzed2['pnl'].mean():+.1f}pips")
    print(f"  合計損益: {df_analyzed2['pnl'].sum():+.1f}pips")
else:
    print("同方向2本以上の同時保有なし")
