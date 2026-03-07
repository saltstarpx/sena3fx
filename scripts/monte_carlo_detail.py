"""
monte_carlo_detail.py
======================
v75のモンテカルロシミュレーション詳細分析

【分析内容】
1. 資金推移のパーセンタイル分布（5th/25th/50th/75th/95th）
2. 最終資金の分布ヒストグラム
3. 最大ドローダウンの分布
4. リスク設定別の比較（0.5%, 1.0%, 1.5%, 2.0%）
5. 連敗シナリオの影響分析
"""
import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

OUT = "/home/ubuntu/sena3fx/results"

# v75の統計値（全決済ベース：HALF_TP含む）
WIN_RATE  = 0.688
AVG_WIN   = 78.04   # pips
AVG_LOSS  = 36.31   # pips（正の値）
N_TRADES  = 53      # 実際のトレード数（7ヶ月）
N_SIM     = 10000   # シミュレーション回数
SEED      = 42

# 実際のトレード損益シーケンス（v75の実データ）
# CSVから読み込む
TRADES_CSV = f"{OUT}/trades_v75.csv"


def load_actual_trades():
    """実際のトレードデータを読み込む"""
    if os.path.exists(TRADES_CSV):
        df = pd.read_csv(TRADES_CSV)
        return df["pnl"].values
    return None


def run_monte_carlo(win_rate, avg_win, avg_loss, risk_pct,
                    n_trades=N_TRADES, n_sim=N_SIM, seed=SEED,
                    ruin_threshold=50.0):
    """
    モンテカルロシミュレーション

    Returns:
    --------
    capitals: shape (n_sim, n_trades+1) の資金推移行列
    max_dds: shape (n_sim,) の最大ドローダウン配列
    final_caps: shape (n_sim,) の最終資金配列
    ruin_rate: 破産確率
    """
    np.random.seed(seed)
    payoff = avg_win / avg_loss

    capitals = np.zeros((n_sim, n_trades + 1))
    capitals[:, 0] = 100.0  # 初期資金100%

    for i in range(n_trades):
        rand = np.random.random(n_sim)
        wins = rand < win_rate
        # 勝ち: +risk_pct × payoff
        # 負け: -risk_pct
        gain = np.where(wins,
                        capitals[:, i] * (risk_pct / 100) * payoff,
                        -capitals[:, i] * (risk_pct / 100))
        capitals[:, i + 1] = capitals[:, i] + gain

    # 最大ドローダウン計算
    max_dds = np.zeros(n_sim)
    for s in range(n_sim):
        cap_series = capitals[s]
        peak = cap_series[0]
        max_dd = 0.0
        for c in cap_series:
            if c > peak:
                peak = c
            dd = (peak - c) / peak * 100
            if dd > max_dd:
                max_dd = dd
        max_dds[s] = max_dd

    final_caps = capitals[:, -1]
    ruin_rate = np.mean(final_caps <= ruin_threshold)

    return capitals, max_dds, final_caps, ruin_rate


def calc_percentiles(capitals):
    """各トレード時点でのパーセンタイルを計算"""
    pcts = [5, 25, 50, 75, 95]
    result = {}
    for p in pcts:
        result[p] = np.percentile(capitals, p, axis=0)
    return result


if __name__ == "__main__":
    risk_configs = [
        {"risk": 0.5,  "color": "#4CAF50", "label": "0.5%/トレード（保守的）"},
        {"risk": 1.0,  "color": "#2196F3", "label": "1.0%/トレード（安全）"},
        {"risk": 1.5,  "color": "#FF9800", "label": "1.5%/トレード（ハーフケリー）"},
        {"risk": 2.0,  "color": "#F44336", "label": "2.0%/トレード（積極的）"},
    ]

    print("=" * 65)
    print("  v75 モンテカルロシミュレーション詳細分析")
    print(f"  シミュレーション回数: {N_SIM:,}回 / トレード数: {N_TRADES}回")
    print(f"  勝率: {WIN_RATE:.1%} / 平均利益: +{AVG_WIN:.1f}pips / 平均損失: -{AVG_LOSS:.1f}pips")
    print("=" * 65)

    all_results = {}
    for cfg in risk_configs:
        risk = cfg["risk"]
        capitals, max_dds, final_caps, ruin_rate = run_monte_carlo(
            WIN_RATE, AVG_WIN, AVG_LOSS, risk,
            n_trades=N_TRADES, n_sim=N_SIM, ruin_threshold=50.0
        )
        pcts = calc_percentiles(capitals)
        all_results[risk] = {
            "capitals": capitals,
            "max_dds": max_dds,
            "final_caps": final_caps,
            "ruin_rate": ruin_rate,
            "pcts": pcts,
        }

        print(f"\n  【{cfg['label']}】")
        print(f"  破産確率（資金50%以下）: {ruin_rate*100:.3f}%")
        print(f"  最終資金（中央値）:      {np.median(final_caps):.1f}%")
        print(f"  最終資金（5th〜95th）:   {np.percentile(final_caps,5):.1f}% 〜 {np.percentile(final_caps,95):.1f}%")
        print(f"  最大DD（中央値）:        {np.median(max_dds):.1f}%")
        print(f"  最大DD（95th）:          {np.percentile(max_dds,95):.1f}%")
        print(f"  プラス終了確率:          {np.mean(final_caps > 100)*100:.1f}%")

    # ────────────────────────────────────────────────
    # 可視化（2×3グリッド）
    # ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    x_trades = np.arange(N_TRADES + 1)

    # ── 1. リスク1.0%の資金推移パーセンタイル ──
    ax1 = fig.add_subplot(gs[0, 0])
    r = all_results[1.0]
    pcts = r["pcts"]
    ax1.fill_between(x_trades, pcts[5], pcts[95], alpha=0.12, color="#2196F3", label="5th〜95th")
    ax1.fill_between(x_trades, pcts[25], pcts[75], alpha=0.22, color="#2196F3", label="25th〜75th")
    ax1.plot(x_trades, pcts[50], color="#2196F3", lw=2.2, label="中央値（50th）")
    ax1.plot(x_trades, pcts[5],  color="#2196F3", lw=0.8, ls="--", alpha=0.7)
    ax1.plot(x_trades, pcts[95], color="#2196F3", lw=0.8, ls="--", alpha=0.7)
    ax1.axhline(100, color="gray", lw=0.8, ls=":")
    ax1.axhline(50,  color="#F44336", lw=0.8, ls=":", label="破産ライン(50%)")
    ax1.set_title("資金推移パーセンタイル\n（リスク1.0%/トレード）", fontsize=11)
    ax1.set_xlabel("トレード数"); ax1.set_ylabel("資金残高（初期=100%）")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.35)

    # ── 2. リスク1.5%の資金推移パーセンタイル ──
    ax2 = fig.add_subplot(gs[0, 1])
    r = all_results[1.5]
    pcts = r["pcts"]
    ax2.fill_between(x_trades, pcts[5], pcts[95], alpha=0.12, color="#FF9800", label="5th〜95th")
    ax2.fill_between(x_trades, pcts[25], pcts[75], alpha=0.22, color="#FF9800", label="25th〜75th")
    ax2.plot(x_trades, pcts[50], color="#FF9800", lw=2.2, label="中央値（50th）")
    ax2.plot(x_trades, pcts[5],  color="#FF9800", lw=0.8, ls="--", alpha=0.7)
    ax2.plot(x_trades, pcts[95], color="#FF9800", lw=0.8, ls="--", alpha=0.7)
    ax2.axhline(100, color="gray", lw=0.8, ls=":")
    ax2.axhline(50,  color="#F44336", lw=0.8, ls=":", label="破産ライン(50%)")
    ax2.set_title("資金推移パーセンタイル\n（リスク1.5%/トレード・ハーフケリー）", fontsize=11)
    ax2.set_xlabel("トレード数"); ax2.set_ylabel("資金残高（初期=100%）")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.35)

    # ── 3. 最終資金の分布比較 ──
    ax3 = fig.add_subplot(gs[0, 2])
    for cfg in risk_configs:
        r = all_results[cfg["risk"]]
        ax3.hist(r["final_caps"], bins=60, alpha=0.45, color=cfg["color"],
                 label=f"{cfg['risk']:.1f}% (中央値:{np.median(r['final_caps']):.0f}%)",
                 density=True)
    ax3.axvline(100, color="black", lw=1.2, ls="--", label="初期資金")
    ax3.axvline(50,  color="#F44336", lw=1.0, ls=":", label="破産ライン")
    ax3.set_title(f"最終資金の分布\n（{N_SIM:,}回シミュレーション）", fontsize=11)
    ax3.set_xlabel("最終資金（%）"); ax3.set_ylabel("確率密度")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.35)
    ax3.set_xlim(0, 400)

    # ── 4. 最大ドローダウンの分布比較 ──
    ax4 = fig.add_subplot(gs[1, 0])
    for cfg in risk_configs:
        r = all_results[cfg["risk"]]
        ax4.hist(r["max_dds"], bins=50, alpha=0.45, color=cfg["color"],
                 label=f"{cfg['risk']:.1f}% (中央値:{np.median(r['max_dds']):.1f}%)",
                 density=True)
    ax4.axvline(20, color="#FF9800", lw=1.0, ls=":", label="20%ライン")
    ax4.axvline(30, color="#F44336", lw=1.0, ls=":", label="30%ライン")
    ax4.set_title("最大ドローダウンの分布", fontsize=11)
    ax4.set_xlabel("最大ドローダウン（%）"); ax4.set_ylabel("確率密度")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.35)

    # ── 5. リスク別サマリー棒グラフ ──
    ax5 = fig.add_subplot(gs[1, 1])
    risks = [cfg["risk"] for cfg in risk_configs]
    medians = [np.median(all_results[r]["final_caps"]) for r in risks]
    p5s    = [np.percentile(all_results[r]["final_caps"], 5) for r in risks]
    p95s   = [np.percentile(all_results[r]["final_caps"], 95) for r in risks]
    colors = [cfg["color"] for cfg in risk_configs]
    x = np.arange(len(risks))
    bars = ax5.bar(x, medians, color=colors, alpha=0.75, width=0.5, label="中央値")
    for i, (p5, p95, med) in enumerate(zip(p5s, p95s, medians)):
        ax5.errorbar(i, med, yerr=[[med-p5], [p95-med]],
                     fmt='none', color='black', capsize=5, lw=1.5)
        ax5.text(i, p95 + 3, f"{med:.0f}%", ha='center', fontsize=9, fontweight='bold')
    ax5.axhline(100, color="gray", lw=0.8, ls="--", label="初期資金")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{r:.1f}%" for r in risks])
    ax5.set_xlabel("リスク/トレード"); ax5.set_ylabel("最終資金（%）")
    ax5.set_title("リスク別 最終資金\n（中央値 ± 5th〜95th）", fontsize=11)
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.35, axis="y")

    # ── 6. 破産確率 & プラス終了確率 ──
    ax6 = fig.add_subplot(gs[1, 2])
    ruin_rates  = [all_results[r]["ruin_rate"] * 100 for r in risks]
    plus_rates  = [np.mean(all_results[r]["final_caps"] > 100) * 100 for r in risks]
    x = np.arange(len(risks)); w = 0.35
    ax6.bar(x - w/2, plus_rates,  w, color="#4CAF50", alpha=0.8, label="プラス終了確率")
    ax6.bar(x + w/2, ruin_rates,  w, color="#F44336", alpha=0.8, label="破産確率（資金50%以下）")
    for i, (pl, ru) in enumerate(zip(plus_rates, ruin_rates)):
        ax6.text(i - w/2, pl + 0.5, f"{pl:.1f}%", ha='center', fontsize=8)
        ax6.text(i + w/2, ru + 0.5, f"{ru:.2f}%", ha='center', fontsize=8)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"{r:.1f}%" for r in risks])
    ax6.set_xlabel("リスク/トレード"); ax6.set_ylabel("確率（%）")
    ax6.set_title("プラス終了確率 vs 破産確率", fontsize=11)
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.35, axis="y")

    fig.suptitle(
        f"v75 モンテカルロシミュレーション詳細分析\n"
        f"（勝率{WIN_RATE:.0%}・平均利益+{AVG_WIN:.0f}pips・平均損失-{AVG_LOSS:.0f}pips・{N_TRADES}トレード・{N_SIM:,}回試行）",
        fontsize=13
    )

    out_path = f"{OUT}/v75_monte_carlo_detail.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart: {out_path}")

    # ── テキストサマリー ──
    print(f"\n{'='*65}")
    print("  【リスク設定別 詳細サマリー】")
    print(f"{'='*65}")
    print(f"  {'設定':<22} {'中央値':>8} {'5th':>8} {'95th':>8} {'最大DD中央':>10} {'破産確率':>10} {'プラス率':>9}")
    print(f"  {'-'*75}")
    for cfg in risk_configs:
        r = cfg["risk"]
        res = all_results[r]
        med  = np.median(res["final_caps"])
        p5   = np.percentile(res["final_caps"], 5)
        p95  = np.percentile(res["final_caps"], 95)
        mdd  = np.median(res["max_dds"])
        ruin = res["ruin_rate"] * 100
        plus = np.mean(res["final_caps"] > 100) * 100
        print(f"  {cfg['label']:<22} {med:>7.1f}% {p5:>7.1f}% {p95:>7.1f}% {mdd:>9.1f}% {ruin:>9.3f}% {plus:>8.1f}%")
