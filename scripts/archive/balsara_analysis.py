"""
balsara_analysis.py
====================
v75のバルサラ破産確率を計算し、損失許容率別の破産確率を分析する

【バルサラ破産確率の計算式】
破産確率 = ((1 - Edge) / (1 + Edge)) ^ (Capital / Unit_Risk)

Edge（優位性）= 勝率 × 平均利益率 - 負率 × 平均損失率
Capital = 総資金（許容損失率で割った値）
Unit_Risk = 1トレードのリスク（資金に対する割合）
"""
import pandas as pd
import numpy as np
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'

OUT = "/home/ubuntu/sena3fx/results"


def balsara_ruin_probability(win_rate, avg_win, avg_loss, risk_per_trade_pct, ruin_pct=100):
    """
    バルサラ破産確率を計算する

    Parameters:
    -----------
    win_rate: 勝率（0〜1）
    avg_win: 平均利益（pips）
    avg_loss: 平均損失（pips）※正の値で渡す
    risk_per_trade_pct: 1トレードのリスク（資金に対する%）
    ruin_pct: 破産とみなす損失率（デフォルト100%=全損）

    Returns:
    --------
    破産確率（0〜1）
    """
    if avg_loss == 0:
        return 0.0

    loss_rate = 1 - win_rate
    # ペイオフレシオ（平均利益/平均損失）
    payoff_ratio = avg_win / avg_loss

    # エッジ（優位性）
    edge = win_rate * payoff_ratio - loss_rate

    if edge <= 0:
        return 1.0  # エッジがマイナスなら必ず破産

    # バルサラの公式
    # Z = ((1-edge)/(1+edge))
    # 破産確率 = Z^(ruin_units)
    # ruin_units = 破産とみなす損失額 / 1トレードのリスク額
    ruin_units = ruin_pct / risk_per_trade_pct

    z = (1 - edge) / (1 + edge)
    if z >= 1:
        return 1.0

    prob = z ** ruin_units
    return min(prob, 1.0)


def monte_carlo_ruin(win_rate, avg_win, avg_loss, risk_per_trade_pct,
                     ruin_threshold=50, n_trades=1000, n_sim=5000, seed=42):
    """
    モンテカルロシミュレーションによる破産確率の推定

    ruin_threshold: 資金がこの%以下になったら破産とみなす
    """
    np.random.seed(seed)
    ruin_count = 0
    capital_start = 100.0  # 資金を100%として正規化

    for _ in range(n_sim):
        capital = capital_start
        for _ in range(n_trades):
            if capital <= ruin_threshold:
                ruin_count += 1
                break
            if np.random.random() < win_rate:
                # 勝ち: 資金の risk_per_trade_pct × payoff_ratio 増加
                gain = capital * (risk_per_trade_pct / 100) * (avg_win / avg_loss)
                capital += gain
            else:
                # 負け: 資金の risk_per_trade_pct 減少
                loss = capital * (risk_per_trade_pct / 100)
                capital -= loss

    return ruin_count / n_sim


if __name__ == "__main__":
    # v75の統計値を読み込む
    trades_path = f"{OUT}/trades_v75_ref.csv"
    if not os.path.exists(trades_path):
        # v75のトレードデータが別の場所にある場合
        trades_path = f"{OUT}/trades_v75.csv"

    if os.path.exists(trades_path):
        df = pd.read_csv(trades_path)
    else:
        # データがない場合は既知の統計値を使用
        print("トレードデータを直接使用します（v75の既知統計値）")
        df = None

    # v75の統計値（バックテスト結果から）
    # SL/TP決済のみを対象（HALF_TPは除く）
    if df is not None:
        closed = df[df["type"].isin(["SL", "TP"])]
        wins = closed[closed["pnl"] > 0]
        losses = closed[closed["pnl"] < 0]
        win_rate = len(wins) / len(closed) if len(closed) > 0 else 0
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 0
        n_trades = len(closed)
        total_pnl = closed["pnl"].sum()
        print(f"v75 統計値（SL/TP決済のみ）:")
        print(f"  トレード数: {n_trades}回")
        print(f"  勝率: {win_rate:.1%}")
        print(f"  平均利益: +{avg_win:.2f}pips")
        print(f"  平均損失: -{avg_loss:.2f}pips")
        print(f"  ペイオフレシオ: {avg_win/avg_loss:.2f}")
    else:
        # 全決済（HALF_TP含む）の統計値を使用
        win_rate = 0.688
        avg_win = 78.04
        avg_loss = 36.31
        n_trades = 53
        total_pnl = 2707.53
        print(f"v75 統計値（全決済ベース）:")
        print(f"  トレード数: {n_trades}回")
        print(f"  勝率: {win_rate:.1%}")
        print(f"  平均利益: +{avg_win:.2f}pips")
        print(f"  平均損失: -{avg_loss:.2f}pips")

    payoff = avg_win / avg_loss
    edge = win_rate * payoff - (1 - win_rate)
    kelly = edge / payoff
    print(f"  ペイオフレシオ: {payoff:.2f}")
    print(f"  エッジ: {edge:.4f}")
    print(f"  ケリー基準: {kelly:.4f} → ハーフケリー: {kelly/2:.4f} ({kelly/2*100:.2f}%/トレード)")

    # ────────────────────────────────────────────────
    # バルサラ破産確率の計算（損失許容率別）
    # ────────────────────────────────────────────────
    risk_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]  # 1トレードのリスク%
    ruin_thresholds = [20, 30, 50]  # 破産とみなす損失率%

    print(f"\n{'='*70}")
    print("  バルサラ破産確率（理論値）")
    print(f"{'='*70}")
    print(f"  {'リスク/トレード':<18}", end="")
    for rt in ruin_thresholds:
        print(f"  {'破産='+str(rt)+'%損失':<18}", end="")
    print()
    print(f"  {'-'*65}")

    balsara_results = {}
    for risk in risk_levels:
        row = {}
        print(f"  {risk:.1f}%/トレード{'':<8}", end="")
        for rt in ruin_thresholds:
            prob = balsara_ruin_probability(win_rate, avg_win, avg_loss, risk, rt)
            row[rt] = prob
            mark = "✓" if prob < 0.01 else ("△" if prob < 0.05 else "✗")
            print(f"  {mark} {prob*100:.4f}%{'':<10}", end="")
        print()
        balsara_results[risk] = row

    # ────────────────────────────────────────────────
    # モンテカルロ破産確率（50%損失をルイン基準）
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  モンテカルロ破産確率（5000回シミュレーション、資金50%以下で破産）")
    print(f"{'='*70}")
    print(f"  {'リスク/トレード':<18} {'破産確率':<15} {'推奨'}")
    print(f"  {'-'*50}")

    mc_results = {}
    for risk in risk_levels:
        prob = monte_carlo_ruin(win_rate, avg_win, avg_loss, risk,
                                ruin_threshold=50, n_trades=500, n_sim=5000)
        mc_results[risk] = prob
        if prob < 0.005:
            rec = "✓ 非常に安全"
        elif prob < 0.02:
            rec = "✓ 安全"
        elif prob < 0.05:
            rec = "△ 許容範囲"
        elif prob < 0.10:
            rec = "△ やや危険"
        else:
            rec = "✗ 危険"
        print(f"  {risk:.1f}%/トレード{'':<8} {prob*100:.3f}%{'':<8} {rec}")

    # ────────────────────────────────────────────────
    # ケリー基準との比較
    # ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  リスク設定の推奨値まとめ")
    print(f"{'='*70}")
    print(f"  フルケリー:   {kelly*100:.2f}%/トレード（理論上最大成長、実用には過剰）")
    print(f"  ハーフケリー: {kelly/2*100:.2f}%/トレード（推奨：安定成長）")
    print(f"  クォーターケリー: {kelly/4*100:.2f}%/トレード（保守的・安全重視）")

    # ────────────────────────────────────────────────
    # 可視化
    # ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. バルサラ破産確率（理論値）
    ax1 = axes[0]
    colors = ["#2196F3", "#FF9800", "#F44336"]
    for rt, color in zip(ruin_thresholds, colors):
        probs = [balsara_results[r][rt] * 100 for r in risk_levels]
        ax1.plot(risk_levels, probs, marker="o", label=f"破産={rt}%損失", color=color, lw=2)
    ax1.axvline(kelly/2*100, color="green", ls="--", lw=1.5, label=f"ハーフケリー({kelly/2*100:.2f}%)")
    ax1.axhline(1, color="gray", ls=":", lw=1, label="1%ライン")
    ax1.axhline(5, color="orange", ls=":", lw=1, label="5%ライン")
    ax1.set_xlabel("1トレードのリスク（資金に対する%）")
    ax1.set_ylabel("破産確率（%）")
    ax1.set_title("バルサラ破産確率（理論値）", fontsize=12)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)
    ax1.set_ylim(-0.5, 20)

    # 2. モンテカルロ破産確率
    ax2 = axes[1]
    mc_probs = [mc_results[r] * 100 for r in risk_levels]
    bar_colors = ["#4CAF50" if p < 2 else ("#FF9800" if p < 5 else "#F44336") for p in mc_probs]
    bars = ax2.bar(risk_levels, mc_probs, color=bar_colors, alpha=0.8, width=0.3)
    ax2.axvline(kelly/2*100, color="green", ls="--", lw=1.5, label=f"ハーフケリー({kelly/2*100:.2f}%)")
    ax2.axhline(2, color="orange", ls=":", lw=1, label="2%ライン")
    ax2.axhline(5, color="red", ls=":", lw=1, label="5%ライン")
    for bar, prob in zip(bars, mc_probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{prob:.2f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("1トレードのリスク（資金に対する%）")
    ax2.set_ylabel("破産確率（%）")
    ax2.set_title("モンテカルロ破産確率\n（500トレード×5000回、資金50%以下で破産）", fontsize=11)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4, axis="y")

    # 3. 資金成長シミュレーション（各リスク設定）
    ax3 = axes[2]
    np.random.seed(42)
    sim_risks = [0.5, 1.0, 1.5, 2.0, 3.0]
    sim_colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    n_sim_trades = 53  # v75の実際のトレード数

    for risk, color in zip(sim_risks, sim_colors):
        capitals = [100.0]
        cap = 100.0
        for _ in range(n_sim_trades):
            if np.random.random() < win_rate:
                cap *= (1 + risk/100 * payoff)
            else:
                cap *= (1 - risk/100)
            capitals.append(cap)
        ax3.plot(range(len(capitals)), capitals, color=color, lw=1.5,
                 label=f"{risk:.1f}%/トレード → {capitals[-1]:.0f}%")

    ax3.axhline(100, color="gray", ls="--", lw=0.8)
    ax3.set_xlabel("トレード数")
    ax3.set_ylabel("資金残高（初期=100%）")
    ax3.set_title(f"資金成長シミュレーション\n（勝率{win_rate:.0%}、PO{payoff:.1f}、{n_sim_trades}トレード）", fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

    plt.suptitle(f"v75 バルサラ破産確率分析（勝率{win_rate:.0%}、PF{avg_win/avg_loss*win_rate/(1-win_rate):.2f}）",
                 fontsize=13)
    plt.tight_layout()
    out_path = f"{OUT}/v75_balsara_analysis.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Chart: {out_path}")
