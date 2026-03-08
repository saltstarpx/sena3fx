"""
RUN-20260305-002: 4時間足フィルター比較分析・可視化
"""
import sys, os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams['font.family'] = 'Noto Sans CJK JP'
rcParams['axes.unicode_minus'] = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'yagami_htf_backtest_summary.csv'))
    return df


# ===== 図5: RUN-001 vs RUN-002 比較 =====
def plot_comparison(df):
    baseline = df[df['htf_filter'] == 'なし'].iloc[0]
    htf_df = df[df['htf_filter'] == 'あり'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('RUN-001 vs RUN-002: 4時間足フィルター効果の検証\n(USD/JPY 1時間足 2024/07〜2025/02)',
                 fontsize=13, fontweight='bold')

    short_names = [n.replace('HTF_', '').replace('_sl', '\nsl').replace('_tp', '\ntp')
                   for n in htf_df['name']]

    # PF比較
    ax = axes[0]
    colors = ['#2196F3' if pf >= 1.0 else '#F44336' for pf in htf_df['profit_factor']]
    ax.barh(range(len(short_names)), htf_df['profit_factor'], color=colors, alpha=0.8)
    ax.axvline(baseline['profit_factor'], color='gold', linewidth=2.5, linestyle='--',
               label=f'ベースライン PF={baseline["profit_factor"]:.3f}')
    ax.axvline(1.0, color='black', linewidth=1, linestyle=':', label='PF=1.0')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title('プロフィットファクター', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel('PF')

    # 勝率比較
    ax = axes[1]
    wr_colors = ['#4CAF50' if wr >= 40 else '#FF9800' if wr >= 30 else '#F44336'
                 for wr in htf_df['win_rate_pct']]
    ax.barh(range(len(short_names)), htf_df['win_rate_pct'], color=wr_colors, alpha=0.8)
    ax.axvline(baseline['win_rate_pct'], color='gold', linewidth=2.5, linestyle='--',
               label=f'ベースライン {baseline["win_rate_pct"]:.1f}%')
    ax.axvline(40, color='green', linewidth=1, linestyle=':', label='目標40%')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title('勝率 (%)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel('勝率 (%)')

    # トレード数 vs 総リターン
    ax = axes[2]
    ret_colors = ['#4CAF50' if r >= 0 else '#F44336' for r in htf_df['total_return_pct']]
    scatter = ax.scatter(htf_df['total_trades'], htf_df['total_return_pct'],
                         c=htf_df['profit_factor'], cmap='RdYlGn', s=100,
                         vmin=0.3, vmax=1.1, zorder=3)
    ax.scatter([baseline['total_trades']], [baseline['total_return_pct']],
               marker='*', s=300, color='gold', zorder=5, label=f'ベースライン')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(baseline['total_trades'], color='gold', linewidth=1, linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='PF')
    ax.set_title('トレード数 vs 総リターン', fontweight='bold')
    ax.set_xlabel('トレード数')
    ax.set_ylabel('総リターン (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_htf_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図5保存: {path}")
    return path


# ===== 図6: フィルター除外率とPFの関係 =====
def plot_filter_effect(df):
    baseline = df[df['htf_filter'] == 'なし'].iloc[0]
    htf_df = df[df['htf_filter'] == 'あり'].copy()
    baseline_n = baseline['total_trades']

    # 除外率の近似（ベースラインとの差）
    htf_df = htf_df.copy()
    htf_df['exclusion_rate'] = (1 - htf_df['total_trades'] / baseline_n) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('4時間足フィルター: 除外率・ゾーン幅とパフォーマンスの関係',
                 fontsize=12, fontweight='bold')

    # 除外率 vs PF
    ax = axes[0]
    zone_colors = {1.0: '#F44336', 1.5: '#FF9800', 2.0: '#4CAF50'}
    for z, grp in htf_df.groupby('htf_zone_atr'):
        ax.scatter(grp['exclusion_rate'], grp['profit_factor'],
                   color=zone_colors.get(z, 'gray'), s=80, label=f'zone_atr={z}', zorder=3)
    ax.axhline(baseline['profit_factor'], color='gold', linewidth=2, linestyle='--',
               label=f'ベースライン PF={baseline["profit_factor"]:.3f}')
    ax.axhline(1.0, color='black', linewidth=1, linestyle=':', label='PF=1.0')
    ax.set_xlabel('フィルター除外率 (%)')
    ax.set_ylabel('プロフィットファクター')
    ax.set_title('除外率 vs PF', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ゾーン幅別ボックスプロット
    ax = axes[1]
    zone_groups = [htf_df[htf_df['htf_zone_atr'] == z]['profit_factor'].values
                   for z in [1.0, 1.5, 2.0]]
    bp = ax.boxplot(zone_groups, labels=['zone_atr=1.0', 'zone_atr=1.5', 'zone_atr=2.0'],
                    patch_artist=True)
    colors_box = ['#F44336', '#FF9800', '#4CAF50']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(baseline['profit_factor'], color='gold', linewidth=2, linestyle='--',
               label=f'ベースライン PF={baseline["profit_factor"]:.3f}')
    ax.axhline(1.0, color='black', linewidth=1, linestyle=':')
    ax.set_ylabel('プロフィットファクター')
    ax.set_title('ゾーン幅別 PF分布', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_filter_effect.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図6保存: {path}")
    return path


# ===== 図7: 仮説検証サマリー =====
def plot_hypothesis_summary(df):
    baseline = df[df['htf_filter'] == 'なし'].iloc[0]
    best_htf = df[df['htf_filter'] == 'あり'].sort_values('profit_factor', ascending=False).iloc[0]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis('off')

    data = [
        ['検証項目', 'ベースライン\n(フィルターなし)', '最良HTFフィルター\n(lb20_z1.5)', '変化', '判定'],
        ['プロフィットファクター', f"{baseline['profit_factor']:.3f}",
         f"{best_htf['profit_factor']:.3f}",
         f"{best_htf['profit_factor'] - baseline['profit_factor']:+.3f}",
         '悪化' if best_htf['profit_factor'] < baseline['profit_factor'] else '改善'],
        ['勝率', f"{baseline['win_rate_pct']:.1f}%",
         f"{best_htf['win_rate_pct']:.1f}%",
         f"{best_htf['win_rate_pct'] - baseline['win_rate_pct']:+.1f}%",
         '悪化' if best_htf['win_rate_pct'] < baseline['win_rate_pct'] else '改善'],
        ['最大DD', f"{baseline['max_drawdown_pct']:.1f}%",
         f"{best_htf['max_drawdown_pct']:.1f}%",
         f"{best_htf['max_drawdown_pct'] - baseline['max_drawdown_pct']:+.1f}%",
         '改善' if best_htf['max_drawdown_pct'] < baseline['max_drawdown_pct'] else '悪化'],
        ['総リターン', f"{baseline['total_return_pct']:.2f}%",
         f"{best_htf['total_return_pct']:.2f}%",
         f"{best_htf['total_return_pct'] - baseline['total_return_pct']:+.2f}%",
         '悪化' if best_htf['total_return_pct'] < baseline['total_return_pct'] else '改善'],
        ['トレード数', f"{int(baseline['total_trades'])}",
         f"{int(best_htf['total_trades'])}",
         f"{int(best_htf['total_trades'] - baseline['total_trades'])}",
         '減少（想定内）'],
        ['', '', '', '', ''],
        ['仮説', '4時間足フィルターでPF>1.2、勝率>45%を達成できる', '', '', ''],
        ['検証結果', '仮説は棄却。フィルターにより勝率・PFともに低下。', '', '', ''],
        ['原因考察', '現在のデータ期間（8ヶ月）でHTFゾーンが機能しない可能性', '', '', ''],
        ['次のアクション', '1時間足シグナル自体の精度向上を優先する方向へ転換', '', '', ''],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for j in range(5):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 判定列の色付け
    for i in range(1, 6):
        val = data[i][4]
        color = '#FFCDD2' if '悪化' in val else '#C8E6C9' if '改善' in val else '#FFF9C4'
        table[i, 4].set_facecolor(color)

    for i in range(7, len(data)):
        for j in range(5):
            table[i, j].set_facecolor('#E3F2FD')

    ax.set_title('RUN-20260305-002: 仮説検証サマリー', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig7_hypothesis_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図7保存: {path}")
    return path


if __name__ == '__main__':
    print("=== RUN-002 定量分析 ===")
    df = load_data()
    fig5 = plot_comparison(df)
    fig6 = plot_filter_effect(df)
    fig7 = plot_hypothesis_summary(df)
    print("完了。")
