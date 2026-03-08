"""
RUN-20260305-003: 方向A・方向B 比較分析・可視化
"""
import sys, os
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

BASELINE_PF = 1.030
BASELINE_WR = 40.91
BASELINE_N  = 22


def load():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'run003_summary.csv'))
    return df


# ===== 図8: 全戦略比較（方向A・B・ベースライン） =====
def plot_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RUN-20260305-003: 方向A・方向B 全戦略比較\n(USD/JPY 1時間足 2024/07〜2025/02)',
                 fontsize=13, fontweight='bold')

    color_map = {'baseline': '#FFD700', 'A': '#2196F3', 'B': '#4CAF50'}
    labels_short = {
        'BASELINE': 'BASE',
        'A1_IB_only': 'A1\nIBのみ', 'A2_Vol_only': 'A2\nVolのみ',
        'A3_Sess_only': 'A3\nSessのみ', 'A4_IB_Vol': 'A4\nIB+Vol',
        'A5_IB_Sess': 'A5\nIB+Sess', 'A6_Vol_Sess': 'A6\nVol+Sess',
        'A7_All': 'A7\n全部', 'A8_All_HighVol': 'A8\n高Vol', 'A9_LondonNY': 'A9\nLDN/NY',
        'B1_EMA20_50_noSlope': 'B1\n20/50', 'B2_EMA20_50_slope': 'B2\n20/50s',
        'B3_EMA10_30_noSlope': 'B3\n10/30', 'B4_EMA10_30_slope': 'B4\n10/30s',
        'B5_EMA50_200_noSlope': 'B5\n50/200', 'B6_EMA20_100_noSlope': 'B6\n20/100',
        'B7_EMA20_50_strongSlp': 'B7\n強Slp', 'B8_EMA10_50_noSlope': 'B8\n10/50',
        'B9_EMA30_100_noSlope': 'B9\n30/100',
    }

    names = df['name'].tolist()
    short = [labels_short.get(n, n) for n in names]
    colors = [color_map.get(d, 'gray') for d in df['direction']]
    x = range(len(df))

    # PF
    ax = axes[0, 0]
    pf_vals = df['profit_factor'].clip(upper=5.0)
    bars = ax.bar(x, pf_vals, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(BASELINE_PF, color='gold', linewidth=2, linestyle='--', label=f'ベースライン {BASELINE_PF}')
    ax.axhline(1.0, color='black', linewidth=1, linestyle=':', label='PF=1.0')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5)
    ax.set_title('プロフィットファクター（上限5でクリップ）', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylabel('PF')
    # 異常値に注記
    for i, (v, orig) in enumerate(zip(pf_vals, df['profit_factor'])):
        if orig > 5:
            ax.text(i, v + 0.05, f'N={int(df["total_trades"].iloc[i])}\n(少)' , ha='center', fontsize=5.5, color='red')

    # 勝率
    ax = axes[0, 1]
    ax.bar(x, df['win_rate_pct'], color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(BASELINE_WR, color='gold', linewidth=2, linestyle='--', label=f'ベースライン {BASELINE_WR:.1f}%')
    ax.axhline(40, color='green', linewidth=1, linestyle=':', label='目標40%')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5)
    ax.set_title('勝率 (%)', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylabel('勝率 (%)')

    # トレード数
    ax = axes[1, 0]
    ax.bar(x, df['total_trades'], color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(BASELINE_N, color='gold', linewidth=2, linestyle='--', label=f'ベースライン N={BASELINE_N}')
    ax.axhline(10, color='red', linewidth=1, linestyle=':', label='最低10件ライン')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5)
    ax.set_title('トレード数（統計的信頼性）', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylabel('件数')

    # 総リターン
    ax = axes[1, 1]
    ret_colors = ['#4CAF50' if r >= 0 else '#F44336' for r in df['total_return_pct']]
    ax.bar(x, df['total_return_pct'], color=ret_colors, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(df[df['direction']=='baseline']['total_return_pct'].iloc[0],
               color='gold', linewidth=2, linestyle='--', label='ベースライン')
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.5)
    ax.set_title('総リターン (%)', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylabel('リターン (%)')

    # 凡例パッチ
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               [('#FFD700','ベースライン'), ('#2196F3','方向A（精度向上）'), ('#4CAF50','方向B（EMAトレンド）')]]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(FIGURES_DIR, 'fig8_run003_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図8保存: {path}")
    return path


# ===== 図9: 統計的信頼性フィルター後の有効戦略比較 =====
def plot_valid_only(df):
    # N>=8 かつ 方向Bの異常値（PF>5）を除外
    valid = df[(df['total_trades'] >= 8) & (df['profit_factor'] <= 5.0)].copy()

    if len(valid) == 0:
        print("有効戦略なし（N>=8）")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle('RUN-003: 統計的信頼性あり戦略のみ（N≥8、PF≤5）\nベースラインとの比較',
                 fontsize=12, fontweight='bold')

    color_map = {'baseline': '#FFD700', 'A': '#2196F3', 'B': '#4CAF50'}
    colors = [color_map.get(d, 'gray') for d in valid['direction']]
    short_names = [n.replace('_', '\n') for n in valid['name']]
    x = range(len(valid))

    for ax, col, title, hline, hline_label in [
        (axes[0], 'profit_factor', 'プロフィットファクター', BASELINE_PF, f'BASE {BASELINE_PF}'),
        (axes[1], 'win_rate_pct',  '勝率 (%)',              BASELINE_WR, f'BASE {BASELINE_WR:.1f}%'),
        (axes[2], 'total_return_pct', '総リターン (%)',     0.02,        'BASE +0.02%'),
    ]:
        ax.bar(x, valid[col], color=colors, alpha=0.85, edgecolor='white')
        ax.axhline(hline, color='gold', linewidth=2, linestyle='--', label=hline_label)
        if col == 'profit_factor':
            ax.axhline(1.0, color='black', linewidth=1, linestyle=':', label='PF=1.0')
        if col == 'win_rate_pct':
            ax.axhline(40, color='green', linewidth=1, linestyle=':', label='目標40%')
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=7)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    patches = [mpatches.Patch(color=c, label=l) for c, l in
               [('#FFD700','ベースライン'), ('#2196F3','方向A'), ('#4CAF50','方向B')]]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(FIGURES_DIR, 'fig9_run003_valid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図9保存: {path}")
    return path


# ===== 図10: 3サイクル PDCA 進捗サマリー =====
def plot_pdca_progress(df):
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis('off')

    # 方向Aの最良（N>=8）
    a_valid = df[(df['direction']=='A') & (df['total_trades']>=8) & (df['profit_factor']<=5)]
    a_best = a_valid.sort_values('profit_factor', ascending=False).iloc[0] if len(a_valid) > 0 else None

    # 方向Bの最良（N>=5）
    b_valid = df[(df['direction']=='B') & (df['total_trades']>=5) & (df['profit_factor']<=5)]
    b_best = b_valid.sort_values('profit_factor', ascending=False).iloc[0] if len(b_valid) > 0 else None

    rows = [
        ['', 'RUN-001\nベースライン', 'RUN-002\n4hゾーンフィルター', 'RUN-003\n方向A（精度向上）', 'RUN-003\n方向B（EMAトレンド）'],
        ['PF', '1.030', '0.763', f"{a_best['profit_factor']:.3f}" if a_best is not None else '-', '99.9\n(N少・無効)'],
        ['勝率', '40.9%', '29.4%', f"{a_best['win_rate_pct']:.1f}%" if a_best is not None else '-', '100%\n(N少・無効)'],
        ['最大DD', '0.3%', '0.4%', f"{a_best['max_drawdown_pct']:.1f}%" if a_best is not None else '-', '0.0%\n(N少・無効)'],
        ['総リターン', '+0.02%', '-0.16%', f"{a_best['total_return_pct']:+.2f}%" if a_best is not None else '-', '-'],
        ['トレード数', '22', '17', f"{int(a_best['total_trades'])}" if a_best is not None else '-', '3〜4\n(統計不足)'],
        ['最良戦略名', 'PA1_TightSL', 'HTF_lb20_z1.5', a_best['name'] if a_best is not None else '-', '判定不可'],
        ['', '', '', '', ''],
        ['判定', '✓ PF>1.0\n損益分岐点', '✗ 仮説棄却\nゾーン型不適合', '△ PF>1.0維持\nサンプル不足', '✗ 統計的無効\nデータ不足'],
        ['次アクション', '→ フィルター追加', '→ 方向転換', '→ データ拡充\n(2年分)', '→ データ拡充\n(2年分)'],
    ]

    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # ヘッダー色
    header_colors = ['#37474F', '#1565C0', '#6A1B9A', '#0277BD', '#2E7D32']
    for j, c in enumerate(header_colors):
        table[0, j].set_facecolor(c)
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 判定行の色
    for j in range(5):
        cell = table[9, j]
        text = rows[9][j]
        if '✓' in text:
            cell.set_facecolor('#C8E6C9')
        elif '✗' in text:
            cell.set_facecolor('#FFCDD2')
        elif '△' in text:
            cell.set_facecolor('#FFF9C4')

    ax.set_title('3サイクル PDCA 進捗サマリー（RUN-001〜003）',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig10_pdca_progress.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図10保存: {path}")
    return path


if __name__ == '__main__':
    print("=== RUN-003 定量分析 ===")
    df = load()
    plot_overview(df)
    plot_valid_only(df)
    plot_pdca_progress(df)
    print("完了。")
