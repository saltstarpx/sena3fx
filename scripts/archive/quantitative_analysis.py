"""
やがみPA バックテスト + 1トレード 定量分析スクリプト
=====================================================
バックテスト結果と実行トレードを統合して定量分析レポート用の図表を生成する。
"""
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from datetime import datetime, timezone

# 日本語フォント設定
rcParams['font.family'] = 'Noto Sans CJK JP'
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===== データ読み込み =====

def load_backtest_summary():
    path = os.path.join(RESULTS_DIR, 'yagami_backtest_summary.csv')
    return pd.read_csv(path)


def load_best_trades():
    path = os.path.join(RESULTS_DIR, 'best_trades_PA1_Reversal_TightSL.csv')
    return pd.read_csv(path)


def load_trade_result():
    path = os.path.join(RESULTS_DIR, 'latest_trade_result.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_context_bars(context_path):
    df = pd.read_csv(context_path, index_col='timestamp', parse_dates=True)
    return df


# ===== 図1: バックテスト比較棒グラフ =====

def plot_backtest_comparison(summary_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('やがみプライスアクション バックテスト比較\n(USD/JPY 1時間足 2024/07〜2025/02)',
                 fontsize=14, fontweight='bold', y=1.01)

    names = [n.replace('_', '\n') for n in summary_df['name']]
    colors = ['#2196F3' if pf >= 1.0 else '#F44336' for pf in summary_df['profit_factor']]

    # PF
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), summary_df['profit_factor'], color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='PF=1.0')
    ax.set_title('プロフィットファクター (PF)', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylabel('PF')
    for i, v in enumerate(summary_df['profit_factor']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    # 勝率
    ax = axes[0, 1]
    wr_colors = ['#4CAF50' if wr >= 40 else '#FF9800' if wr >= 30 else '#F44336'
                 for wr in summary_df['win_rate_pct']]
    bars = ax.bar(range(len(names)), summary_df['win_rate_pct'], color=wr_colors, alpha=0.85, edgecolor='white')
    ax.axhline(40, color='green', linestyle='--', linewidth=1, label='目標40%')
    ax.set_title('勝率 (%)', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.legend(fontsize=8)
    ax.set_ylabel('勝率 (%)')
    for i, v in enumerate(summary_df['win_rate_pct']):
        ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

    # 最大DD
    ax = axes[1, 0]
    dd_colors = ['#4CAF50' if dd <= 5 else '#FF9800' if dd <= 10 else '#F44336'
                 for dd in summary_df['max_drawdown_pct']]
    bars = ax.bar(range(len(names)), summary_df['max_drawdown_pct'], color=dd_colors, alpha=0.85, edgecolor='white')
    ax.set_title('最大ドローダウン (%)', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('最大DD (%)')
    for i, v in enumerate(summary_df['max_drawdown_pct']):
        ax.text(i, v + 0.02, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)

    # トレード数 & 総リターン
    ax = axes[1, 1]
    x = np.arange(len(names))
    width = 0.35
    ax2 = ax.twinx()
    b1 = ax.bar(x - width/2, summary_df['total_trades'], width, color='#9C27B0', alpha=0.7, label='トレード数')
    ret_colors = ['#4CAF50' if r >= 0 else '#F44336' for r in summary_df['total_return_pct']]
    b2 = ax2.bar(x + width/2, summary_df['total_return_pct'], width, color=ret_colors, alpha=0.7, label='総リターン(%)')
    ax.set_title('トレード数 & 総リターン', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('トレード数', color='#9C27B0')
    ax2.set_ylabel('総リターン (%)', color='gray')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    lines = [mpatches.Patch(color='#9C27B0', alpha=0.7, label='トレード数'),
             mpatches.Patch(color='#4CAF50', alpha=0.7, label='総リターン(+)'),
             mpatches.Patch(color='#F44336', alpha=0.7, label='総リターン(-)')]
    ax.legend(handles=lines, fontsize=7, loc='upper right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_backtest_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図1保存: {path}")
    return path


# ===== 図2: 最良ロジックの累積損益曲線 =====

def plot_equity_curve(trades_df):
    if trades_df.empty:
        return None

    # pnlカラムを確認
    pnl_col = 'pnl' if 'pnl' in trades_df.columns else 'pnl_pips'
    if pnl_col not in trades_df.columns:
        print(f"  警告: pnlカラムが見つかりません。カラム: {trades_df.columns.tolist()}")
        return None

    trades_df = trades_df.copy()
    trades_df['cumulative_pnl'] = trades_df[pnl_col].cumsum()
    trades_df['trade_num'] = range(1, len(trades_df) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('PA1_Reversal_TightSL: 最良ロジック パフォーマンス分析\n(USD/JPY 1時間足)',
                 fontsize=13, fontweight='bold')

    # 累積損益
    ax = axes[0]
    colors = ['#4CAF50' if p >= 0 else '#F44336' for p in trades_df[pnl_col]]
    ax.bar(trades_df['trade_num'], trades_df[pnl_col], color=colors, alpha=0.7, width=0.8)
    ax.plot(trades_df['trade_num'], trades_df['cumulative_pnl'], 'b-o', linewidth=2,
            markersize=4, label='累積損益')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('個別トレード損益 & 累積損益', fontweight='bold')
    ax.set_xlabel('トレード番号')
    ax.set_ylabel('損益')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 勝敗分布
    ax = axes[1]
    wins = trades_df[trades_df[pnl_col] > 0][pnl_col]
    losses = trades_df[trades_df[pnl_col] <= 0][pnl_col]

    bins = np.linspace(trades_df[pnl_col].min(), trades_df[pnl_col].max(), 20)
    ax.hist(wins, bins=bins, color='#4CAF50', alpha=0.7, label=f'勝ちトレード ({len(wins)}件)')
    ax.hist(losses, bins=bins, color='#F44336', alpha=0.7, label=f'負けトレード ({len(losses)}件)')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(wins.mean() if len(wins) > 0 else 0, color='green', linestyle='--', linewidth=1.5,
               label=f'平均利益: {wins.mean():.2f}' if len(wins) > 0 else '')
    ax.axvline(losses.mean() if len(losses) > 0 else 0, color='red', linestyle='--', linewidth=1.5,
               label=f'平均損失: {losses.mean():.2f}' if len(losses) > 0 else '')
    ax.set_title('損益分布ヒストグラム', fontweight='bold')
    ax.set_xlabel('損益')
    ax.set_ylabel('頻度')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_equity_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図2保存: {path}")
    return path


# ===== 図3: 1トレードのローソク足チャート =====

def plot_trade_chart(context_bars, trade_result, signal_info):
    fig, ax = plt.subplots(figsize=(14, 7))

    # ローソク足描画（簡易版）
    bars = context_bars.copy()
    bars = bars.reset_index()
    n = len(bars)
    x = range(n)

    for i, (_, row) in enumerate(bars.iterrows()):
        color = '#4CAF50' if row['close'] >= row['open'] else '#F44336'
        # 実体
        ax.bar(i, abs(row['close'] - row['open']), bottom=min(row['open'], row['close']),
               color=color, width=0.6, alpha=0.9)
        # ヒゲ
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)

    # エントリー・SL・TP ライン
    entry = trade_result['entry_price']
    sl = trade_result['sl']
    tp = trade_result['tp']
    direction = trade_result['signal']

    ax.axhline(entry, color='blue', linestyle='-', linewidth=1.5, alpha=0.8, label=f'エントリー: {entry:.3f}')
    ax.axhline(sl, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'SL: {sl:.3f}')
    ax.axhline(tp, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'TP: {tp:.3f}')

    # 決済ライン
    if trade_result['exit_price']:
        exit_price = trade_result['exit_price']
        exit_color = '#4CAF50' if trade_result['pnl_pips'] > 0 else '#F44336'
        ax.axhline(exit_price, color=exit_color, linestyle=':', linewidth=2.0, alpha=0.9,
                   label=f'決済({trade_result["exit_reason"]}): {exit_price:.3f}')

    # シグナル位置にマーカー
    signal_time = str(signal_info.get('bar_time', ''))
    if 'timestamp' in bars.columns:
        for i, row in bars.iterrows():
            if str(row['timestamp']) == signal_time:
                marker = '^' if direction == 'long' else 'v'
                color = 'blue' if direction == 'long' else 'orange'
                ax.scatter(i, entry, marker=marker, color=color, s=200, zorder=5,
                           label=f'シグナル({direction.upper()})')
                break

    # 軸設定
    tick_step = max(1, n // 15)
    ax.set_xticks(range(0, n, tick_step))
    if 'timestamp' in bars.columns:
        labels = [str(bars['timestamp'].iloc[j])[:16] if j < n else '' for j in range(0, n, tick_step)]
        ax.set_xticklabels(labels, rotation=45, fontsize=7)

    pnl_str = f"+{trade_result['pnl_pips']:.1f}" if trade_result['pnl_pips'] > 0 else f"{trade_result['pnl_pips']:.1f}"
    result_str = 'WIN' if trade_result['pnl_pips'] > 0 else 'LOSS'
    ax.set_title(f'PA1_Reversal_TightSL: 実行トレード ({direction.upper()}) | '
                 f'損益: {pnl_str} pips | {result_str}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('USD/JPY')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_trade_chart.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図3保存: {path}")
    return path


# ===== 図4: 戦略スコアカード =====

def plot_scorecard(summary_df, best_name, trade_result):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    best = summary_df[summary_df['name'] == best_name].iloc[0]

    data = [
        ['指標', '最良ロジック\n(PA1_Reversal_TightSL)', '全戦略平均', '評価'],
        ['プロフィットファクター', f"{best['profit_factor']:.3f}",
         f"{summary_df['profit_factor'].mean():.3f}",
         '✓ PF>1.0' if best['profit_factor'] >= 1.0 else '✗ PF<1.0'],
        ['勝率', f"{best['win_rate_pct']:.1f}%",
         f"{summary_df['win_rate_pct'].mean():.1f}%",
         '✓ 40%超' if best['win_rate_pct'] >= 40 else '△ 40%未満'],
        ['最大DD', f"{best['max_drawdown_pct']:.1f}%",
         f"{summary_df['max_drawdown_pct'].mean():.1f}%",
         '✓ 5%未満' if best['max_drawdown_pct'] < 5 else '△'],
        ['総リターン', f"{best['total_return_pct']:.1f}%",
         f"{summary_df['total_return_pct'].mean():.1f}%",
         '✓ プラス' if best['total_return_pct'] >= 0 else '✗ マイナス'],
        ['トレード数', f"{int(best['total_trades'])}",
         f"{summary_df['total_trades'].mean():.0f}",
         '参考値'],
        ['RR比', f"{best['rr_ratio']:.2f}",
         f"{summary_df['rr_ratio'].mean():.2f}",
         '✓ 1.0超' if best['rr_ratio'] >= 1.0 else '✗'],
        ['', '', '', ''],
        ['実行トレード結果', '', '', ''],
        ['方向', trade_result['signal'].upper(), '', ''],
        ['損益', f"{trade_result['pnl_pips']:.1f} pips", '',
         'WIN' if trade_result['pnl_pips'] > 0 else 'LOSS'],
        ['決済理由', trade_result['exit_reason'], '', ''],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # ヘッダースタイル
    for j in range(4):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 行ごとの色付け
    for i in range(1, len(data)):
        for j in range(4):
            if data[i][0] in ['実行トレード結果', '']:
                table[i, j].set_facecolor('#E3F2FD')
            elif i % 2 == 0:
                table[i, j].set_facecolor('#F5F5F5')

    ax.set_title('やがみPA戦略 スコアカード', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_scorecard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"図4保存: {path}")
    return path


# ===== メイン =====

if __name__ == '__main__':
    print("=== 定量分析 開始 ===")

    summary_df = load_backtest_summary()
    trades_df = load_best_trades()
    trade_result_data = load_trade_result()

    trade_result = trade_result_data['trade']
    signal_info = trade_result_data['signal_info']
    context_path = trade_result_data['bars_context']

    print(f"バックテスト戦略数: {len(summary_df)}")
    print(f"最良ロジックのトレード数: {len(trades_df)}")
    print(f"実行トレード: {trade_result['signal'].upper()} @ {trade_result['entry_price']:.3f}")

    # 図生成
    fig1 = plot_backtest_comparison(summary_df)
    fig2 = plot_equity_curve(trades_df)
    fig3_path = None
    try:
        context_bars = load_context_bars(context_path)
        fig3_path = plot_trade_chart(context_bars, trade_result, signal_info)
    except Exception as e:
        print(f"図3生成エラー: {e}")

    fig4 = plot_scorecard(summary_df, 'PA1_Reversal_TightSL', trade_result)

    print("\n=== 生成された図 ===")
    for path in [fig1, fig2, fig3_path, fig4]:
        if path:
            print(f"  {path}")

    # 図パスをJSONに保存
    figures = {
        'fig1_comparison': fig1,
        'fig2_equity': fig2,
        'fig3_trade_chart': fig3_path,
        'fig4_scorecard': fig4,
    }
    with open(os.path.join(RESULTS_DIR, 'figures_manifest.json'), 'w') as f:
        json.dump(figures, f, indent=2)

    print("\n定量分析完了。")
