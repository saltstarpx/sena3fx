import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def analyze_v61_factors():
    # トレード履歴の読み込み
    trades_df = pd.read_csv("/home/ubuntu/sena3fx/results/trades_v61.csv")
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['hour'] = trades_df['exit_time'].dt.hour
    
    # trade_idごとに集計（PnLを合計）
    final_trades = trades_df.groupby('trade_id').agg({
        'pnl_pips': 'sum',
        'hour': 'first',
        'exit_reason': lambda x: ' / '.join(x.unique())
    })

    # 1. 時間帯別分析
    hourly_stats = final_trades.groupby('hour')['pnl_pips'].agg(['sum', 'count', 'mean'])
    
    # 2. 決済理由別分析
    exit_stats = trades_df.groupby('exit_reason')['pnl_pips'].agg(['sum', 'count', 'mean'])

    # 3. ケリー係数の算出
    win_trades = final_trades[final_trades['pnl_pips'] > 0]['pnl_pips']
    loss_trades = final_trades[final_trades['pnl_pips'] <= 0]['pnl_pips']
    
    win_rate = len(win_trades) / len(final_trades) if len(final_trades) > 0 else 0
    avg_win = win_trades.mean() if len(win_trades) > 0 else 0
    avg_loss = abs(loss_trades.mean()) if len(loss_trades) > 0 else 0
    odds = avg_win / avg_loss if avg_loss > 0 else 0
    kelly_f = (odds * win_rate - (1 - win_rate)) / odds if odds > 0 else 0

    # グラフ作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # 時間帯別累積損益
    sns.barplot(x=hourly_stats.index, y=hourly_stats['sum'], ax=axes[0], palette='viridis')
    axes[0].set_title('v61: 時間帯別累積損益 (pips)')
    axes[0].set_xlabel('時間 (Hour, UTC)')
    axes[0].set_ylabel('累積損益 (pips)')

    # 決済理由別件数
    sns.barplot(x=exit_stats.index, y=exit_stats['count'], ax=axes[1], palette='magma')
    axes[1].set_title('v61: 決済理由別トレード件数')
    axes[1].set_xlabel('決済理由')
    axes[1].set_ylabel('件数')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("/home/ubuntu/sena3fx/results/v61_factor_analysis.png")
    
    # 結果の出力
    print("--- v61 Factor Analysis Results ---")
    print(f"Total Trades: {len(final_trades)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Win: {avg_win:.4f} pips, Avg Loss: {avg_loss:.4f} pips")
    print(f"Odds Ratio: {odds:.4f}")
    print(f"Kelly Fraction: {kelly_f:.4f}")
    print(f"Total PnL: {final_trades['pnl_pips'].sum():.2f} pips")
    print("\n--- Exit Reason Stats ---")
    print(exit_stats)

if __name__ == "__main__":
    analyze_v61_factors()
