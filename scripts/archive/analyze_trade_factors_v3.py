import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def analyze_factors_v3():
    # トレード履歴の読み込み
    trades_path = "/home/ubuntu/sena3fx/results/trades_v60_kelly.csv"
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return
    
    trades_df = pd.read_csv(trades_path)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['hour'] = trades_df['exit_time'].dt.hour
    trades_df['is_win'] = trades_df['pnl_pips'] > 0
    
    # ケリー係数
    win_rate = len(trades_df[trades_df['is_win']]) / len(trades_df)
    avg_win = trades_df[trades_df['is_win']]['pnl_pips'].mean()
    avg_loss = abs(trades_df[~trades_df['is_win']]['pnl_pips'].mean())
    odds = avg_win / avg_loss
    kelly_f = (odds * win_rate - (1 - win_rate)) / odds
    
    # 時間帯別損益
    hour_stats = trades_df.groupby('hour')['pnl_pips'].agg(['count', 'sum', 'mean'])
    
    # エグジット理由別損益
    reason_stats = trades_df.groupby('exit_reason')['pnl_pips'].agg(['count', 'sum', 'mean'])
    
    print("=== Factor Analysis v3 ===")
    print(f"Kelly Fraction: {kelly_f:.4f}")
    print("\n=== Hour Statistics (Worst 5 Hours) ===")
    print(hour_stats.sort_values('sum').head(5))
    print("\n=== Exit Reason Statistics ===")
    print(reason_stats)
    
    # グラフ作成
    plt.figure(figsize=(15, 10))
    
    # 時間帯別損益
    plt.subplot(2, 2, 1)
    sns.barplot(x=hour_stats.index, y=hour_stats['sum'], palette='vlag')
    plt.title('Total PnL by Hour (Feb 2026)')
    
    # エグジット理由別損益
    plt.subplot(2, 2, 2)
    sns.barplot(x=reason_stats.index, y=reason_stats['sum'], palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Total PnL by Exit Reason')
    
    plt.tight_layout()
    plt.savefig("/home/ubuntu/sena3fx/results/v60_factor_analysis_v3.png")
    print("\nAnalysis graph saved to: /home/ubuntu/sena3fx/results/v60_factor_analysis_v3.png")

if __name__ == "__main__":
    analyze_factors_v3()
