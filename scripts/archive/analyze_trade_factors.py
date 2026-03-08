import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 自作モジュールのインポート
sys.path.append('/home/ubuntu/sena3fx')

def analyze_factors():
    # トレード履歴の読み込み（フィルター無効 0.0x の結果を使用）
    trades_path = "/home/ubuntu/sena3fx/results/trades_v60_00x.csv"
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return
    
    trades_df = pd.read_csv(trades_path)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['hour'] = trades_df['exit_time'].dt.hour
    trades_df['day_of_week'] = trades_df['exit_time'].dt.dayofweek # 0=Mon, 6=Sun
    
    # 損益をpips換算
    trades_df['pnl_pips'] = trades_df['pnl'] / 0.01
    trades_df['is_win'] = trades_df['pnl_pips'] > 0
    
    # 1. 時間帯別分析
    hour_stats = trades_df.groupby('hour')['pnl_pips'].agg(['count', 'sum', 'mean'])
    
    # 2. 曜日別分析
    day_stats = trades_df.groupby('day_of_week')['pnl_pips'].agg(['count', 'sum', 'mean'])
    
    # 3. ケリー係数の算出
    # ケリー基準: f = (bp - q) / b
    # b: オッズ（利益/損失）、p: 勝率、q: 敗率 (1-p)
    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['is_win']]
    loss_trades = trades_df[~trades_df['is_win']]
    
    win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
    loss_rate = 1 - win_rate
    
    avg_win = win_trades['pnl_pips'].mean() if len(win_trades) > 0 else 0
    avg_loss = abs(loss_trades['pnl_pips'].mean()) if len(loss_trades) > 0 else 1 # 0除算回避
    
    odds = avg_win / avg_loss if avg_loss > 0 else 0
    
    # ケリー係数
    kelly_f = (odds * win_rate - loss_rate) / odds if odds > 0 else 0
    
    # 結果の表示
    print("=== Quantitative Analysis ===")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win (pips): {avg_win:.2f}")
    print(f"Average Loss (pips): {avg_loss:.2f}")
    print(f"Odds (Risk/Reward): {odds:.2f}")
    print(f"Kelly Fraction: {kelly_f:.4f}")
    
    print("\n=== Hour Statistics (Top 5 Loss Hours) ===")
    print(hour_stats.sort_values('sum').head(5))
    
    # グラフ作成
    plt.figure(figsize=(15, 10))
    
    # 時間帯別損益
    plt.subplot(2, 2, 1)
    sns.barplot(x=hour_stats.index, y=hour_stats['sum'], palette='vlag')
    plt.title('Total PnL by Hour (Feb 2026)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Total PnL (pips)')
    
    # 曜日別損益
    plt.subplot(2, 2, 2)
    sns.barplot(x=day_stats.index, y=day_stats['sum'], palette='vlag')
    plt.title('Total PnL by Day of Week (0=Mon)')
    plt.xlabel('Day of Week')
    plt.ylabel('Total PnL (pips)')
    
    # 損益の分布
    plt.subplot(2, 2, 3)
    sns.histplot(trades_df['pnl_pips'], bins=30, kde=True, color='purple')
    plt.title('Distribution of Trade PnL (pips)')
    plt.axvline(0, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("/home/ubuntu/sena3fx/results/v60_factor_analysis.png")
    print("\nAnalysis graph saved to: /home/ubuntu/sena3fx/results/v60_factor_analysis.png")

if __name__ == "__main__":
    analyze_factors()
