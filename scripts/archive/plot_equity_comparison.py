import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # データ読み込み
    df_05 = pd.read_csv("/home/ubuntu/sena3fx/results/trades_v60_05x.csv")
    df_00 = pd.read_csv("/home/ubuntu/sena3fx/results/trades_v60_00x.csv")
    
    # 時間でソート
    df_05['exit_time'] = pd.to_datetime(df_05['exit_time'])
    df_00['exit_time'] = pd.to_datetime(df_00['exit_time'])
    df_05 = df_05.sort_values('exit_time')
    df_00 = df_00.sort_values('exit_time')
    
    # 累積損益の計算 (pips換算)
    df_05['cumulative_pnl'] = (df_05['pnl'] / 0.01).cumsum()
    df_00['cumulative_pnl'] = (df_00['pnl'] / 0.01).cumsum()
    
    # グラフ作成
    plt.figure(figsize=(12, 6))
    plt.plot(df_05['exit_time'], df_05['cumulative_pnl'], label='Volatility Filter 0.5x (Relaxed)', color='#3498DB', linewidth=2)
    plt.plot(df_00['exit_time'], df_00['cumulative_pnl'], label='Volatility Filter 0.0x (Disabled)', color='#E74C3C', linestyle='--', alpha=0.7)
    
    plt.title('Equity Curve Comparison: Volatility Filter Effect (Feb 2026)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative PnL (pips)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 保存
    save_path = "/home/ubuntu/sena3fx/results/v60_equity_comparison.png"
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    plot_comparison()
