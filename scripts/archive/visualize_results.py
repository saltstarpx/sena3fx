import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_backtest(trades_file, output_prefix):
    if not os.path.exists(trades_file):
        print(f"File not found: {trades_file}")
        return

    df = pd.read_csv(trades_file)
    if df.empty:
        print(f"No trades in {trades_file}")
        return

    # 累積損益の計算
    df['Cumulative_PnL'] = df['PnL_pips'].cumsum()
    
    # プロット作成
    plt.figure(figsize=(12, 8))
    
    # 1. 損益曲線
    plt.subplot(2, 1, 1)
    plt.plot(df['Cumulative_PnL'], marker='o')
    plt.title(f'Cumulative PnL - {output_prefix}')
    plt.ylabel('Pips')
    plt.grid(True)

    # 2. トレードごとの損益
    plt.subplot(2, 1, 2)
    colors = ['green' if x > 0 else 'red' for x in df['PnL_pips']]
    plt.bar(range(len(df)), df['PnL_pips'], color=colors)
    plt.title('Individual Trade PnL')
    plt.ylabel('Pips')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/sena3fx/results/{output_prefix}_equity_curve.png')
    print(f"Visualization saved to /home/ubuntu/sena3fx/results/{output_prefix}_equity_curve.png")

if __name__ == "__main__":
    visualize_backtest('/home/ubuntu/sena3fx/results/trades_v67.csv', 'v67')
    visualize_backtest('/home/ubuntu/sena3fx/results/trades_v66.csv', 'v66')
    visualize_backtest('/home/ubuntu/sena3fx/results/trades_v65.csv', 'v65')
    visualize_backtest('/home/ubuntu/sena3fx/results/trades_v64.csv', 'v64')
    visualize_backtest('/home/ubuntu/sena3fx/results/trades_v63.csv', 'v63')
    visualize_backtest('/home/ubuntu/sena3fx/results/trades_v62.csv', 'v62')
