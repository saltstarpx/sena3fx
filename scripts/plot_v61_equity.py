import pandas as pd
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def plot_v61_equity():
    # v61の読み込み
    v61_df = pd.read_csv("/home/ubuntu/sena3fx/results/trades_v61.csv")
    v61_df['exit_time'] = pd.to_datetime(v61_df['exit_time'])
    v61_equity = v61_df.sort_values('exit_time')['pnl_pips'].cumsum()
    
    # v60 (0.5x) の読み込み (以前作成したファイルがあれば)
    try:
        v60_df = pd.read_csv("/home/ubuntu/sena3fx/results/trades_v60_05x.csv")
        v60_df['exit_time'] = pd.to_datetime(v60_df['exit_time'])
        v60_equity = v60_df.sort_values('exit_time')['pnl_pips'].cumsum()
    except:
        v60_equity = None

    plt.figure(figsize=(12, 6))
    plt.plot(v61_equity.values, label='v61 (No Time Decay, N-Value TP, Partial TP)', color='blue')
    if v60_equity is not None:
        plt.plot(v60_equity.values, label='v60 (0.5x Volatility Filter)', color='red', linestyle='--')
    
    plt.title('Yagami MTF v61: 累積損益 (pips) 推移')
    plt.xlabel('トレード件数')
    plt.ylabel('累積損益 (pips)')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/sena3fx/results/v61_equity_curve.png")
    print("Equity curve saved to v61_equity_curve.png")

if __name__ == "__main__":
    plot_v61_equity()
