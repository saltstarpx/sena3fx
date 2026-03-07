import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 日本語フォント設定（Noto Sans CJK JP）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def perform_quant_analysis(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(file_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # 1. 基本統計量
    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    win_rate = len(wins) / total_trades * 100
    
    avg_win = wins['pnl'].mean()
    avg_loss = abs(losses['pnl'].mean())
    rr_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Win: {avg_win:.5f}")
    print(f"Avg Loss: {avg_loss:.5f}")
    print(f"Realized RR: {rr_ratio:.2f}")

    # 2. 時間帯別分析 (Entry Hour)
    df['hour'] = df['entry_time'].dt.hour
    hour_analysis = df.groupby('hour')['pnl'].agg(['count', 'sum', 'mean']).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=hour_analysis, x='hour', y='sum', palette='viridis')
    plt.title('時間帯別 合計損益 (Entry Hour)')
    plt.xlabel('Hour (UTC)')
    plt.ylabel('Total PnL')
    plt.savefig(f"{output_dir}/pnl_by_hour.png")
    plt.close()

    # 3. 累積損益 (Equity Curve)
    df['cum_pnl'] = df['pnl'].cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(df['entry_time'], df['cum_pnl'])
    plt.title('累積損益曲線 (2月)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/equity_curve.png")
    plt.close()

    # 4. ドローダウン分析
    df['max_cum_pnl'] = df['cum_pnl'].cummax()
    df['drawdown'] = df['cum_pnl'] - df['max_cum_pnl']
    max_dd = df['drawdown'].min()
    print(f"Max Drawdown: {max_dd:.5f}")
    
    plt.figure(figsize=(12, 6))
    plt.fill_between(df['entry_time'], df['drawdown'], 0, color='red', alpha=0.3)
    plt.title('ドローダウン推移')
    plt.xlabel('Time')
    plt.ylabel('Drawdown')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/drawdown.png")
    plt.close()

    # 5. 保有時間分析
    df['duration_min'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_min'], bins=30, kde=True)
    plt.title('トレード保有時間分布 (分)')
    plt.xlabel('Duration (minutes)')
    plt.savefig(f"{output_dir}/duration_dist.png")
    plt.close()
    
    # 損益と保有時間の相関
    plt.figure(figsize=(10, 6))
    plt.scatter(df['duration_min'], df['pnl'], alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('保有時間と損益の相関')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('PnL')
    plt.savefig(f"{output_dir}/duration_vs_pnl.png")
    plt.close()

    # レポート作成
    report = f"""# 定量分析レポート (2月 v58)

## 基本統計
- 総取引数: {total_trades}
- 勝率: {win_rate:.2f}%
- 平均利益: {avg_win:.5f}
- 平均損失: {avg_loss:.5f}
- 実現リスクリワード比: {rr_ratio:.2f}
- 最大ドローダウン: {max_dd:.5f}

## 考察
- **時間帯:** 利益が出やすい時間帯と損失が出やすい時間帯が明確に分かれているか確認してください。
- **保有時間:** 利益トレードと損失トレードで保有時間に有意な差があるか。
- **ドローダウン:** 連続損失が発生している期間の相場環境（ボラティリティの低下など）を再確認する必要があります。
"""
    with open(f"{output_dir}/quant_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    file_feb = "/home/ubuntu/sena3fx/results/run012v58_USDJPY_1m_2026_Feb.csv_trades.csv"
    output_dir = "/home/ubuntu/sena3fx/results/quant_analysis_feb_v58"
    perform_quant_analysis(file_feb, output_dir)
