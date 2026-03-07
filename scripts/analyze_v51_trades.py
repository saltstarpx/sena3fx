import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_trades(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No trades to analyze.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 基本統計
    total_trades = len(df)
    win_trades = len(df[df['pnl'] > 0])
    loss_trades = len(df[df['pnl'] <= 0])
    win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    
    # 損切りまでの時間
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['duration_min'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60

    report = f"""
# RUN-012v51 取引分析レポート

## 基本統計
- 総取引数: {total_trades}
- 勝ちトレード数: {win_trades}
- 負けトレード数: {loss_trades}
- 勝率: {win_rate:.2f}%
- 総損益: {total_pnl:.2f}
- 平均損益: {avg_pnl:.2f}

## 保持時間の分析
- 平均保持時間: {df['duration_min'].mean():.2f} 分
- 最大保持時間: {df['duration_min'].max():.2f} 分
- 最小保持時間: {df['duration_min'].min():.2f} 分

## 考察
勝率が {win_rate:.2f}% と極めて低いです。
平均保持時間が {df['duration_min'].mean():.2f} 分であることから、エントリー直後に逆行して損切りされている可能性が高いです。
1.0pipsのスプレッド環境下で、15分足の髭先付近でのエントリー精度が不足しているか、
あるいは「背」としての損切り位置が近すぎることが原因と考えられます。
"""

    with open(os.path.join(output_dir, "analysis_report_v51.md"), "w") as f:
        f.write(report)

    # グラフ作成
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pnl'], bins=50, kde=True)
    plt.title('PnL Distribution (v51)')
    plt.savefig(os.path.join(output_dir, "pnl_dist_v51.png"))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_min'], bins=50, kde=True)
    plt.title('Trade Duration Distribution (min) (v51)')
    plt.savefig(os.path.join(output_dir, "duration_dist_v51.png"))

    print(f"Analysis completed. Report saved to {output_dir}")

if __name__ == "__main__":
    analyze_trades("/home/ubuntu/sena3fx/results/run012v51_trades.csv", "/home/ubuntu/sena3fx/results/analysis_v51")
