#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定（Noto Sans CJK JP）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def analyze_trades(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['hour'] = df['entry_time'].dt.hour
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 時間帯別損益
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='hour', y='pnl', estimator=sum, errorbar=None)
    plt.title('時間帯別合計損益 (1月データ)')
    plt.xlabel('時間 (UTC)')
    plt.ylabel('合計損益 (pips * 10000)')
    plt.savefig(os.path.join(output_dir, 'pnl_by_hour.png'))
    
    # 2. 時間帯別取引回数
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='hour')
    plt.title('時間帯別取引回数')
    plt.xlabel('時間 (UTC)')
    plt.ylabel('取引回数')
    plt.savefig(os.path.join(output_dir, 'trades_by_hour.png'))
    
    # 3. 勝敗比率
    df['result'] = df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
    plt.figure(figsize=(8, 8))
    df['result'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'green'])
    plt.title('勝敗比率 (スプレッド1.0pips負荷)')
    plt.savefig(os.path.join(output_dir, 'win_loss_ratio.pie.png'))
    
    # 4. 損益分布
    plt.figure(figsize=(12, 6))
    sns.histplot(df['pnl'], bins=50, kde=True)
    plt.title('損益分布')
    plt.xlabel('損益')
    plt.ylabel('頻度')
    plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'))
    
    # 定量的レポートの作成
    report = []
    report.append("# 取引履歴定量分析レポート")
    report.append(f"分析対象: {csv_path}")
    report.append(f"総取引数: {len(df)}")
    report.append(f"勝率: {(df['pnl'] > 0).mean()*100:.2f}%")
    report.append(f"平均損益: {df['pnl'].mean():.2f}")
    report.append(f"最大利益: {df['pnl'].max():.2f}")
    report.append(f"最大損失: {df['pnl'].min():.2f}")
    
    # 時間帯別の詳細
    hourly_stats = df.groupby('hour')['pnl'].agg(['count', 'mean', 'sum'])
    report.append("\n## 時間帯別統計 (UTC)")
    report.append(hourly_stats.to_markdown())
    
    with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"分析完了。結果は {output_dir} に保存されました。")

if __name__ == '__main__':
    csv_path = '/home/ubuntu/sena3fx/results/run012v23_trades.csv'
    output_dir = '/home/ubuntu/sena3fx/results/analysis_v23'
    analyze_trades(csv_path, output_dir)
