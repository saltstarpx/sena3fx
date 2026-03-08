#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def main():
    trades_path = os.path.join(BASE_DIR, 'results', 'run011_trades_with_spread.csv')
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found.")
        return

    trades_df = pd.read_csv(trades_path)
    if trades_df.empty:
        print("No trades to analyze.")
        return

    print("=" * 80)
    print("RUN-011 (2026年1月) 負けパターン詳細分析")
    print("=" * 80)

    # 1. 損益分布
    print(f"\n総トレード数: {len(trades_df)}")
    print(f"勝ちトレード: {len(trades_df[trades_df['pnl'] > 0])}")
    print(f"負けトレード: {len(trades_df[trades_df['pnl'] <= 0])}")

    # 2. ホールド時間の分析
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['duration_min'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60

    print(f"\n平均ホールド時間: {trades_df['duration_min'].mean():.2f} 分")
    print(f"負けトレードの平均ホールド時間: {trades_df[trades_df['pnl'] <= 0]['duration_min'].mean():.2f} 分")
    
    # 短時間で切られているトレードの割合
    short_loss = trades_df[(trades_df['pnl'] <= 0) & (trades_df['duration_min'] < 5)]
    print(f"5分以内に損切りされたトレード: {len(short_loss)} 件 ({len(short_loss)/len(trades_df)*100:.1f}%)")

    # 3. エントリー価格とSLの関係 (スリッページ・スプレッド耐性)
    # entry_price と exit_price の差がスプレッド(1.0pips=0.01)に近いものを探す
    trades_df['price_diff'] = abs(trades_df['exit'] - trades_df['entry'])
    near_spread_loss = trades_df[(trades_df['pnl'] <= 0) & (trades_df['price_diff'] <= 0.02)]
    print(f"スプレッド(1.0pips)圏内で損切りされたトレード: {len(near_spread_loss)} 件 ({len(near_spread_loss)/len(trades_df)*100:.1f}%)")

    # 4. 時間帯別の分析
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    hourly_pnl = trades_df.groupby('hour')['pnl'].sum()
    print("\n時間帯別損益 (Top 5 損失時間帯):")
    print(hourly_pnl.sort_values().head(5))

    # 分析結果をMarkdownに保存
    report_path = os.path.join(BASE_DIR, 'results', 'run011_failure_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write("# RUN-011 (2026年1月) 負けパターン定量分析レポート\n\n")
        f.write(f"## 1. 概要\n- 総トレード数: {len(trades_df)}\n- 勝率: {len(trades_df[trades_df['pnl'] > 0])/len(trades_df)*100:.2f}%\n- 平均損失: {trades_df[trades_df['pnl'] <= 0]['pnl'].mean():.2f}\n\n")
        f.write("## 2. 核心的な課題\n")
        f.write(f"### ① 極端に短いホールド時間\n- 負けトレードの平均ホールド時間は **{trades_df[trades_df['pnl'] <= 0]['duration_min'].mean():.2f}分** です。\n")
        f.write(f"- 特に5分以内に損切りされたトレードが **{len(short_loss)}件** あり、エントリー直後に「ノイズ」で切られている可能性が極めて高いです。\n\n")
        f.write("### ② スプレッド耐性の欠如\n")
        f.write(f"- スプレッド(1.0pips)程度の値動きで損切りされているトレードが **{len(near_spread_loss)}件** あります。\n")
        f.write("- これは「背」が近すぎて、実戦環境（スプレッドあり）では息ができていない状態です。\n\n")
        f.write("### ③ 期待値の低い時間帯での乱発\n")
        f.write("- 特定の時間帯に損失が集中しており、相場環境（4H）のフィルターが甘い可能性があります。\n")

    print(f"\n✓ 分析レポートを保存しました: {report_path}")

if __name__ == '__main__':
    main()
