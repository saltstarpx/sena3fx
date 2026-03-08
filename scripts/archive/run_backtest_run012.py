#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v4 import generate_signals

def main():
    # データ読み込み
    data_path = os.path.join(BASE_DIR, 'data', 'ohlc', 'USDJPY_1m_2026_Jan.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print("=" * 80)
    print("RUN-012: やがみ式3層MTF戦略 v4 - スプレッド1.0pips負荷テスト")
    print("=" * 80)
    print(f"データ期間: {df.index[0]} 〜 {df.index[-1]}")
    print(f"1分足バー数: {len(df)}")
    
    # 15分足・4時間足へのリサンプリング
    df_15m = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    df_4h = df.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    print(f"15分足バー数: {len(df_15m)}")
    print(f"4時間足バー数: {len(df_4h)}")
    
    # シグナル生成
    print("\nシグナル生成中...")
    signal_series, signals_detail = generate_signals(df, df_15m, df_4h, spread=0.01)
    
    print(f"生成されたシグナル: {len(signals_detail)} 件")
    
    # バックテスト実行
    print("\n" + "=" * 80)
    print("バックテスト実行中...")
    print("=" * 80)
    
    engine = BacktestEngine(
        init_cash=1000000,
        risk_pct=0.02,
        slippage_pips=1.0  # 1.0 pips
    )
    
    def signal_func(data):
        """シグナル関数"""
        return signal_series.loc[:data.index[-1]].iloc[-1]
    
    results = engine.run(
        data=df,
        signal_func=signal_func,
        freq='1min',
        name="Yagami_v4_Spread_Test"
    )
    
    # 結果表示
    print("\n" + "=" * 80)
    print("【RUN-012】スプレッド1.0pips負荷テスト結果")
    print("=" * 80)
    
    if results and 'pf' in results:
        print(f"プロフィットファクター (PF): {results['pf']:.4f}")
        print(f"勝率: {results['win_rate']:.2f}%")
        print(f"総取引数: {results['total_trades']}")
        print(f"最終利益: {results['final_pnl']:.2f}")
        print(f"最大ドローダウン: {results['max_dd']:.2f}%")
    else:
        print("バックテスト結果が取得できません。")
        print(f"Results: {results}")
    
    # トレード詳細をCSVに保存
    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_path = os.path.join(BASE_DIR, 'results', 'run012_trades.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"\n✓ トレード詳細を保存しました: {trades_path}")

if __name__ == '__main__':
    main()
