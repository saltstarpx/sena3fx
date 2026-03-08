#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v6 import generate_signals

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
    print("RUN-012v6: やがみ式3層MTF戦略 v6 - 高頻度・スプレッド耐性再テスト")
    print("=" * 80)
    print(f"データ期間: {df.index[0]} 〜 {df.index[-1]}")
    
    # 15分足・4時間足へのリサンプリング
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
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
        slippage_pips=1.0  # 1.0 pipsのハンデ
    )
    
    def signal_func(data):
        """シグナル関数"""
        # Seriesから直接取得（Series.indexはDatetimeIndex）
        try:
            return signal_series.loc[data.index[-1]]
        except KeyError:
            return 0
    
    results = engine.run(
        data=df,
        signal_func=signal_func,
        freq='1min',
        name="Yagami_v6_Stress_Test"
    )
    
    # 結果表示
    print("\n" + "=" * 80)
    print("【RUN-012v6】高頻度・スプレッド耐性テスト結果")
    print("=" * 80)
    
    if results and 'pf' in results:
        print(f"プロフィットファクター (PF): {results['pf']:.4f}")
        print(f"勝率: {results['win_rate']:.2f}%")
        print(f"総取引数: {results['total_trades']}")
        print(f"最終利益: {results['final_pnl']:.2f}")
        print(f"最大ドローダウン: {results['max_dd']:.2f}%")
    else:
        print("バックテスト結果が取得できません。")
    
    # トレード詳細をCSVに保存
    if 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_path = os.path.join(BASE_DIR, 'results', 'run012v6_trades.csv')
        trades_df.to_csv(trades_path, index=False)
        print(f"\n✓ トレード詳細を保存しました: {trades_path}")

if __name__ == '__main__':
    main()
