#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from lib.backtest import BacktestEngine
from strategies.yagami_mtf_v3 import signal_yagami_mtf_v3

def resample_ohlc(df, freq):
    if df.index.name != 'timestamp':
        df = df.reset_index()
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns: agg_dict['volume'] = 'sum'
    elif 'tick_count' in df.columns: agg_dict['tick_count'] = 'sum'
    resampled = df.resample(freq).agg(agg_dict)
    return resampled.dropna(subset=['open'])

def main():
    print("RUN-010: やがみ式3層MTF戦略 (SL/TP組み込み版)")
    m1_path = os.path.join(BASE_DIR, 'data', 'ohlc', 'USDJPY_1m.csv')
    m1_df = pd.read_csv(m1_path, index_col=0, parse_dates=True)
    m1_df.index.name = 'timestamp'
    
    m15_df = resample_ohlc(m1_df, '15min')
    h4_df = resample_ohlc(m1_df, '4h')
    
    print("シグナル生成中...")
    all_signals = pd.Series(index=m1_df.index, dtype=object)
    sl_map = {}
    tp_map = {}
    
    for i in range(100, len(m1_df)):
        current_time = m1_df.index[i]
        h4_slice = h4_df[h4_df.index <= current_time].tail(20)
        m15_slice = m15_df[m15_df.index <= current_time].tail(20)
        m1_slice = m1_df[m1_df.index <= current_time].tail(20)
        
        if len(h4_slice) < 5 or len(m15_slice) < 5 or len(m1_slice) < 5: continue
        
        signal = signal_yagami_mtf_v3(h4_slice, m15_slice, m1_slice)
        if signal.get('signal'):
            all_signals.iloc[i] = signal['signal']
            sl_map[current_time] = signal['stop_loss']
            tp_map[current_time] = signal['take_profit']
    
    print(f"生成されたシグナル: {all_signals.count()} 件")
    if all_signals.count() == 0: return

    # Safety Valveを無効化するために極小値を設定
    engine = BacktestEngine(
        init_cash=1000000, 
        risk_pct=0.02,
        min_sl_atr_mult=0.0,
        min_sl_price_pct=0.0
    )
    
    engine._find_swing_low = lambda bars, idx, htf_bars=None, n_confirm=2: sl_map.get(bars.index[idx], bars['close'].iloc[idx] * 0.99)
    engine._find_swing_high = lambda bars, idx, htf_bars=None, n_confirm=2: sl_map.get(bars.index[idx], bars['close'].iloc[idx] * 1.01)
    engine.use_dynamic_sl = True

    def signal_func(bars):
        return all_signals

    results = engine.run(data=m1_df, signal_func=signal_func, name="Yagami_MTF_v3_RUN010", freq="1m")
    
    if results:
        print(f"\n--- バックテスト結果 ---")
        print(f"プロフィットファクター (PF): {results.get('profit_factor', 'N/A')}")
        print(f"勝率: {results.get('win_rate_pct', 'N/A')}%")
        print(f"総取引数: {results.get('total_trades', 'N/A')}")
        print(f"最終利益: {results.get('total_pnl', 'N/A')}")
        print(f"最大ドローダウン: {results.get('max_drawdown_pct', 'N/A')}%")
        
        if results.get('trades'):
            trades_path = os.path.join(BASE_DIR, 'results', 'run010_trades.csv')
            pd.DataFrame(results['trades']).to_csv(trades_path, index=False)
            print(f"トレード詳細を保存: {trades_path}")
    else:
        print("バックテスト結果が空です。")

if __name__ == '__main__':
    main()
