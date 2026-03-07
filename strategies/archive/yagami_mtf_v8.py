#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v8 - 高頻度・実戦的適応版

改善点：
1. 15分足判定の超緩和: 4Hトレンド方向への「少しの髭」でもシグナル化し、頻度を確保
2. 1分足エントリーの単純化: 15分足のシグナル後、1分足で直近3本の高安を実体で抜ければ即エントリー
3. ホールド時間の延長: 1分足のノイズを無視し、15分〜60分単位の大きな波を狙う
"""

import pandas as pd
import numpy as np

def analyze_h4_environment(h4_data):
    h4_data = h4_data.copy()
    h4_data['trend'] = 0
    h4_data['ema20'] = h4_data['close'].ewm(span=20).mean()
    h4_data.loc[h4_data['close'] > h4_data['ema20'], 'trend'] = 1
    h4_data.loc[h4_data['close'] < h4_data['ema20'], 'trend'] = -1
    return h4_data

def analyze_m15_pattern(m15_data, h4_trend):
    m15_data = m15_data.copy()
    m15_data['lower_wick'] = m15_data[['open', 'close']].min(axis=1) - m15_data['low']
    m15_data['upper_wick'] = m15_data['high'] - m15_data[['open', 'close']].max(axis=1)
    
    m15_data['signal'] = 0
    # 4Hトレンド方向への「少しでも髭があれば」シグナル（超緩和）
    m15_data.loc[(m15_data['lower_wick'] > 0.001) & (h4_trend >= 0), 'signal'] = 1
    m15_data.loc[(m15_data['upper_wick'] > 0.001) & (h4_trend <= 0), 'signal'] = -1
    
    return m15_data

def analyze_m1_execution(m1_data, m15_signal, spread=0.01):
    m1_data = m1_data.copy()
    signals = []
    
    for i in range(3, len(m1_data)):
        curr_bar = m1_data.iloc[i]
        recent_m1 = m1_data.iloc[i-3:i]
        
        if m15_signal > 0: # ロング
            # 1分足で直近3本の高値を抜ける
            if curr_bar['close'] > recent_m1['high'].max():
                signals.append({
                    'time': m1_data.index[i],
                    'direction': 'LONG',
                    'entry': curr_bar['close'],
                    'stop_loss': recent_m1['low'].min() - spread
                })
                break # 1つの15分足につき1回
                    
        elif m15_signal < 0: # ショート
            if curr_bar['close'] < recent_m1['low'].min():
                signals.append({
                    'time': m1_data.index[i],
                    'direction': 'SHORT',
                    'entry': curr_bar['close'],
                    'stop_loss': recent_m1['high'].max() + spread
                })
                break
                    
    return signals

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    signals_list = []
    
    for i in range(len(data_15m)):
        # 4H環境の取得
        h4_idx = data_4h.index.get_indexer([data_15m.index[i]], method='ffill')[0]
        if h4_idx < 0: h4_idx = 0
        h4_trend = h4_env.iloc[h4_idx]['trend']
        
        m15_start_time = data_15m.index[i]
        m15_end_time = m15_start_time + pd.Timedelta(minutes=15)
        m1_slice = data_1m.loc[m15_start_time:m15_end_time]
        
        if len(m1_slice) < 3: continue
        
        m15_pattern = analyze_m15_pattern(data_15m.iloc[i:i+1], h4_trend)
        m15_signal = m15_pattern['signal'].iloc[0]
        
        if m15_signal != 0:
            m1_exec_signals = analyze_m1_execution(m1_slice, m15_signal, spread)
            signals_list.extend(m1_exec_signals)
            
    signal_series = pd.Series(0, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig['direction'] == 'LONG' else -1
        signal_series.loc[sig['time']] = val
        
    return signal_series, signals_list
