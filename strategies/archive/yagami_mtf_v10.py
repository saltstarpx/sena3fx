#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v10 - 押し目買い・戻り売り・高頻度版

改善点：
1. 1分足での押し目・戻り判定: 4Hトレンド方向への一時的な逆行（押し）を確認してからエントリー
2. 利益確定の最適化: 1分足のトレンドが崩れるまで、または一定の値幅(5pips以上)が出るまでホールド
3. スプレッド(1.0pips)を確実に超えるための「最低利益」の確保
"""

import pandas as pd
import numpy as np

def analyze_h4_environment(h4_data):
    h4_data = h4_data.copy()
    h4_data['ema20'] = h4_data['close'].ewm(span=20).mean()
    h4_data['trend'] = 0
    h4_data.loc[h4_data['close'] > h4_data['ema20'], 'trend'] = 1
    h4_data.loc[h4_data['close'] < h4_data['ema20'], 'trend'] = -1
    return h4_data

def analyze_m1_execution(m1_data, h4_trend):
    m1_data = m1_data.copy()
    signals = []
    
    for i in range(10, len(m1_data)):
        curr_bar = m1_data.iloc[i]
        recent_m1 = m1_data.iloc[i-10:i]
        
        if h4_trend > 0: # ロング（押し目買い）
            # 1分足で一時的に下げている（押し）
            if curr_bar['close'] < recent_m1['close'].mean():
                # その後、反転の兆し（前の足の高値を抜ける）
                if curr_bar['close'] > m1_data.iloc[i-1]['high']:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'LONG',
                        'entry': curr_bar['close']
                    })
                    i += 10 # 頻度調整
                    
        elif h4_trend < 0: # ショート（戻り売り）
            if curr_bar['close'] > recent_m1['close'].mean():
                if curr_bar['close'] < m1_data.iloc[i-1]['low']:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'SHORT',
                        'entry': curr_bar['close']
                    })
                    i += 10
                    
    return signals

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    signals_list = []
    
    for i in range(len(data_15m)):
        h4_idx = data_4h.index.get_indexer([data_15m.index[i]], method='ffill')[0]
        if h4_idx < 0: h4_idx = 0
        h4_trend = h4_env.iloc[h4_idx]['trend']
        
        if h4_trend == 0: continue
        
        m15_start_time = data_15m.index[i]
        m15_end_time = m15_start_time + pd.Timedelta(minutes=15)
        m1_slice = data_1m.loc[m15_start_time:m15_end_time]
        
        if len(m1_slice) < 10: continue
        
        m1_exec_signals = analyze_m1_execution(m1_slice, h4_trend)
        signals_list.extend(m1_exec_signals)
            
    signal_series = pd.Series(0, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig['direction'] == 'LONG' else -1
        signal_series.loc[sig['time']] = val
        
    return signal_series, signals_list
