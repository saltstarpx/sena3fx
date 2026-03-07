#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v11 - 強気ブレイクアウト・高頻度版

改善点：
1. 1分足での強気ブレイク判定: 直近20本の最高値/最安値を「強い勢い（大陽線/大陰線）」で抜けた瞬間にエントリー
2. 利益確定の最大化: 4時間ホールドを維持し、トレンドの初動から終動までを狙う
3. スプレッド(1.0pips)を瞬時に克服するための「モメンタム」の利用
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
    
    for i in range(20, len(m1_data)):
        curr_bar = m1_data.iloc[i]
        recent_m1 = m1_data.iloc[i-20:i]
        
        # 勢いの判定（現在の足のサイズが直近平均の3倍以上）
        avg_body = abs(recent_m1['close'] - recent_m1['open']).mean()
        curr_body = abs(curr_bar['close'] - curr_bar['open'])
        
        if h4_trend > 0: # ロング（ブレイクアウト）
            if curr_bar['close'] > recent_m1['high'].max() and curr_body > avg_body * 3:
                signals.append({
                    'time': m1_data.index[i],
                    'direction': 'LONG',
                    'entry': curr_bar['close']
                })
                i += 20 # 頻度調整
                    
        elif h4_trend < 0: # ショート（ブレイクアウト）
            if curr_bar['close'] < recent_m1['low'].min() and curr_body > avg_body * 3:
                signals.append({
                    'time': m1_data.index[i],
                    'direction': 'SHORT',
                    'entry': curr_bar['close']
                })
                i += 20
                    
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
        
        if len(m1_slice) < 20: continue
        
        m1_exec_signals = analyze_m1_execution(m1_slice, h4_trend)
        signals_list.extend(m1_exec_signals)
            
    signal_series = pd.Series(0, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig['direction'] == 'LONG' else -1
        signal_series.loc[sig['time']] = val
        
    return signal_series, signals_list
