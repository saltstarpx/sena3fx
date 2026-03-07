#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v7 - スプレッド克服・高頻度版

改善点：
1. エントリー条件の厳格化: 単なる放れではなく、ボラティリティの拡大（勢い）を伴う放れを検知
2. 15分足パターンの強化: 実体と髭の比率に加え、4Hトレンドとの一致を重視
3. スプレッド(1.0pips)を上回るための「最低値幅」の考慮
"""

import pandas as pd
import numpy as np

def calculate_atr(data, period=14):
    data = data.copy()
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    return data['tr'].rolling(period).mean()

def analyze_h4_environment(h4_data):
    h4_data = h4_data.copy()
    h4_data['atr'] = calculate_atr(h4_data)
    h4_data['trend'] = 0
    # 4H EMA20 との関係でトレンド判定
    h4_data['ema20'] = h4_data['close'].ewm(span=20).mean()
    h4_data.loc[h4_data['close'] > h4_data['ema20'], 'trend'] = 1
    h4_data.loc[h4_data['close'] < h4_data['ema20'], 'trend'] = -1
    return h4_data

def analyze_m15_pattern(m15_data, h4_trend, h4_atr):
    m15_data = m15_data.copy()
    m15_data['body_size'] = abs(m15_data['close'] - m15_data['open'])
    m15_data['lower_wick'] = m15_data[['open', 'close']].min(axis=1) - m15_data['low']
    m15_data['upper_wick'] = m15_data['high'] - m15_data[['open', 'close']].max(axis=1)
    
    m15_data['signal'] = 0
    # 4Hトレンド方向への反転パターンのみに絞る
    # ロング: 下髭が実体の1.5倍以上、かつ4Hが上昇トレンドまたは中立
    m15_data.loc[(m15_data['lower_wick'] > m15_data['body_size'] * 1.5) & (h4_trend >= 0), 'signal'] = 1
    # ショート: 上髭が実体の1.5倍以上、かつ4Hが下降トレンドまたは中立
    m15_data.loc[(m15_data['upper_wick'] > m15_data['body_size'] * 1.5) & (h4_trend <= 0), 'signal'] = -1
    
    return m15_data

def analyze_m1_execution(m1_data, m15_signal, spread=0.01):
    m1_data = m1_data.copy()
    signals = []
    
    # 1分足の勢い（直近ATRとの比較）
    m1_data['atr'] = calculate_atr(m1_data)
    
    for i in range(5, len(m1_data)):
        curr_bar = m1_data.iloc[i]
        recent_m1 = m1_data.iloc[i-5:i]
        
        if m15_signal > 0: # ロング
            # 勢いのある放れ（現在の足のサイズが直近ATRの2倍以上）
            if (curr_bar['close'] > recent_m1['high'].max()) and (curr_bar['close'] - curr_bar['open'] > curr_bar['atr'] * 1.5):
                sl_price = recent_m1['low'].min() - spread
                entry_price = curr_bar['close']
                if entry_price - sl_price > spread * 1.5:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'LONG',
                        'entry': entry_price,
                        'stop_loss': sl_price
                    })
                    break
                    
        elif m15_signal < 0: # ショート
            if (curr_bar['close'] < recent_m1['low'].min()) and (curr_bar['open'] - curr_bar['close'] > curr_bar['atr'] * 1.5):
                sl_price = recent_m1['high'].max() + spread
                entry_price = curr_bar['close']
                if sl_price - entry_price > spread * 1.5:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'SHORT',
                        'entry': entry_price,
                        'stop_loss': sl_price
                    })
                    break
                    
    return signals

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    signals_list = []
    
    for i in range(len(data_15m)):
        h4_idx = data_4h.index.get_indexer([data_15m.index[i]], method='ffill')[0]
        if h4_idx < 0: continue
        
        h4_trend = h4_env.iloc[h4_idx]['trend']
        h4_atr = h4_env.iloc[h4_idx]['atr']
        
        m15_start_time = data_15m.index[i]
        m15_end_time = m15_start_time + pd.Timedelta(minutes=15)
        m1_slice = data_1m.loc[m15_start_time:m15_end_time]
        
        if len(m1_slice) < 5: continue
        
        m15_pattern = analyze_m15_pattern(data_15m.iloc[i:i+1], h4_trend, h4_atr)
        m15_signal = m15_pattern['signal'].iloc[0]
        
        if m15_signal != 0:
            m1_exec_signals = analyze_m1_execution(m1_slice, m15_signal, spread)
            signals_list.extend(m1_exec_signals)
            
    signal_series = pd.Series(0, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig['direction'] == 'LONG' else -1
        signal_series.loc[sig['time']] = val
        
    return signal_series, signals_list
