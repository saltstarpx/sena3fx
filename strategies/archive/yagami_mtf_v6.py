#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v6 - 高頻度・スプレッド耐性版

改善点：
1. 15分足の髭判定を「相対的な長さ」に変更（ATRや実体との比率）
2. エントリー条件を「1分足の放れ」のみに簡略化し、頻度を確保
3. スプレッド耐性を維持しつつ、判定を緩和
"""

import pandas as pd
import numpy as np

def calculate_atr(data, period=14):
    """ATR計算"""
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
    """4時間足での環境認識"""
    h4_data = h4_data.copy()
    h4_data['atr'] = calculate_atr(h4_data)
    
    # トレンド判定（極めて緩やか）
    h4_data['trend'] = 0
    h4_data.loc[h4_data['close'] >= h4_data['close'].shift(1), 'trend'] = 1
    h4_data.loc[h4_data['close'] < h4_data['close'].shift(1), 'trend'] = -1
    
    return h4_data

def analyze_m15_pattern(m15_data, h4_trend, h4_atr):
    """15分足でのパターン認識（相対判定）"""
    m15_data = m15_data.copy()
    
    # 実体サイズと髭の長さを比較
    m15_data['body_size'] = abs(m15_data['close'] - m15_data['open'])
    m15_data['lower_wick'] = m15_data[['open', 'close']].min(axis=1) - m15_data['low']
    m15_data['upper_wick'] = m15_data['high'] - m15_data[['open', 'close']].max(axis=1)
    
    m15_data['signal'] = 0
    
    # 下髭が実体と同等以上、またはATRの一定割合以上ならシグナル
    # ロング
    m15_data.loc[(m15_data['lower_wick'] > m15_data['body_size'] * 0.5) | (m15_data['lower_wick'] > 0.001), 'signal'] = 1
    # ショート
    m15_data.loc[(m15_data['upper_wick'] > m15_data['body_size'] * 0.5) | (m15_data['upper_wick'] > 0.001), 'signal'] = -1
    
    return m15_data

def analyze_m1_execution(m1_data, m15_signal, spread=0.01):
    """1分足での精密な執行（頻度重視）"""
    m1_data = m1_data.copy()
    signals = []
    
    for i in range(3, len(m1_data)):
        if m15_signal == 0:
            continue
            
        curr_bar = m1_data.iloc[i]
        recent_m1 = m1_data.iloc[i-3:i]
        
        if m15_signal > 0: # ロング検討
            # 前の足の高値を抜ける
            if curr_bar['close'] > recent_m1['high'].max():
                # SLは直近安値からスプレッド分(1.0pips)離す
                sl_price = recent_m1['low'].min() - spread
                entry_price = curr_bar['close']
                
                # SLが極端に近くないかだけ確認
                if entry_price - sl_price > spread * 0.5:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'LONG',
                        'entry': entry_price,
                        'stop_loss': sl_price
                    })
                    break # 1つの15分足につき1エントリーまで
                    
        elif m15_signal < 0: # ショート検討
            if curr_bar['close'] < recent_m1['low'].min():
                sl_price = recent_m1['high'].max() + spread
                entry_price = curr_bar['close']
                
                if sl_price - entry_price > spread * 0.5:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'SHORT',
                        'entry': entry_price,
                        'stop_loss': sl_price
                    })
                    break
                    
    return signals

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    """3層MTF戦略 v6 シグナル生成"""
    h4_env = analyze_h4_environment(data_4h)
    
    signals_list = []
    
    # 15分足ループ
    for i in range(len(data_15m)):
        h4_idx = i // 16
        if h4_idx >= len(h4_env): h4_idx = len(h4_env) - 1
        
        h4_trend = h4_env.iloc[h4_idx]['trend']
        h4_atr = h4_env.iloc[h4_idx]['atr']
        
        # 15分足区間の1分足データを抽出
        m15_start_time = data_15m.index[i]
        m15_end_time = m15_start_time + pd.Timedelta(minutes=15)
        m1_slice = data_1m.loc[m15_start_time:m15_end_time]
        
        if len(m1_slice) < 3: continue
        
        # 15分足のパターン判定
        m15_pattern = analyze_m15_pattern(data_15m.iloc[i:i+1], h4_trend, h4_atr)
        m15_signal = m15_pattern['signal'].iloc[0]
        
        if m15_signal != 0:
            m1_exec_signals = analyze_m1_execution(m1_slice, m15_signal, spread)
            signals_list.extend(m1_exec_signals)
            
    # Seriesに変換（バックテストエンジン用）
    signal_series = pd.Series(0, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig['direction'] == 'LONG' else -1
        signal_series.loc[sig['time']] = val
        
    return signal_series, signals_list
