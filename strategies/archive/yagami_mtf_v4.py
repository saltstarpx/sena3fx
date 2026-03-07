#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v4 - スプレッド耐性版

改善点：
1. 「実体ベースの壁」: スプレッド以上の値幅を持つレンジを検出
2. 「ミスプライスの埋め」: 髭を付けた後、実体で埋め戻す動きを待つ
3. 「横軸の強度」: 1分足での止まりが、スプレッド以上の値幅を持つことを確認
"""

import pandas as pd
import numpy as np

def analyze_h4_environment(h4_data, atr_period=14):
    """4時間足での環境認識"""
    h4_data = h4_data.copy()
    
    # ATR計算
    h4_data['atr'] = calculate_atr(h4_data, atr_period)
    
    # トレンド判定（簡易版）
    h4_data['trend'] = 0
    h4_data.loc[h4_data['close'] > h4_data['close'].shift(3), 'trend'] = 1  # 上昇
    h4_data.loc[h4_data['close'] < h4_data['close'].shift(3), 'trend'] = -1  # 下降
    
    # 「実体ベースの壁」: 複数の足の実体が揃っているか
    h4_data['body_bottom'] = h4_data[['open', 'close']].min(axis=1)
    h4_data['body_top'] = h4_data[['open', 'close']].max(axis=1)
    
    # 直近3本の実体の下端が揃っているか（スプレッド以上の精度で）
    h4_data['body_alignment'] = 0
    for i in range(3, len(h4_data)):
        bodies = h4_data['body_bottom'].iloc[i-3:i].values
        body_std = np.std(bodies)
        # スプレッド相当（0.01 = 1pips）以下の分散なら「揃っている」
        if body_std <= 0.015:  # 1.5pips相当
            h4_data.loc[h4_data.index[i], 'body_alignment'] = 1
    
    return h4_data

def analyze_m15_pattern(m15_data, h4_trend, h4_atr, spread=0.01):
    """15分足でのパターン認識"""
    m15_data = m15_data.copy()
    
    # 「ミスプライスの埋め」: 髭を付けた後、実体で埋め戻す
    m15_data['signal'] = 0
    
    for i in range(2, len(m15_data)):
        # 直前2本を確認
        prev_bar = m15_data.iloc[i-1]
        curr_bar = m15_data.iloc[i]
        
        # ロングシグナル: 下髭が長い陽線（ミスプライスの埋め）
        if h4_trend > 0:
            lower_wick = prev_bar['low']
            body_low = min(prev_bar['open'], prev_bar['close'])
            wick_length = body_low - lower_wick
            
            # 髭が十分に長く、かつ現在の足が上昇している
            if wick_length > spread * 3 and curr_bar['close'] > curr_bar['open']:
                m15_data.loc[m15_data.index[i], 'signal'] = 1
        
        # ショートシグナル: 上髭が長い陰線
        elif h4_trend < 0:
            upper_wick = prev_bar['high']
            body_high = max(prev_bar['open'], prev_bar['close'])
            wick_length = upper_wick - body_high
            
            if wick_length > spread * 3 and curr_bar['close'] < curr_bar['open']:
                m15_data.loc[m15_data.index[i], 'signal'] = -1
    
    return m15_data

def analyze_m1_execution(m1_data, m15_signal, spread=0.01):
    """1分足での精密な執行"""
    m1_data = m1_data.copy()
    
    signals = []
    
    for i in range(10, len(m1_data)):
        if m15_signal == 0:
            continue
        
        # 直近10本の1分足を確認
        recent_m1 = m1_data.iloc[i-10:i]
        
        # 「横軸の強度」: ボラティリティの収縮
        recent_high = recent_m1['high'].max()
        recent_low = recent_m1['low'].min()
        volatility = recent_high - recent_low
        
        # スプレッド以上の値幅がある場合のみ
        if volatility < spread * 5:  # 5pips未満は無視
            continue
        
        # 現在の足がボラティリティ収縮の後、放れているか
        curr_bar = m1_data.iloc[i]
        
        if m15_signal > 0:  # ロング
            # 下値が堅い + 上に放れている
            if curr_bar['close'] > recent_m1['close'].iloc[-2]:
                # 「背」: 直近10本の安値 - スプレッド
                stop_loss = recent_low - spread
                entry_price = curr_bar['close']
                
                signals.append({
                    'time': m1_data.index[i],
                    'direction': 'LONG',
                    'entry': entry_price,
                    'stop_loss': stop_loss,
                    'sl_distance': entry_price - stop_loss,
                    'volatility': volatility
                })
        
        elif m15_signal < 0:  # ショート
            # 上値が重い + 下に放れている
            if curr_bar['close'] < recent_m1['close'].iloc[-2]:
                # 「背」: 直近10本の高値 + スプレッド
                stop_loss = recent_high + spread
                entry_price = curr_bar['close']
                
                signals.append({
                    'time': m1_data.index[i],
                    'direction': 'SHORT',
                    'entry': entry_price,
                    'stop_loss': stop_loss,
                    'sl_distance': stop_loss - entry_price,
                    'volatility': volatility
                })
    
    return signals

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    """3層MTF戦略のシグナル生成"""
    
    # 4時間足の環境認識
    h4_env = analyze_h4_environment(data_4h)
    
    signals_list = []
    
    # 15分足ごとに処理
    for i in range(len(data_15m)):
        # 対応する4時間足のインデックスを取得
        h4_idx = i // 16  # 4時間 = 15分 × 16本
        if h4_idx >= len(h4_env):
            break
        
        h4_trend = h4_env.iloc[h4_idx]['trend']
        h4_atr = h4_env.iloc[h4_idx]['atr']
        
        # 15分足のパターン認識
        m15_start = i * 15
        m15_end = min(m15_start + 15, len(data_1m))
        m15_data = data_1m.iloc[m15_start:m15_end]
        
        if len(m15_data) < 3:
            continue
        
        m15_pattern = analyze_m15_pattern(m15_data, h4_trend, h4_atr, spread)
        m15_signal = m15_pattern['signal'].iloc[-1]
        
        if m15_signal == 0:
            continue
        
        # 1分足での執行
        m1_signals = analyze_m1_execution(data_1m.iloc[m15_start:m15_end], m15_signal, spread)
        signals_list.extend(m1_signals)
    
    # シグナルをSeriesに変換
    signal_series = pd.Series(0, index=data_1m.index)
    
    for sig in signals_list:
        if sig['direction'] == 'LONG':
            signal_series.loc[sig['time']] = 1
        else:
            signal_series.loc[sig['time']] = -1
    
    return signal_series, signals_list

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
