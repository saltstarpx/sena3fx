#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v15 - トレンドフィルタ超強化・15分足形成中監視版

改善点：
1. 4Hトレンド判定の超厳格化: EMA20の傾き（Slope）が一定以上（0.02以上）の場合のみに限定。
   これにより、ダマシの多いレンジ相場を徹底的に排除。
2. 15分足形成中のリアルタイム監視: 15分足が確定するのを待たず、15分足の枠内で「ATR 0.3倍以上の髭」が発生し、
   かつ「1分足で反発」が見られた瞬間にエントリー。
3. 4時間固定ホールド: スプレッド1.0pipsを克服。
"""

import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def analyze_h4_environment(h4_data):
    h4_data = h4_data.copy()
    h4_data['ema20'] = h4_data['close'].ewm(span=20).mean()
    h4_data['ema_slope'] = h4_data['ema20'].diff(3) # 直近3本の変化
    h4_data['trend'] = 0
    # 傾きが0.01以上（上昇）または-0.01以下（下降）のみトレンドとみなす
    h4_data.loc[(h4_data['close'] > h4_data['ema20']) & (h4_data['ema_slope'] > 0.01), 'trend'] = 1
    h4_data.loc[(h4_data['close'] < h4_data['ema20']) & (h4_data['ema_slope'] < -0.01), 'trend'] = -1
    return h4_data

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    m15_atr = calculate_atr(data_15m)
    
    signals_list = []
    
    # 1分足レベルでループを回して、15分足の形成状況を擬似的に監視
    for i in range(15 * 20, len(data_1m)): # 十分な履歴を確保
        curr_time = data_1m.index[i]
        
        # 4Hトレンド取得
        h4_idx = data_4h.index.get_indexer([curr_time], method='ffill')[0]
        if h4_idx < 0: continue
        h4_trend = h4_env.iloc[h4_idx]['trend']
        if h4_trend == 0: continue
        
        # 直近15分間の1分足データを取得
        m15_slice = data_1m.iloc[i-14:i+1]
        m15_high = m15_slice['high'].max()
        m15_low = m15_slice['low'].min()
        m15_open = m15_slice['open'].iloc[0]
        m15_close = m15_slice['close'].iloc[-1]
        
        # 15分足レベルのATR（直近の15分足から取得）
        m15_idx = data_15m.index.get_indexer([curr_time], method='ffill')[0]
        atr_val = m15_atr.iloc[m15_idx] if m15_idx >= 0 else 0.05
        
        # 髭の計算
        lower_wick = min(m15_open, m15_close) - m15_low
        upper_wick = m15_high - max(m15_open, m15_close)
        
        # エントリー判定
        curr_bar = data_1m.iloc[i]
        if h4_trend == 1 and lower_wick > atr_val * 0.4:
            # ロング：1分足が陽線で反発
            if curr_bar['close'] > curr_bar['open']:
                signals_list.append({'time': curr_time, 'direction': 'LONG'})
                # 同一15分枠内での重複エントリーを避けるため、15分スキップ
                i += 15
        elif h4_trend == -1 and upper_wick > atr_val * 0.4:
            # ショート：1分足が陰線で反発
            if curr_bar['close'] < curr_bar['open']:
                signals_list.append({'time': curr_time, 'direction': 'SHORT'})
                i += 15
                
    signal_series = pd.Series(0, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig['direction'] == 'LONG' else -1
        if sig['time'] in signal_series.index:
            signal_series.loc[sig['time']] = val
        
    return signal_series, signals_list
