#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v5 - 動的ボラティリティ適応・高頻度版

改善点：
1. 「動的しきい値」: ATRに基づいた判定で、静かな相場でもチャンスを捉える（100回以上目標）
2. 「実体ベースの壁 + スプレッド余裕」: SLをスプレッド(1.0pips)の倍以上に設定し、即死を回避
3. 「1分足の収束判定の緩和」: 横軸の形成をより柔軟に検知
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
    
    # トレンド判定（緩やか）
    h4_data['trend'] = 0
    h4_data.loc[h4_data['close'] > h4_data['close'].shift(2), 'trend'] = 1
    h4_data.loc[h4_data['close'] < h4_data['close'].shift(2), 'trend'] = -1
    
    return h4_data

def analyze_m15_pattern(m15_data, h4_trend, h4_atr):
    """15分足でのパターン認識（動的しきい値）"""
    m15_data = m15_data.copy()
    
    # ATRに基づいた動的な髭の長さ基準（最小でも0.005 = 0.5pips）
    min_wick = max(h4_atr * 0.1, 0.005) if not pd.isna(h4_atr) else 0.005
    
    m15_data['lower_wick'] = m15_data[['open', 'close']].min(axis=1) - m15_data['low']
    m15_data['upper_wick'] = m15_data['high'] - m15_data[['open', 'close']].max(axis=1)
    
    m15_data['signal'] = 0
    
    # トレンドに沿った押し目・戻り、または反転
    # ロング: 下髭がある程度あり、陽線または下げ止まり
    m15_data.loc[(m15_data['lower_wick'] >= min_wick) & (m15_data['close'] >= m15_data['open']), 'signal'] = 1
    # ショート: 上髭がある程度あり、陰線または上げ止まり
    m15_data.loc[(m15_data['upper_wick'] >= min_wick) & (m15_data['close'] <= m15_data['open']), 'signal'] = -1
    
    return m15_data

def analyze_m1_execution(m1_data, m15_signal, spread=0.01):
    """1分足での精密な執行（スプレッド耐性重視）"""
    m1_data = m1_data.copy()
    signals = []
    
    # 直近5本のボラティリティ
    m1_data['range'] = m1_data['high'] - m1_data['low']
    m1_data['vol_ma'] = m1_data['range'].rolling(5).mean()
    
    for i in range(5, len(m1_data)):
        if m15_signal == 0:
            continue
            
        curr_bar = m1_data.iloc[i]
        recent_m1 = m1_data.iloc[i-5:i]
        
        # 「横軸」: 直近5本が大きく動いていない（収束）
        if curr_bar['vol_ma'] > 0.05: # 5pips以上の激しい動きの時は見送り
            continue
            
        if m15_signal > 0: # ロング検討
            # 前の足の高値を抜ける動き
            if curr_bar['close'] > recent_m1['high'].max():
                # SLは直近安値からスプレッド分(1.0pips)以上離す
                sl_price = recent_m1['low'].min() - spread * 1.2
                entry_price = curr_bar['close']
                
                # SLが近すぎないか確認（スプレッド即死回避）
                if entry_price - sl_price > spread * 1.5:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'LONG',
                        'entry': entry_price,
                        'stop_loss': sl_price
                    })
                    
        elif m15_signal < 0: # ショート検討
            if curr_bar['close'] < recent_m1['low'].min():
                sl_price = recent_m1['high'].max() + spread * 1.2
                entry_price = curr_bar['close']
                
                if sl_price - entry_price > spread * 1.5:
                    signals.append({
                        'time': m1_data.index[i],
                        'direction': 'SHORT',
                        'entry': entry_price,
                        'stop_loss': sl_price
                    })
                    
    return signals

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    """3層MTF戦略 v5 シグナル生成"""
    h4_env = analyze_h4_environment(data_4h)
    
    signals_list = []
    
    # 15分足ループ
    for i in range(len(data_15m)):
        h4_idx = i // 16
        if h4_idx >= len(h4_env): break
        
        h4_trend = h4_env.iloc[h4_idx]['trend']
        h4_atr = h4_env.iloc[h4_idx]['atr']
        
        # 15分足区間の1分足データを抽出
        m15_start_time = data_15m.index[i]
        m15_end_time = m15_start_time + pd.Timedelta(minutes=15)
        m1_slice = data_1m.loc[m15_start_time:m15_end_time]
        
        if len(m1_slice) < 5: continue
        
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
