
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

def _calc_atr(bars, period=14):
    h, l, c = bars['high'].values, bars['low'].values, bars['close'].values
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars.index).rolling(period).mean()

def _is_inside_bar(bars, idx):
    if idx < 1: return False
    cur, prev = bars.iloc[idx], bars.iloc[idx-1]
    return (cur['high'] <= prev['high']) and (cur['low'] >= prev['low'])

def _body_alignment_score(bars, idx, lookback=3):
    if idx < lookback: return np.nan
    closes = bars['close'].iloc[idx-lookback+1 : idx+1]
    return np.std(closes)

def analyze_h4_environment(h4_bars, atr_period=14, lookback=20):
    atr = _calc_atr(h4_bars, atr_period)
    current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.1
    recent = h4_bars.iloc[-lookback:]
    return {'wall_high': recent['high'].max(), 'wall_low': recent['low'].min(), 'atr': current_atr}

def analyze_m15_pattern(m15_bars, h4_env, atr_period=14, alignment_threshold=0.20):
    atr = _calc_atr(m15_bars, atr_period)
    current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else h4_env['atr']
    bar = m15_bars.iloc[-1]
    align_score = _body_alignment_score(m15_bars, len(m15_bars)-1, lookback=3)
    align_normalized = align_score / current_atr if current_atr > 0 else 10.0
    
    signal = None
    if bar['low'] <= h4_env['wall_low'] + 2.0 * h4_env['atr'] and align_normalized < alignment_threshold:
        signal = 'long_ready'
    elif bar['high'] >= h4_env['wall_high'] - 2.0 * h4_env['atr'] and align_normalized < alignment_threshold:
        signal = 'short_ready'
    return {'signal': signal}

def analyze_m1_execution(m1_bars, m15_signal, atr_period=14, lookback=5):
    if not m15_signal['signal']: return {'execute': False}
    atr = _calc_atr(m1_bars, atr_period)
    current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.01
    bar = m1_bars.iloc[-1]
    prev = m1_bars.iloc[-2]
    
    # 1Mでのボラティリティ収束（直近5本の平均レンジの半分以下）
    recent = m1_bars.iloc[-lookback:]
    avg_range = (recent['high'] - recent['low']).mean()
    bar_range = bar['high'] - bar['low']
    converged = bar_range < avg_range * 0.6
    
    execute = False
    direction = None
    # 収束からの放れ（陽線/陰線確定）
    if m15_signal['signal'] == 'long_ready' and bar['close'] > bar['open'] and converged:
        execute, direction = True, 'long'
    elif m15_signal['signal'] == 'short_ready' and bar['close'] < bar['open'] and converged:
        execute, direction = True, 'short'
    
    if execute:
        # やがみ氏の「髭先を背にする」: 直近1Mの安値/高値にSL
        sl = bar['low'] if direction == 'long' else bar['high']
        # スプレッド等を考慮して極小バッファ
        sl = sl - 0.005 if direction == 'long' else sl + 0.005
        
        return {
            'execute': True, 'direction': direction, 'entry_price': bar['close'],
            'stop_loss': sl, 'reason': '1M_Convergence_Break'
        }
    return {'execute': False}

def signal_yagami_mtf_v3(h4_bars, m15_bars, m1_bars):
    h4_env = analyze_h4_environment(h4_bars)
    m15_pattern = analyze_m15_pattern(m15_bars, h4_env)
    m1_exec = analyze_m1_execution(m1_bars, m15_pattern)
    if not m1_exec['execute']: return {'signal': None}
    
    # TPは4Hの反対側の壁
    tp = h4_env['wall_high'] if m1_exec['direction'] == 'long' else h4_env['wall_low']
    
    # RR比が低すぎる場合は見送り
    sl_dist = abs(m1_exec['entry_price'] - m1_exec['stop_loss'])
    tp_dist = abs(tp - m1_exec['entry_price'])
    if sl_dist == 0 or tp_dist / sl_dist < 1.5: return {'signal': None}
    
    return {
        'signal': m1_exec['direction'], 'entry_price': m1_exec['entry_price'],
        'stop_loss': m1_exec['stop_loss'], 'take_profit': tp, 'reason': m1_exec['reason']
    }
