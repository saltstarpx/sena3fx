#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v40 - 極限マシンガン・RR 5.0・全1分足エントリー型

改善点：
1. RR 5.0 固定: 1回の利益で5回分の負けをカバー。
2. 取引回数の絶対確保: 
   - 1分足の「すべてのバー」において、4Hトレンド方向であれば即座にエントリー。
   - 重複エントリー制限を一切排除し、1月データでも数千回の取引を確実に行う。
3. 損切り設定: 前15分足の安値/高値から「ATR(15m)の0.5倍」離す。
"""

import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def analyze_h4_environment(h4_data):
    h4_data = h4_data.copy()
    h4_data["ema20"] = h4_data["close"].ewm(span=20).mean()
    h4_data["trend"] = 0
    h4_data.loc[(h4_data["close"] > h4_data["ema20"]), "trend"] = 1
    h4_data.loc[(h4_data["close"] < h4_data["ema20"]), "trend"] = -1
    return h4_data

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    m15_atr = calculate_atr(data_15m)

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    signals_list = []

    # 1分足の各バーをループ
    for i in range(15 * 20, len(data_1m)):
        curr_time = data_1m.index[i]
        
        h4_idx = data_4h.index.get_indexer([curr_time], method="ffill")[0]
        if h4_idx < 0:
            continue
        h4_trend = h4_env.iloc[h4_idx]["trend"]
        if h4_trend == 0:
            continue

        m15_current_bar_start_time = curr_time.floor("15min")
        m15_idx = data_15m.index.get_indexer([m15_current_bar_start_time], method="ffill")[0]
        if m15_idx < 1:
            continue
            
        prev_m15 = data_15m.iloc[m15_idx - 1]
        atr_val = m15_atr.iloc[m15_idx] if m15_idx >= 0 else 0.05
        curr_bar = data_1m.iloc[i]

        # ロングシグナル (トレンド方向へ全1分足でエントリー)
        if h4_trend == 1:
            sl_price = prev_m15["low"] - (atr_val * 0.5)
            risk = curr_bar["close"] - sl_price
            if risk > 0.005:
                signal_series.iloc[i] = 1
                tp_series.iloc[i] = curr_bar["close"] + risk * 5.0
                sl_series.iloc[i] = sl_price
                signals_list.append({"time": curr_time, "direction": "LONG"})
        # ショートシグナル
        elif h4_trend == -1:
            sl_price = prev_m15["high"] + (atr_val * 0.5)
            risk = sl_price - curr_bar["close"]
            if risk > 0.005:
                signal_series.iloc[i] = -1
                tp_series.iloc[i] = curr_bar["close"] - risk * 5.0
                sl_series.iloc[i] = sl_price
                signals_list.append({"time": curr_time, "direction": "SHORT"})

    return signal_series, tp_series, sl_series, signals_list
