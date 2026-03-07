#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v46 - 4Hトレンド緩和・RR 5.0・15分足開始エントリー型

改善点：
1. RR 5.0 固定: 1回の利益で5回分の負けをカバー。
2. 取引回数の絶対確保（4Hトレンド緩和）: 
   - 4H EMA20との乖離が小さくてもエントリーを許可。
   - 15分足の「各バーの開始時」に1回だけエントリー。
   - 1月データでも100回以上の取引を確実に行いつつ、バックテストの計算負荷を最小化。
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
    # トレンド判定を極限まで緩和
    h4_data.loc[(h4_data["close"] >= h4_data["ema20"]), "trend"] = 1
    h4_data.loc[(h4_data["close"] < h4_data["ema20"]), "trend"] = -1
    return h4_data

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    m15_atr = calculate_atr(data_15m)

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    signals_list = []

    # 15分足の各バーの開始時刻をループ
    for start_time in data_15m.index:
        if start_time not in data_1m.index:
            continue
            
        h4_idx = data_4h.index.get_indexer([start_time], method="ffill")[0]
        if h4_idx < 0:
            continue
        h4_trend = h4_env.iloc[h4_idx]["trend"]

        m15_idx = data_15m.index.get_indexer([start_time], method="ffill")[0]
        if m15_idx < 1:
            continue
            
        prev_m15 = data_15m.iloc[m15_idx - 1]
        atr_val = m15_atr.iloc[m15_idx] if m15_idx >= 0 else 0.05
        curr_bar = data_1m.loc[start_time]

        # ロングシグナル
        if h4_trend == 1:
            sl_price = prev_m15["low"] - (atr_val * 0.5)
            risk = curr_bar["close"] - sl_price
            if risk > 0.005:
                signal_series.loc[start_time] = 1
                tp_series.loc[start_time] = curr_bar["close"] + risk * 5.0
                sl_series.loc[start_time] = sl_price
                signals_list.append({"time": start_time, "direction": "LONG"})
        # ショートシグナル
        elif h4_trend == -1:
            sl_price = prev_m15["high"] + (atr_val * 0.5)
            risk = sl_price - curr_bar["close"]
            if risk > 0.005:
                signal_series.loc[start_time] = -1
                tp_series.loc[start_time] = curr_bar["close"] - risk * 5.0
                sl_series.loc[start_time] = sl_price
                signals_list.append({"time": start_time, "direction": "SHORT"})

    return signal_series, tp_series, sl_series, signals_list
