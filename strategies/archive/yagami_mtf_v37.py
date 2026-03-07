#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v37 - 超高頻度・RR 5.0・全バーエントリー型

改善点：
1. RR 5.0 固定: 1回の利益で5回分の負けをカバー。
2. 取引回数の絶対確保: 
   - 15分足の各バーの開始時に、4Hトレンド方向であれば即座にエントリー。
   - フィルタを一切排除し、1月データでも100回以上の取引を確実に行う。
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

    signals_list = []
    # 15分足の開始時刻をリスト化
    m15_start_times = data_15m.index.tolist()
    
    for start_time in m15_start_times:
        if start_time not in data_1m.index:
            continue
            
        h4_idx = data_4h.index.get_indexer([start_time], method="ffill")[0]
        if h4_idx < 0:
            continue
        h4_trend = h4_env.iloc[h4_idx]["trend"]
        if h4_trend == 0:
            continue

        m15_idx = data_15m.index.get_indexer([start_time], method="ffill")[0]
        if m15_idx < 1:
            continue
            
        prev_m15 = data_15m.iloc[m15_idx - 1]
        atr_val = m15_atr.iloc[m15_idx] if m15_idx >= 0 else 0.05
        curr_bar = data_1m.loc[start_time]

        # ロングシグナル (トレンド方向へ全バーエントリー)
        if h4_trend == 1:
            sl_price = prev_m15["low"] - (atr_val * 0.5)
            risk = curr_bar["close"] - sl_price
            if risk > 0.005:
                signals_list.append(
                    {
                        "time": start_time,
                        "direction": "LONG",
                        "tp": curr_bar["close"] + risk * 5.0,
                        "sl": sl_price,
                    }
                )
        # ショートシグナル
        elif h4_trend == -1:
            sl_price = prev_m15["high"] + (atr_val * 0.5)
            risk = sl_price - curr_bar["close"]
            if risk > 0.005:
                signals_list.append(
                    {
                        "time": start_time,
                        "direction": "SHORT",
                        "tp": curr_bar["close"] - risk * 5.0,
                        "sl": sl_price,
                    }
                )

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig["direction"] == "LONG" else -1
        if sig["time"] in signal_series.index:
            signal_series.loc[sig["time"]] = val
            tp_series.loc[sig["time"]] = sig["tp"]
            sl_series.loc[sig["time"]] = sig["sl"]

    return signal_series, tp_series, sl_series, signals_list
