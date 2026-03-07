#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v26 - 4H確定待機・極限厳格化・高値安値決済型

改善点：
1. 4H確定待機: 4時間足が確定した直後の15分間のみエントリーチャンスとする。
2. 髭判定の極限厳格化: 髭の長さがATR(15m)の0.3倍以上の場合のみ。
3. 高値安値決済: 
   - ロング: 直近4時間足の最高値で利益確定。
   - ショート: 直近4時間足の最安値で利益確定。
4. 損切り: エントリーした15分足の髭先。
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
    h4_data["ema_slope"] = h4_data["ema20"].diff(3)
    h4_data["trend"] = 0
    h4_data.loc[(h4_data["close"] > h4_data["ema20"]) & (h4_data["ema_slope"] > 0.001), "trend"] = 1
    h4_data.loc[(h4_data["close"] < h4_data["ema20"]) & (h4_data["ema_slope"] < -0.001), "trend"] = -1
    return h4_data

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    m15_atr = calculate_atr(data_15m)

    signals_list = []
    i = 15 * 20
    while i < len(data_1m):
        curr_time = data_1m.index[i]

        # 4時間足の確定直後（各4時間の最初の15分間）のみ
        if not (curr_time.hour % 4 == 0 and curr_time.minute < 15):
            i += 1
            continue

        h4_idx = data_4h.index.get_indexer([curr_time], method="ffill")[0]
        if h4_idx < 1:
            i += 1
            continue
        h4_trend = h4_env.iloc[h4_idx]["trend"]
        if h4_trend == 0:
            i += 1
            continue
        
        # 直近4時間足の高値・安値
        prev_h4 = data_4h.iloc[h4_idx-1]
        h4_high = prev_h4["high"]
        h4_low = prev_h4["low"]

        m15_current_bar_start_time = curr_time.floor("15min")
        if m15_current_bar_start_time not in data_15m.index:
            i += 1
            continue

        m15_bar_data = data_15m.loc[m15_current_bar_start_time]
        m15_open = m15_bar_data["open"]
        m15_close = m15_bar_data["close"]
        m15_high = m15_bar_data["high"]
        m15_low = m15_bar_data["low"]

        m15_idx = data_15m.index.get_indexer([m15_current_bar_start_time], method="ffill")[0]
        atr_val = m15_atr.iloc[m15_idx] if m15_idx >= 0 else 0.05

        lower_wick = min(m15_open, m15_close) - m15_low
        upper_wick = m15_high - max(m15_open, m15_close)
        body_size = abs(m15_close - m15_open)

        curr_bar = data_1m.iloc[i]

        # ロングシグナル判定
        if h4_trend == 1 and lower_wick > body_size * 2.0 and lower_wick > atr_val * 0.3:
            if curr_bar["close"] > curr_bar["open"] and curr_bar["close"] > m15_low:
                signals_list.append(
                    {
                        "time": curr_time,
                        "direction": "LONG",
                        "tp": h4_high,
                        "sl": m15_low - 0.01,
                    }
                )
                i += 15
                continue
        # ショートシグナル判定
        elif h4_trend == -1 and upper_wick > body_size * 2.0 and upper_wick > atr_val * 0.3:
            if curr_bar["close"] < curr_bar["open"] and curr_bar["close"] < m15_high:
                signals_list.append(
                    {
                        "time": curr_time,
                        "direction": "SHORT",
                        "tp": h4_low,
                        "sl": m15_high + 0.01,
                    }
                )
                i += 15
                continue
        i += 1

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
