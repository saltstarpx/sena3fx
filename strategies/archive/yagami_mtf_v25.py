#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v25 - 髭厳格化・極端BB決済・高期待値型

改善点：
1. 髭判定の厳格化: 髭の長さがATR(15m)の0.2倍以上（以前は0.05倍）の場合のみ。
2. ボリンジャーバンド(20, 2.5)決済: 極端なオーバーシュートでの利益確定。
3. 損切り: エントリーした15分足の髭先（絶対的な壁）の外側に設定。
4. 4Hトレンドフィルタ: 傾き条件を0.0005以上に厳格化。
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

def calculate_bb(df, period=20, std_dev=2.5):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def analyze_h4_environment(h4_data):
    h4_data = h4_data.copy()
    h4_data["ema20"] = h4_data["close"].ewm(span=20).mean()
    h4_data["ema_slope"] = h4_data["ema20"].diff(3)
    h4_data["trend"] = 0
    h4_data.loc[(h4_data["close"] > h4_data["ema20"]) & (h4_data["ema_slope"] > 0.0005), "trend"] = 1
    h4_data.loc[(h4_data["close"] < h4_data["ema20"]) & (h4_data["ema_slope"] < -0.0005), "trend"] = -1
    return h4_data

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    m15_atr = calculate_atr(data_15m)
    m1_upper, m1_lower = calculate_bb(data_1m, period=20, std_dev=2.5)

    signals_list = []
    i = 15 * 20
    while i < len(data_1m):
        curr_time = data_1m.index[i]

        h4_idx = data_4h.index.get_indexer([curr_time], method="ffill")[0]
        if h4_idx < 0:
            i += 1
            continue
        h4_trend = h4_env.iloc[h4_idx]["trend"]
        if h4_trend == 0:
            i += 1
            continue

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

        # ロングシグナル判定 (髭厳格化)
        if h4_trend == 1 and lower_wick > body_size * 1.5 and lower_wick > atr_val * 0.2:
            if curr_bar["close"] > curr_bar["open"] and curr_bar["close"] > m15_low:
                signals_list.append(
                    {
                        "time": curr_time,
                        "direction": "LONG",
                        "sl": m15_low - 0.01, # 髭先から1pips下
                    }
                )
                i += 15
                continue
        # ショートシグナル判定 (髭厳格化)
        elif h4_trend == -1 and upper_wick > body_size * 1.5 and upper_wick > atr_val * 0.2:
            if curr_bar["close"] < curr_bar["open"] and curr_bar["close"] < m15_high:
                signals_list.append(
                    {
                        "time": curr_time,
                        "direction": "SHORT",
                        "sl": m15_high + 0.01, # 髭先から1pips上
                    }
                )
                i += 15
                continue
        i += 1

    signal_series = pd.Series(0, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    for sig in signals_list:
        val = 1 if sig["direction"] == "LONG" else -1
        if sig["time"] in signal_series.index:
            signal_series.loc[sig["time"]] = val
            sl_series.loc[sig["time"]] = sig["sl"]

    return signal_series, sl_series, m1_upper, m1_lower, signals_list
