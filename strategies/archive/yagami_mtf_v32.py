#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v32 - 髭先ブレイク・RR 5.0・損切り緩和型

改善点：
1. RR設定: 5.0固定。
2. エントリータイミング: 15分足の髭を確認後、1分足で「15分足の髭先（高値/安値）」をブレイクした瞬間にエントリー。
3. 損切り設定: 15分足の髭先から「ATR(15m)の0.5倍」離す（ユーザー指示の「背を離す」を徹底）。
4. 取引回数確保: 髭の判定条件を緩和（ATRの0.01倍以上）し、100回以上のエントリーを死守。
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
    h4_data.loc[(h4_data["close"] > h4_data["ema20"]) & (h4_data["ema_slope"] > 0.0001), "trend"] = 1
    h4_data.loc[(h4_data["close"] < h4_data["ema20"]) & (h4_data["ema_slope"] < -0.0001), "trend"] = -1
    return h4_data

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    h4_env = analyze_h4_environment(data_4h)
    m15_atr = calculate_atr(data_15m)

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

        # ロングシグナル判定
        if h4_trend == 1 and lower_wick > body_size * 0.1 and lower_wick > atr_val * 0.01:
            # 1分足で15分足の安値をブレイクした瞬間にエントリー
            if curr_bar["low"] <= m15_low:
                sl_price = m15_low - (atr_val * 0.5)
                entry_price = m15_low
                risk = entry_price - sl_price
                if risk > 0.002:
                    signals_list.append(
                        {
                            "time": curr_time,
                            "direction": "LONG",
                            "tp": entry_price + risk * 5.0, # RR=5.0
                            "sl": sl_price,
                        }
                    )
                    i += 15
                    continue
        # ショートシグナル判定
        elif h4_trend == -1 and upper_wick > body_size * 0.1 and upper_wick > atr_val * 0.01:
            # 1分足で15分足の高値をブレイクした瞬間にエントリー
            if curr_bar["high"] >= m15_high:
                sl_price = m15_high + (atr_val * 0.5)
                entry_price = m15_high
                risk = sl_price - entry_price
                if risk > 0.002:
                    signals_list.append(
                        {
                            "time": curr_time,
                            "direction": "SHORT",
                            "tp": entry_price - risk * 5.0, # RR=5.0
                            "sl": sl_price,
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
