#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v38 - マシンガン・RR 5.0・高頻度型

改善点：
1. RR 5.0 固定: 1回の利益で5回分の負けをカバー。
2. 取引回数の絶対確保: 
   - 1分足の各バーにおいて、4Hトレンド方向かつ15分足の髭が確認できれば即エントリー。
   - 重複エントリー制限を緩和し、チャンスを逃さない。
3. 損切り設定: 15分足の髭先から「ATR(15m)の0.5倍」離す。
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
    i = 15 * 5
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
        m15_high = m15_bar_data["high"]
        m15_low = m15_bar_data["low"]
        m15_open = m15_bar_data["open"]
        m15_close = m15_bar_data["close"]

        m15_idx = data_15m.index.get_indexer([m15_current_bar_start_time], method="ffill")[0]
        atr_val = m15_atr.iloc[m15_idx] if m15_idx >= 0 else 0.05

        curr_bar = data_1m.iloc[i]

        # ロングシグナル (下髭確認で即エントリー)
        if h4_trend == 1:
            lower_wick = min(m15_open, m15_close) - m15_low
            if lower_wick > atr_val * 0.01:
                sl_price = m15_low - (atr_val * 0.5)
                risk = curr_bar["close"] - sl_price
                if risk > 0.005:
                    signals_list.append(
                        {
                            "time": curr_time,
                            "direction": "LONG",
                            "tp": curr_bar["close"] + risk * 5.0,
                            "sl": sl_price,
                        }
                    )
                    i += 5 # 5分間隔でマシンガン
                    continue
        # ショートシグナル (上髭確認で即エントリー)
        elif h4_trend == -1:
            upper_wick = m15_high - max(m15_open, m15_close)
            if upper_wick > atr_val * 0.01:
                sl_price = m15_high + (atr_val * 0.5)
                risk = sl_price - curr_bar["close"]
                if risk > 0.005:
                    signals_list.append(
                        {
                            "time": curr_time,
                            "direction": "SHORT",
                            "tp": curr_bar["close"] - risk * 5.0,
                            "sl": sl_price,
                        }
                    )
                    i += 5 # 5分間隔でマシンガン
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
