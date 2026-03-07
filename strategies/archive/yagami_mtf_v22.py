#!/usr/bin/env python3
"""
やがみ式3層MTF戦略 v22 - 実体ベースの壁・ミスプライス埋めエントリー版

改善点：
1. 4Hトレンド判定の緩和: EMA20の傾き（Slope）が0.0001以上の場合のみに限定。
2. 15分足の髭判定: 髭の長さが実体の1.0倍以上、かつATRの0.05倍以上の場合を「強い髭」と定義。
3. ミスプライス埋めエントリー: 15分足で強い髭が出現した後、その髭が1分足で埋められる動き（髭の安値/高値を実体で更新する）を確認してからエントリー。
4. ストップロス: エントリーした15分足の「実体ベースの壁」の外側に設定（例: ロングなら15分足の安値 - 0.02pips）。
5. 利益確定: 4時間固定ホールドを継続。
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

        # 現在の1分足が属する15分足の情報を取得
        m15_current_bar_start_time = curr_time.floor("15min")
        m15_current_bar_end_time = m15_current_bar_start_time + pd.Timedelta(minutes=14, seconds=59)
        
        # 15分足のデータが揃っているか確認
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
        if h4_trend == 1 and lower_wick > body_size * 1.0 and lower_wick > atr_val * 0.05:
            # ミスプライス埋めエントリー: 1分足が陽線で、かつ15分足の安値を実体で更新しないことを確認
            # ここでは、1分足の終値が15分足の安値より上にあることを条件とする
            if curr_bar["close"] > curr_bar["open"] and curr_bar["close"] > m15_low:
                signals_list.append(
                    {
                        "time": curr_time,
                        "direction": "LONG",
                        "sl": m15_low - 0.02,  # 15分足の安値から2pips下にSL
                    }
                )
                i += 15  # 同一15分枠内での重複エントリーを避けるため、15分スキップ
                continue
        # ショートシグナル判定
        elif h4_trend == -1 and upper_wick > body_size * 1.0 and upper_wick > atr_val * 0.05:
            # ミスプライス埋めエントリー: 1分足が陰線で、かつ15分足の高値を実体で更新しないことを確認
            # ここでは、1分足の終値が15分足の高値より下にあることを条件とする
            if curr_bar["close"] < curr_bar["open"] and curr_bar["close"] < m15_high:
                signals_list.append(
                    {
                        "time": curr_time,
                        "direction": "SHORT",
                        "sl": m15_high + 0.02,  # 15分足の高値から2pips上にSL
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

    return signal_series, sl_series, signals_list
