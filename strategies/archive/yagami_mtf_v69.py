"""
Yagami MTF v69
==============
診断結果に基づく修正:
  ❶ 髭閾値を ATR×0.5 → ATR×0.3 に緩和（最大のボトルネック解消）
  ❷ 1時間足トレンドフィルターを除外（v58本来のシンプルさに戻す）
  ❸ タイム・ディケイなし（水原様の指示）
  ❹ ボラティリティフィルターは維持（spread×3 の極端な低ボラのみ除外）
  ❺ SL: 髭の先端 - ATR×0.5、TP: リスク×5.0（v58/v59と同一）
"""
import pandas as pd
import numpy as np


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips=0.2):
    spread = spread_pips * 0.01  # USDJPY: 0.2pips = 0.002円

    m15_atr = calculate_atr(data_15m)

    # 4時間足トレンド（v58/v59と同一: EMA20）
    data_4h = data_4h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    entry_time_series = pd.Series(pd.NaT, index=data_1m.index)
    atr_at_entry_series = pd.Series(np.nan, index=data_1m.index)

    # ボラティリティフィルター（spread×3 の極端な低ボラのみ除外）
    volatility_threshold = spread * 3.0

    # 髭閾値: ATR×0.3（緩和）
    WICK_MULT = 0.3

    for i in range(len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]

        if pd.isna(atr_val):
            atr_val = 0.0005
            continue

        if atr_val < volatility_threshold:
            continue

        h4_time = current_m15_bar.name.floor("4h")
        if h4_time not in data_4h.index:
            continue
        current_4h_trend = data_4h.loc[h4_time]["trend"]

        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low  = min(current_m15_bar["open"], current_m15_bar["close"])

        lower_wick = body_low - current_m15_bar["low"]
        upper_wick = current_m15_bar["high"] - body_high
        wick_threshold = atr_val * WICK_MULT

        long_condition  = (current_4h_trend == 1)  and (lower_wick > wick_threshold)
        short_condition = (current_4h_trend == -1) and (upper_wick > wick_threshold)

        if long_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i + 1] if i + 1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]

            for _, bar in entry_1m_bars.iterrows():
                mid_point = current_m15_bar["low"] + lower_wick * 0.5
                if bar["low"] <= mid_point:
                    entry_price = mid_point
                    sl_price = current_m15_bar["low"] - (atr_val * 0.5)
                    risk = entry_price - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price + (risk * 5.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        break

        elif short_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i + 1] if i + 1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]

            for _, bar in entry_1m_bars.iterrows():
                mid_point = current_m15_bar["high"] - upper_wick * 0.5
                if bar["high"] >= mid_point:
                    entry_price = mid_point
                    sl_price = current_m15_bar["high"] + (atr_val * 0.5)
                    risk = sl_price - entry_price
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price - (risk * 5.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        break

    return signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series
