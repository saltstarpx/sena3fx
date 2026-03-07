import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def generate_signals(data_15m, data_1h, data_4h, spread=0.00002):
    # data_15m: 執行足 (旧data_1m)
    # data_1h: パターン足 (旧data_15m)
    # data_4h: 環境認識足 (旧data_4h)

    m1h_atr = calculate_atr(data_1h)
    
    # 4時間足のトレンド判定
    data_4h["ema"] = data_4h["close"].ewm(span=20).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema"], 1, -1)

    signal_series = pd.Series(0, index=data_15m.index)
    tp_series = pd.Series(np.nan, index=data_15m.index)
    sl_series = pd.Series(np.nan, index=data_15m.index)
    signals_list = []

    for i in range(len(data_1h)):
        current_m1h_bar = data_1h.iloc[i]
        atr_val = m1h_atr.iloc[i] if not pd.isna(m1h_atr.iloc[i]) else 0.0005
        
        # 4時間足のトレンド確認
        current_4h_trend = data_4h.loc[:current_m1h_bar.name].iloc[-1]["trend"] if not data_4h.loc[:current_m1h_bar.name].empty else 0

        # 実体ベースの壁
        body_high = max(current_m1h_bar["open"], current_m1h_bar["close"])
        body_low = min(current_m1h_bar["open"], current_m1h_bar["close"])

        # 髭の定義（ATRベース）
        long_condition = (current_4h_trend == 1) and ((body_low - current_m1h_bar["low"]) > (atr_val * 0.5))
        short_condition = (current_4h_trend == -1) and ((current_m1h_bar["high"] - body_high) > (atr_val * 0.5))

        if long_condition:
            start_15m_time = current_m1h_bar.name
            end_15m_time = data_1h.index[i+1] if i+1 < len(data_1h) else data_15m.index[-1]
            entry_15m_bars = data_15m.loc[start_15m_time:end_15m_time]
            
            for _, bar in entry_15m_bars.iterrows():
                # 髭の中間点（50%戻し）まで引きつける
                mid_point = current_m1h_bar["low"] + (body_low - current_m1h_bar["low"]) * 0.5
                if bar["low"] <= mid_point:
                    sl_price = current_m1h_bar["low"] - (atr_val * 1.0)
                    risk = bar["close"] - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = bar["close"] + risk * 5.0 # RR 5.0
                        signals_list.append({"time": bar.name, "direction": "LONG"})
                        break

        elif short_condition:
            start_15m_time = current_m1h_bar.name
            end_15m_time = data_1h.index[i+1] if i+1 < len(data_1h) else data_15m.index[-1]
            entry_15m_bars = data_15m.loc[start_15m_time:end_15m_time]
            
            for _, bar in entry_15m_bars.iterrows():
                # 髭の中間点（50%戻し）まで引きつける
                mid_point = current_m1h_bar["high"] - (current_m1h_bar["high"] - body_high) * 0.5
                if bar["high"] >= mid_point:
                    sl_price = current_m1h_bar["high"] + (atr_val * 1.0)
                    risk = sl_price - bar["close"]
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = bar["close"] - risk * 5.0 # RR 5.0
                        signals_list.append({"time": bar.name, "direction": "SHORT"})
                        break

    return signal_series, tp_series, sl_series, signals_list
