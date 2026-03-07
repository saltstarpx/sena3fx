import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips=0.2):
    debug_signal_log = open("/home/ubuntu/sena3fx/results/yagami_mtf_v64_signal_debug.log", "w")
    debug_signal_log.write("Timestamp,ATR,Volatility_Threshold,Signal_Type,Entry,TP,SL,Risk,EMA_Diff\n")

    m15_atr = calculate_atr(data_15m)
    
    # 1分足のEMA計算 (短期的な押し目・戻り目の判断用)
    data_1m["ema20"] = data_1m["close"].ewm(span=20, adjust=False).mean()
    
    # 上位足のEMA計算
    data_1h["ema20"] = data_1h["close"].ewm(span=20, adjust=False).mean()
    data_1h["ema75"] = data_1h["close"].ewm(span=75, adjust=False).mean()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["ema75"] = data_4h["close"].ewm(span=75, adjust=False).mean()

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    entry_time_series = pd.Series(pd.NaT, index=data_1m.index)
    atr_at_entry_series = pd.Series(np.nan, index=data_1m.index)

    atr_ma = m15_atr.rolling(window=24).mean()
    volatility_factor = 1.0 
    
    for i in range(24, len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]
        current_atr_ma = atr_ma.iloc[i]

        if pd.isna(atr_val) or pd.isna(current_atr_ma):
            continue

        if atr_val < (current_atr_ma * volatility_factor):
            continue

        # 時間帯フィルター
        current_hour_utc = current_m15_bar.name.hour
        if 15 <= current_hour_utc <= 21:
            continue

        # 上位足トレンド
        h1_time = current_m15_bar.name.floor("1H")
        h4_time = current_m15_bar.name.floor("4H")
        if h1_time not in data_1h.index or h4_time not in data_4h.index:
            continue
            
        current_1h_bar = data_1h.loc[h1_time]
        current_4h_bar = data_4h.loc[h4_time]

        long_trend = (current_1h_bar["close"] > current_1h_bar["ema20"] > current_1h_bar["ema75"]) and \
                     (current_4h_bar["close"] > current_4h_bar["ema20"] > current_4h_bar["ema75"])
        short_trend = (current_1h_bar["close"] < current_1h_bar["ema20"] < current_1h_bar["ema75"]) and \
                      (current_4h_bar["close"] < current_4h_bar["ema20"] < current_4h_bar["ema75"])

        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low = min(current_m15_bar["open"], current_m15_bar["close"])
        
        long_wick = (body_low - current_m15_bar["low"]) > (atr_val * 0.5)
        short_wick = (current_m15_bar["high"] - body_high) > (atr_val * 0.5)
        
        # 実体が大きすぎない（パニック売り・買いではない）
        moderate_move = (body_high - body_low) < (atr_val * 1.5)

        if long_trend and long_wick and moderate_move:
            mid_point = current_m15_bar["low"] + (body_low - current_m15_bar["low"]) * 0.5
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                # 1分足EMA乖離フィルター: 価格がEMA20より下にある（押し目）
                ema_diff = bar["close"] - bar["ema20"]
                if bar["low"] <= mid_point and ema_diff < 0:
                    entry_price = mid_point
                    sl_price = entry_price - (atr_val * 1.5)
                    risk = entry_price - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price + (risk * 2.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        debug_signal_log.write(f"{bar.name}, {atr_val:.5f}, {current_atr_ma:.5f}, LONG, {entry_price:.5f}, {tp_series.loc[bar.name]:.5f}, {sl_price:.5f}, {risk:.5f}, {ema_diff:.5f}\n")
                        break

        elif short_trend and short_wick and moderate_move:
            mid_point = current_m15_bar["high"] - (current_m15_bar["high"] - body_high) * 0.5
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                # 1分足EMA乖離フィルター: 価格がEMA20より上にある（戻り目）
                ema_diff = bar["close"] - bar["ema20"]
                if bar["high"] >= mid_point and ema_diff > 0:
                    entry_price = mid_point
                    sl_price = entry_price + (atr_val * 1.5)
                    risk = sl_price - entry_price
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price - (risk * 2.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        debug_signal_log.write(f"{bar.name}, {atr_val:.5f}, {current_atr_ma:.5f}, SHORT, {entry_price:.5f}, {tp_series.loc[bar.name]:.5f}, {sl_price:.5f}, {risk:.5f}, {ema_diff:.5f}\n")
                        break

    debug_signal_log.close()
    return signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series
