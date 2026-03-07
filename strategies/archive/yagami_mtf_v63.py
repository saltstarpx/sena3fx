import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    # 修正: 正しいATR計算（Wilderの平滑化に近いが、ここでは標準的なrolling meanを使用）
    return true_range.rolling(period).mean()

def generate_signals(data_1m, data_15m, data_1h, data_4h, spread_pips=0.2):
    debug_signal_log = open("/home/ubuntu/sena3fx/results/yagami_mtf_v63_signal_debug.log", "w")
    debug_signal_log.write("Timestamp,ATR,Volatility_Threshold,Signal_Type,Entry,TP,SL,Risk\n")

    m15_atr = calculate_atr(data_15m)
    # データの初期段階でのNaNを回避するため、十分な期間が経過してから開始
    start_idx = 20 
    
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

    # 動的ボラティリティフィルター
    atr_ma = m15_atr.rolling(window=24).mean()
    volatility_factor = 1.0 # 閾値を1.0倍に引き上げ
    
    for i in range(start_idx, len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]
        current_atr_ma = atr_ma.iloc[i]

        if pd.isna(atr_val) or pd.isna(current_atr_ma):
            continue

        dynamic_volatility_threshold = current_atr_ma * volatility_factor
        if atr_val < dynamic_volatility_threshold:
            continue

        # 時間帯フィルター (UTC 15:00-21:00 は日本時間の深夜0時-6時)
        current_hour_utc = current_m15_bar.name.hour
        if 15 <= current_hour_utc <= 21:
            continue

        # 上位足のバーを取得
        h1_time = current_m15_bar.name.floor("1H")
        h4_time = current_m15_bar.name.floor("4H")
        
        if h1_time not in data_1h.index or h4_time not in data_4h.index:
            continue
            
        current_1h_bar = data_1h.loc[h1_time]
        current_4h_bar = data_4h.loc[h4_time]

        # トレンドフィルター（EMAの並び）
        long_trend = (current_1h_bar["close"] > current_1h_bar["ema20"] > current_1h_bar["ema75"]) and \
                     (current_4h_bar["close"] > current_4h_bar["ema20"] > current_4h_bar["ema75"])
        short_trend = (current_1h_bar["close"] < current_1h_bar["ema20"] < current_1h_bar["ema75"]) and \
                      (current_4h_bar["close"] < current_4h_bar["ema20"] < current_4h_bar["ema75"])

        # 実体と髭の分析
        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low = min(current_m15_bar["open"], current_m15_bar["close"])
        
        # 修正: 髭の長さがATRの0.5倍以上、かつ実体がATRの1.0倍以下（行き過ぎた動きの抑制）
        long_wick = (body_low - current_m15_bar["low"]) > (atr_val * 0.5)
        short_wick = (current_m15_bar["high"] - body_high) > (atr_val * 0.5)
        body_size = body_high - body_low
        moderate_move = body_size < (atr_val * 1.5)

        if long_trend and long_wick and moderate_move:
            # 1分足での指値エントリー（15分足の髭の50%戻し）
            mid_point = current_m15_bar["low"] + (body_low - current_m15_bar["low"]) * 0.5
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                if bar["low"] <= mid_point:
                    entry_price = mid_point
                    sl_price = entry_price - (atr_val * 1.5) # SLを1.5倍にタイト化
                    risk = entry_price - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price + (risk * 2.0) # RR 1:2
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        debug_signal_log.write(f"{bar.name}, {atr_val:.5f}, {dynamic_volatility_threshold:.5f}, LONG, {entry_price:.5f}, {tp_series.loc[bar.name]:.5f}, {sl_price:.5f}, {risk:.5f}\n")
                        break

        elif short_trend and short_wick and moderate_move:
            # 1分足での指値エントリー
            mid_point = current_m15_bar["high"] - (current_m15_bar["high"] - body_high) * 0.5
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                if bar["high"] >= mid_point:
                    entry_price = mid_point
                    sl_price = entry_price + (atr_val * 1.5)
                    risk = sl_price - entry_price
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price - (risk * 2.0) # RR 1:2
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        debug_signal_log.write(f"{bar.name}, {atr_val:.5f}, {dynamic_volatility_threshold:.5f}, SHORT, {entry_price:.5f}, {tp_series.loc[bar.name]:.5f}, {sl_price:.5f}, {risk:.5f}\n")
                        break

    debug_signal_log.close()
    return signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series
