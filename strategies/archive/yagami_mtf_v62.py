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
    debug_signal_log = open("/home/ubuntu/sena3fx/results/yagami_mtf_v62_signal_debug.log", "w")
    debug_signal_log.write("Timestamp,ATR,Volatility_Threshold,Long_Wall,Short_Wall,Long_Wick,Short_Wick,Signal_Type,Entry,TP,SL,Risk,N_Value\n")

    spread_value = spread_pips * 0.01 # 0.2pipsの場合、0.002円
    m15_atr = calculate_atr(data_15m)
    m15_atr = m15_atr.fillna(0.0005) # ATRの初期NaN値を0.0005で埋める
    
    # v60: 上位足のEMA計算
    data_1h["ema20"] = data_1h["close"].ewm(span=20, adjust=False).mean()
    data_1h["ema75"] = data_1h["close"].ewm(span=75, adjust=False).mean()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["ema75"] = data_4h["close"].ewm(span=75, adjust=False).mean()

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    entry_time_series = pd.Series(pd.NaT, index=data_1m.index)

    atr_at_entry_series = pd.Series(np.nan, index=data_1m.index) # エントリー時のATRを記録

    # 動的ボラティリティフィルターの閾値
    # 直近のATRの移動平均を計算し、それに対する比率で閾値を設定
    atr_ma = m15_atr.rolling(window=24, min_periods=1).mean() # 例: 過去24期間のATR移動平均 (min_periods=1で初期NaNを回避)
    volatility_factor = 0.8 # ATR_MAの0.8倍を閾値とする
    for i in range(len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]
        current_atr_ma = atr_ma.iloc[i]

        # ATRまたはATR_MAがNaNの場合はスキップ
        if pd.isna(atr_val) or pd.isna(current_atr_ma):
            debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, ATR or ATR_MA is NaN. Skipping.\n")
            continue

        dynamic_volatility_threshold = current_atr_ma * volatility_factor

        debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, ATR: {atr_val:.5f}, Dynamic Volatility Threshold: {dynamic_volatility_threshold:.5f}, ATR MA: {current_atr_ma:.5f}\n")

        if atr_val < dynamic_volatility_threshold:
            debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, Volatility too low (Dynamic Filter): ATR={atr_val:.5f}, Threshold={dynamic_volatility_threshold:.5f}\n")
            continue # ボラティリティが低い場合はスキップ

        # 時間帯フィルター (UTC 15:00-21:00 は日本時間の深夜0時-6時)
        current_hour_utc = current_m15_bar.name.hour
        if 15 <= current_hour_utc <= 21: # UTC 15時-21時 (日本時間 0時-6時)
            debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, Trading prohibited during low liquidity hours (UTC {current_hour_utc}:00)\n")
            continue # 特定の時間帯は取引をスキップ

        # 上位足のバーを取得
        current_1h_bar = data_1h.loc[current_m15_bar.name.floor("1H")] if current_m15_bar.name.floor("1H") in data_1h.index else None
        current_4h_bar = data_4h.loc[current_m15_bar.name.floor("4H")] if current_m15_bar.name.floor("4H") in data_4h.index else None

        if current_1h_bar is None or current_4h_bar is None:
            debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, Upper timeframe bar is None (1H: {current_1h_bar is None}, 4H: {current_4h_bar is None})\n")
            continue

        # v60: 上位足の「壁」フィルター
        long_wall_condition = (current_1h_bar["close"] > current_1h_bar["ema20"] and current_1h_bar["ema20"] > current_1h_bar["ema75"]) and \
                              (current_4h_bar["close"] > current_4h_bar["ema20"] and current_4h_bar["ema20"] > current_4h_bar["ema75"])
        short_wall_condition = (current_1h_bar["close"] < current_1h_bar["ema20"] and current_1h_bar["ema20"] < current_1h_bar["ema75"]) and \
                               (current_4h_bar["close"] < current_4h_bar["ema20"] and current_4h_bar["ema20"] < current_4h_bar["ema75"])

        # 実体ベースの壁
        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low = min(current_m15_bar["open"], current_m15_bar["close"])
        debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, Open: {current_m15_bar['open']:.5f}, Close: {current_m15_bar['close']:.5f}, High: {current_m15_bar['high']:.5f}, Low: {current_m15_bar['low']:.5f}, Body High: {body_high:.5f}, Body Low: {body_low:.5f}, ATR: {atr_val:.5f}\n")

        # 髭の定義（ATRベース）
        long_condition = long_wall_condition and ((body_low - current_m15_bar["low"]) > (atr_val * 0.5))
        short_condition = short_wall_condition and ((current_m15_bar["high"] - body_high) > (atr_val * 0.5))
        long_wall_condition_str = str(long_wall_condition)
        short_wall_condition_str = str(short_wall_condition)
        long_wick_condition_str = str((body_low - current_m15_bar["low"]) > (atr_val * 0.5))
        short_wick_condition_str = str((current_m15_bar["high"] - body_high) > (atr_val * 0.5))
        timestamp_str = str(current_m15_bar.name)
        long_wick_value = body_low - current_m15_bar["low"]
        short_wick_value = current_m15_bar["high"] - body_high
        debug_signal_log.write(f"Timestamp: {str(current_m15_bar.name)}, Long Wall: {long_wall_condition_str}, Short Wall: {short_wall_condition_str}, Long Wick: {long_wick_condition_str}, Short Wick: {short_wick_condition_str}, Long Wick Value: {long_wick_value:.5f}, Short Wick Value: {short_wick_value:.5f}, ATR * 0.5: {(atr_val * 0.5):.5f}\n")

        if long_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                mid_point = current_m15_bar["low"] + (body_low - current_m15_bar["low"]) * 0.5
                if bar["low"] <= mid_point:
                    sl_price = bar["close"] - (atr_val * 3.0) # SLはエントリー価格からATRの3.0倍
                    risk = bar["close"] - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price # SLを先に設定
                        # N計算値ベースの利確
                        # N計算値は直近の波動（高値-安値）
                        n_value = current_m15_bar["high"] - current_m15_bar["low"]
                        # TPはエントリー価格からN計算値分
                        calculated_tp = bar["close"] + n_value
                        # RR 1:1以上を確保: リスク幅がN計算値より大きい場合は、リスク幅をTPとする
                        tp_series.loc[bar.name] = max(calculated_tp, bar["close"] + risk) 
                        entry_time_series.loc[bar.name] = bar.name # エントリー時間を記録
                        atr_at_entry_series.loc[bar.name] = atr_val # エントリー時のATRを記録
                        entry_price = bar["close"]
                        tp = tp_series.loc[bar.name]
                        sl = sl_series.loc[bar.name]
                        debug_signal_log.write(f"Timestamp: {str(bar.name)}, ATR: {atr_val:.5f}, Dynamic Volatility Threshold: {dynamic_volatility_threshold:.5f}, Long Wall: {long_wall_condition_str}, Short Wall: {short_wall_condition_str}, Long Wick: {long_wick_condition_str}, Short Wick: {short_wick_condition_str}, Signal Type: LONG, Entry: {entry_price:.5f}, TP: {tp:.5f}, SL: {sl:.5f}, Risk: {risk:.5f}, N_Value: {n_value:.5f}\n")
                        break


        elif short_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                mid_point = current_m15_bar["high"] - (current_m15_bar["high"] - body_high) * 0.5
                if bar["high"] >= mid_point:
                    sl_price = bar["close"] + (atr_val * 3.0) # SLはエントリー価格からATRの3.0倍
                    risk = sl_price - bar["close"]
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price # SLを先に設定
                        # N計算値ベースの利確
                        # N計算値は直近の波動（高値-安値）
                        n_value = current_m15_bar["high"] - current_m15_bar["low"]
                        # TPはエントリー価格からN計算値分
                        calculated_tp = bar["close"] - n_value
                        # RR 1:1以上を確保: リスク幅がN計算値より大きい場合は、リスク幅をTPとする
                        tp_series.loc[bar.name] = min(calculated_tp, bar["close"] - risk)
                        entry_time_series.loc[bar.name] = bar.name # エントリー時間を記録
                        atr_at_entry_series.loc[bar.name] = atr_val # エントリー時のATRを記録
                        entry_price = bar["close"]
                        tp = tp_series.loc[bar.name]
                        sl = sl_series.loc[bar.name]
                        debug_signal_log.write(f"Timestamp: {str(bar.name)}, ATR: {atr_val:.5f}, Dynamic Volatility Threshold: {dynamic_volatility_threshold:.5f}, Long Wall: {long_wall_condition_str}, Short Wall: {short_wall_condition_str}, Long Wick: {long_wick_condition_str}, Short Wick: {short_wick_condition_str}, Signal Type: SHORT, Entry: {entry_price:.5f}, TP: {tp:.5f}, SL: {sl:.5f}, Risk: {risk:.5f}, N_Value: {n_value:.5f}\n")
                        break


    debug_signal_log.close()
    return signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series
