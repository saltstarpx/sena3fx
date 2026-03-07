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
    spread_value = spread_pips * 0.01 # 0.2pipsの場合、0.002円
    m15_atr = calculate_atr(data_15m)
    
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

    # ボラティリティフィルターの閾値 (例: スプレッドの3倍)
    volatility_threshold = spread_value * 0.5

    debug_signal_log = open("/home/ubuntu/sena3fx/results/yagami_mtf_v61_signal_debug.log", "w")
    debug_signal_log.write("Timestamp,ATR,Volatility_Threshold,Long_Wall,Short_Wall,Long_Wick,Short_Wick,Signal_Type,Entry,TP,SL,Risk,N_Value\n")
    for i in range(len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i] if not pd.isna(m15_atr.iloc[i]) else 0.0005
        debug_signal_log.write(f"[{current_m15_bar.name}] ATR: {atr_val:.5f}, Volatility Threshold: {volatility_threshold:.5f}\n")

        
        # v59: ボラティリティフィルター
        if atr_val < volatility_threshold:
            debug_signal_log.write(f"[{current_m15_bar.name}] Volatility too low: ATR={atr_val:.5f}, Threshold={volatility_threshold:.5f}\n")
            continue # ボラティリティが低い場合はスキップ



        # 上位足のトレンド確認
        current_1h_bar = data_1h.loc[:current_m15_bar.name].iloc[-1] if not data_1h.loc[:current_m15_bar.name].empty else None
        current_4h_bar = data_4h.loc[:current_m15_bar.name].iloc[-1] if not data_4h.loc[:current_m15_bar.name].empty else None

        if current_1h_bar is None or current_4h_bar is None:
            debug_signal_log.write(f"[{current_m15_bar.name}] Upper timeframe bar is None (1H: {current_1h_bar is None}, 4H: {current_4h_bar is None})\n")
            continue

        # v60: 上位足の「壁」フィルター
        long_wall_condition = (current_1h_bar["close"] > current_1h_bar["ema20"] and current_1h_bar["ema20"] > current_1h_bar["ema75"]) and \
                              (current_4h_bar["close"] > current_4h_bar["ema20"] and current_4h_bar["ema20"] > current_4h_bar["ema75"])
        short_wall_condition = (current_1h_bar["close"] < current_1h_bar["ema20"] and current_1h_bar["ema20"] < current_1h_bar["ema75"]) and \
                               (current_4h_bar["close"] < current_4h_bar["ema20"] and current_4h_bar["ema20"] < current_4h_bar["ema75"])

        # 実体ベースの壁
        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low = min(current_m15_bar["open"], current_m15_bar["close"])

        # 髭の定義（ATRベース）
        long_condition = long_wall_condition and ((body_low - current_m15_bar["low"]) > (atr_val * 0.5))
        short_condition = short_wall_condition and ((current_m15_bar["high"] - body_high) > (atr_val * 0.5))
        long_wall_condition_str = str(long_wall_condition)
        short_wall_condition_str = str(short_wall_condition)
        long_wick_condition_str = str((body_low - current_m15_bar["low"]) > (atr_val * 0.5))
        short_wick_condition_str = str((current_m15_bar["high"] - body_high) > (atr_val * 0.5))
        debug_signal_log.write(f"[{current_m15_bar.name}] Long Wall: {long_wall_condition_str}, Short Wall: {short_wall_condition_str}, Long Wick: {long_wick_condition_str}, Short Wick: {short_wick_condition_str}\n")

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
                        sl_series.loc[bar.name] = sl_price
                        # N計算値ベースの利確
                        n_value = current_m15_bar["high"] - current_m15_bar["low"]
                        calculated_tp = bar["close"] + n_value
                        tp_series.loc[bar.name] = max(calculated_tp, bar["close"] + risk) # RR 1:1以上を確保
                        entry_time_series.loc[bar.name] = bar.name # エントリー時間を記録
                        atr_at_entry_series.loc[bar.name] = atr_val # エントリー時のATRを記録
                        entry_price = bar["close"]
                        tp = tp_series.loc[bar.name]
                        sl = sl_series.loc[bar.name]
                        debug_signal_log.write(f"{bar.name},{atr_val:.5f},{volatility_threshold:.5f},{long_wall_condition_str},{short_wall_condition_str},{long_wick_condition_str},{short_wick_condition_str},LONG,{entry_price:.5f},{tp:.5f},{sl:.5f},{risk:.5f},{n_value:.5f}\n")


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
                        sl_series.loc[bar.name] = sl_price
                        # N計算値ベースの利確
                        n_value = current_m15_bar["high"] - current_m15_bar["low"]
                        calculated_tp = bar["close"] - n_value
                        tp_series.loc[bar.name] = min(calculated_tp, bar["close"] - risk) # RR 1:1以上を確保
                        entry_time_series.loc[bar.name] = bar.name # エントリー時間を記録
                        atr_at_entry_series.loc[bar.name] = atr_val # エントリー時のATRを記録
                        entry_price = bar["close"]
                        tp = tp_series.loc[bar.name]
                        sl = sl_series.loc[bar.name]
                        debug_signal_log.write(f"{bar.name},{atr_val:.5f},{volatility_threshold:.5f},{long_wall_condition_str},{short_wall_condition_str},{long_wick_condition_str},{short_wick_condition_str},SHORT,{entry_price:.5f},{tp:.5f},{sl:.5f},{risk:.5f},{n_value:.5f}\n")
                        break


    debug_signal_log.close()
    return signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series
