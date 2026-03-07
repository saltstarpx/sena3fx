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
    m15_atr = calculate_atr(data_15m)
    
    # v58ベース: 4時間足のトレンド判定 (EMA20)
    data_4h['ema20'] = data_4h['close'].ewm(span=20, adjust=False).mean()
    data_4h['trend'] = np.where(data_4h['close'] > data_4h['ema20'], 1, -1)
    
    # 1時間足のトレンド判定 (補助フィルタとして使用)
    data_1h['ema20'] = data_1h['close'].ewm(span=20, adjust=False).mean()

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    entry_time_series = pd.Series(pd.NaT, index=data_1m.index)
    atr_at_entry_series = pd.Series(np.nan, index=data_1m.index)

    # v65: エントリー回数を確保するため、ボラティリティフィルターを緩和 (0.7倍)
    atr_ma = m15_atr.rolling(window=24).mean()
    volatility_factor = 0.7 
    
    for i in range(24, len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]
        current_atr_ma = atr_ma.iloc[i]

        if pd.isna(atr_val) or pd.isna(current_atr_ma):
            continue

        # 低ボラティリティ時は避けるが、閾値は低めに設定
        if atr_val < (current_atr_ma * volatility_factor):
            continue

        # 4時間足のトレンド取得
        h4_time = current_m15_bar.name.floor("4H")
        if h4_time not in data_4h.index:
            continue
        current_4h_trend = data_4h.loc[h4_time]['trend']

        # 1時間足のトレンド取得 (補助)
        h1_time = current_m15_bar.name.floor("1H")
        if h1_time not in data_1h.index:
            continue
        current_1h_ema = data_1h.loc[h1_time]['ema20']
        current_1h_close = data_1h.loc[h1_time]['close']

        # v58の「壁」ロジック: 4時間足のトレンドに順張り
        # さらに1時間足でもEMAの上に価格があるか（ロングの場合）を確認
        long_wall = (current_4h_trend == 1) and (current_1h_close > current_1h_ema)
        short_wall = (current_4h_trend == -1) and (current_1h_close < current_1h_ema)

        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low = min(current_m15_bar["open"], current_m15_bar["close"])
        
        # v58の「髭」定義: ATRの0.5倍以上
        long_wick = (body_low - current_m15_bar["low"]) > (atr_val * 0.5)
        short_wick = (current_m15_bar["high"] - body_high) > (atr_val * 0.5)

        if long_wall and long_wick:
            # v58の「50%戻し」エントリー
            mid_point = current_m15_bar["low"] + (body_low - current_m15_bar["low"]) * 0.5
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                if bar["low"] <= mid_point:
                    entry_price = mid_point
                    # v58のSL: 髭の安値からATR分下
                    sl_price = current_m15_bar['low'] - (atr_val * 0.5)
                    risk = entry_price - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        # TPはv58のRR 5.0から、より現実的な3.0に調整して安定化
                        tp_series.loc[bar.name] = entry_price + (risk * 3.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        break

        elif short_wall and short_wick:
            # v58の「50%戻し」エントリー
            mid_point = current_m15_bar["high"] - (current_m15_bar["high"] - body_high) * 0.5
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                if bar["high"] >= mid_point:
                    entry_price = mid_point
                    # v58のSL: 髭の高値からATR分上
                    sl_price = current_m15_bar['high'] + (atr_val * 0.5)
                    risk = sl_price - entry_price
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = entry_price - (risk * 3.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        break

    return signal_series, tp_series, sl_series, entry_time_series, atr_at_entry_series
