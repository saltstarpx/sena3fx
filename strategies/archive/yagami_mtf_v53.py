import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def generate_signals(data_1m, data_15m, data_4h, spread=0.01):
    m15_atr = calculate_atr(data_15m)
    
    # 4時間足のトレンド判定
    data_4h['ema'] = data_4h['close'].ewm(span=20).mean()
    data_4h['trend'] = np.where(data_4h['close'] > data_4h['ema'], 1, -1)

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    signals_list = []

    for i in range(len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i] if not pd.isna(m15_atr.iloc[i]) else 0.05
        
        # 4時間足のトレンド確認
        current_4h_trend = data_4h.loc[:current_m15_bar.name].iloc[-1]['trend'] if not data_4h.loc[:current_m15_bar.name].empty else 0

        # 精度の追求：髭の定義を厳格化（ATRの0.5倍以上）
        # ロング：4H上昇トレンド中、15分足で下髭を確認
        long_condition = ((current_m15_bar["open"] - current_m15_bar["low"]) > (atr_val * 0.2))
        
        # ショート：4H下落トレンド中、15分足で上髭を確認
        short_condition = ((current_m15_bar["high"] - current_m15_bar["open"]) > (atr_val * 0.2))

        if long_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                # ミスプライス埋め：15分足の髭の安値（low）を1分足が試し、かつ陽線で反発
                if bar['low'] <= current_m15_bar['low']:
                    sl_price = current_m15_bar['low'] - (atr_val * 0.5)
                    risk = bar['close'] - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = bar['close'] + risk * 5.0
                        signals_list.append({"time": bar.name, "direction": "LONG"})
                        break

        elif short_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i+1] if i+1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]
            
            for _, bar in entry_1m_bars.iterrows():
                # ミスプライス埋め：15分足の髭の高値（high）を1分足が試し、かつ陰線で反発
                if bar['high'] >= current_m15_bar['high']:
                    sl_price = current_m15_bar['high'] + (atr_val * 0.5)
                    risk = sl_price - bar['close']
                    if risk > 0:
                        signal_series.loc[bar.name] = -1
                        sl_series.loc[bar.name] = sl_price
                        tp_series.loc[bar.name] = bar['close'] - risk * 5.0
                        signals_list.append({"time": bar.name, "direction": "SHORT"})
                        break

    return signal_series, tp_series, sl_series, signals_list
