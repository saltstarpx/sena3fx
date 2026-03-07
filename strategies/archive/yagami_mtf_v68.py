"""
Yagami MTF v68
==============
ベース: v58/v59の核心ロジックを忠実に継承
- 4時間足 EMA20 によるトレンド判定（壁）
- 15分足の下髭/上髭（ATRの0.5倍以上）を反発サインとして検出
- 1分足で髭の50%戻し地点まで引きつけてエントリー
- SL: 髭の先端 + ATR × 0.5 の余裕
- TP: リスク × 5.0 (v58/v59と同一)
- タイム・ディケイ: なし（水原様の指示により除外）
- ボラティリティフィルター: ATR < スプレッド×3 の極端な低ボラ時のみ除外（v59の設定を継承）
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
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    # 1時間足トレンド（補助: EMA20の向き）
    data_1h["ema20"] = data_1h["close"].ewm(span=20, adjust=False).mean()

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)
    entry_time_series = pd.Series(pd.NaT, index=data_1m.index)
    atr_at_entry_series = pd.Series(np.nan, index=data_1m.index)

    # v59継承: ボラティリティフィルター（スプレッドの3倍未満は除外）
    volatility_threshold = spread * 3.0

    for i in range(len(data_15m)):
        current_m15_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]

        if pd.isna(atr_val):
            atr_val = 0.0005  # v58/v59と同じフォールバック値

        # v59継承: 低ボラティリティ除外
        if atr_val < volatility_threshold:
            continue

        # 4時間足トレンド取得
        h4_time = current_m15_bar.name.floor("4h")
        if h4_time not in data_4h.index:
            continue
        current_4h_trend = data_4h.loc[h4_time]["trend"]

        # 1時間足トレンド取得（補助フィルタ: 4時間足と同方向のみ）
        h1_time = current_m15_bar.name.floor("1h")
        if h1_time not in data_1h.index:
            continue
        current_1h_close = data_1h.loc[h1_time]["close"]
        current_1h_ema20 = data_1h.loc[h1_time]["ema20"]

        # 4時間足と1時間足が同方向のみ（壁の強度確認）
        h1_aligned_long = (current_1h_close > current_1h_ema20)
        h1_aligned_short = (current_1h_close < current_1h_ema20)

        # 実体と髭の計算
        body_high = max(current_m15_bar["open"], current_m15_bar["close"])
        body_low = min(current_m15_bar["open"], current_m15_bar["close"])

        # v58/v59と同一: 髭の長さ > ATR × 0.5
        long_condition = (current_4h_trend == 1) and h1_aligned_long and \
                         ((body_low - current_m15_bar["low"]) > (atr_val * 0.5))
        short_condition = (current_4h_trend == -1) and h1_aligned_short and \
                          ((current_m15_bar["high"] - body_high) > (atr_val * 0.5))

        if long_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i + 1] if i + 1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]

            for _, bar in entry_1m_bars.iterrows():
                # v58/v59と同一: 髭の50%戻し地点
                mid_point = current_m15_bar["low"] + (body_low - current_m15_bar["low"]) * 0.5
                if bar["low"] <= mid_point:
                    entry_price = mid_point
                    # SL: 髭の先端 - ATR × 0.5（v59は ATR×1.0 だったが、先端基準に変更）
                    sl_price = current_m15_bar["low"] - (atr_val * 0.5)
                    risk = entry_price - sl_price
                    if risk > 0:
                        signal_series.loc[bar.name] = 1
                        sl_series.loc[bar.name] = sl_price
                        # v58/v59と同一: RR 5.0
                        tp_series.loc[bar.name] = entry_price + (risk * 5.0)
                        entry_time_series.loc[bar.name] = bar.name
                        atr_at_entry_series.loc[bar.name] = atr_val
                        break

        elif short_condition:
            start_1m_time = current_m15_bar.name
            end_1m_time = data_15m.index[i + 1] if i + 1 < len(data_15m) else data_1m.index[-1]
            entry_1m_bars = data_1m.loc[start_1m_time:end_1m_time]

            for _, bar in entry_1m_bars.iterrows():
                mid_point = current_m15_bar["high"] - (current_m15_bar["high"] - body_high) * 0.5
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
