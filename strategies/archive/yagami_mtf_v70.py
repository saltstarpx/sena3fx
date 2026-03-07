"""
Yagami MTF v70
==============
やがみnote準拠の「15分足更新タイミング成行エントリー」

変更点（v69からの差分）:
  - エントリー方法: 50%戻し待ち指値 → 次の15分足の始値で成行
  - エントリー価格: 次の15分足の open（＋スプレッド）
  - 損切り位置: 髭の先端（チャート構造上の意味ある場所）
  - RR: 5.0（v58/v59と同一）
  - 髭閾値: ATR×0.3（v69と同一）
  - 1時間足フィルター: なし（v69と同一）

やがみnote引用:
  「後1分待ってもし15分での初動が陽線なら反転する確率は上がる」
  「更新から2分くらいのイメージ。RRで計算すると5は軽く超える」
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

    # 4時間足トレンド（EMA20）
    data_4h = data_4h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    signal_series = pd.Series(0, index=data_1m.index)
    tp_series = pd.Series(np.nan, index=data_1m.index)
    sl_series = pd.Series(np.nan, index=data_1m.index)

    # ボラティリティフィルター（spread×3 の極端な低ボラのみ除外）
    volatility_threshold = spread * 3.0

    # 髭閾値: ATR×0.3
    WICK_MULT = 0.3

    for i in range(len(data_15m) - 1):  # 次の足が必要なので -1
        current_bar = data_15m.iloc[i]
        atr_val = m15_atr.iloc[i]

        if pd.isna(atr_val):
            continue
        if atr_val < volatility_threshold:
            continue

        # 4時間足トレンド確認
        h4_time = current_bar.name.floor("4h")
        if h4_time not in data_4h.index:
            continue
        trend = data_4h.loc[h4_time]["trend"]

        # 髭の計算
        body_high = max(current_bar["open"], current_bar["close"])
        body_low  = min(current_bar["open"], current_bar["close"])
        lower_wick = body_low - current_bar["low"]
        upper_wick = current_bar["high"] - body_high
        wick_threshold = atr_val * WICK_MULT

        long_ok  = (trend == 1)  and (lower_wick > wick_threshold)
        short_ok = (trend == -1) and (upper_wick > wick_threshold)

        if not long_ok and not short_ok:
            continue

        # ─── 次の15分足の始値でエントリー ───
        next_bar = data_15m.iloc[i + 1]
        next_bar_open_time = next_bar.name

        # 次の15分足の始値に対応する1分足バーを探す
        # （次の15分足の最初の1分足 = next_bar_open_time）
        if next_bar_open_time not in data_1m.index:
            # 1分足データに該当時刻がない場合はスキップ
            continue

        entry_1m_bar = data_1m.loc[next_bar_open_time]

        if long_ok:
            # ロング: 次の15分足の始値 + スプレッド（成行コスト）
            entry_price = entry_1m_bar["open"] + spread
            # 損切り: 髭の先端（チャート構造上の安値）
            sl_price = current_bar["low"] - (atr_val * 0.2)  # 少し余裕を持たせる
            risk = entry_price - sl_price
            if risk <= 0:
                continue
            tp_price = entry_price + (risk * 5.0)

            signal_series.loc[next_bar_open_time] = 1
            sl_series.loc[next_bar_open_time] = sl_price
            tp_series.loc[next_bar_open_time] = tp_price

        elif short_ok:
            # ショート: 次の15分足の始値 - スプレッド（成行コスト）
            entry_price = entry_1m_bar["open"] - spread
            # 損切り: 髭の先端（チャート構造上の高値）
            sl_price = current_bar["high"] + (atr_val * 0.2)
            risk = sl_price - entry_price
            if risk <= 0:
                continue
            tp_price = entry_price - (risk * 5.0)

            signal_series.loc[next_bar_open_time] = -1
            sl_series.loc[next_bar_open_time] = sl_price
            tp_series.loc[next_bar_open_time] = tp_price

    return signal_series, tp_series, sl_series, pd.Series(pd.NaT, index=data_1m.index), pd.Series(np.nan, index=data_1m.index)
