"""
やがみプライスアクション戦略 - シグナル関数群
==============================================
Note「ローソク足の本」「ローソク足の本2」に基づいて実装。

実装するパターン:
  PA1: リバーサルロー / リバーサルハイ（最強パターン）
  PA2: 安値圏ピンバー / 高値圏ピンバー
  PA3: 包み足（前回陰線を包む陽線 / 前回陽線を包む陰線）
  PA4: 複合（PA1 + PA2 + PA3 の OR）
  PA5: 複合 + スウィングゾーンフィルター強化版

各関数は BacktestEngine.run() の signal_func として使用。
戻り値: pd.Series of 'long' / 'short' / None
"""
import numpy as np
import pandas as pd


def _calc_atr(bars, period=14):
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars.index).rolling(period).mean()


def _swing_zone(bars, lookback=20):
    """直近lookback本のスウィングゾーン（高値・安値）を返す。"""
    rolling_high = bars['high'].rolling(lookback).max()
    rolling_low = bars['low'].rolling(lookback).min()
    return rolling_high, rolling_low


def signal_pa1_reversal(bars, atr_period=14, zone_atr=1.5, lookback=20):
    """
    PA1: リバーサルロー / リバーサルハイ
    
    ロング条件:
      - 直近lookback本の最安値付近（zone_atr×ATR以内）にいる
      - 前回足が大陰線（実体 > ATR×0.6）
      - 今回足が陽線で、前回安値を下回っていない（リバーサル）
    
    ショート条件:
      - 直近lookback本の最高値付近（zone_atr×ATR以内）にいる
      - 前回足が大陽線（実体 > ATR×0.6）
      - 今回足が陰線で、前回高値を上回っていない（リバーサル）
    """
    atr = _calc_atr(bars, atr_period)
    swing_high, swing_low = _swing_zone(bars, lookback)
    signals = pd.Series(index=bars.index, dtype=object)

    for i in range(2, len(bars)):
        bar = bars.iloc[i]
        prev = bars.iloc[i - 1]
        a = atr.iloc[i]
        if np.isnan(a) or a == 0:
            continue

        # 前回足の実体
        prev_body = abs(prev['close'] - prev['open'])
        cur_body = abs(bar['close'] - bar['open'])

        # ロング: 安値圏でリバーサルロー
        sw_low = swing_low.iloc[i]
        if not np.isnan(sw_low):
            in_low_zone = bar['low'] <= sw_low + zone_atr * a
            prev_big_bear = (prev['close'] < prev['open']) and (prev_body > a * 0.6)
            cur_bull = bar['close'] > bar['open']
            no_new_low = bar['low'] >= prev['low'] - a * 0.3  # 前回安値を大きく下回らない
            if in_low_zone and prev_big_bear and cur_bull and no_new_low:
                signals.iloc[i] = 'long'
                continue

        # ショート: 高値圏でリバーサルハイ
        sw_high = swing_high.iloc[i]
        if not np.isnan(sw_high):
            in_high_zone = bar['high'] >= sw_high - zone_atr * a
            prev_big_bull = (prev['close'] > prev['open']) and (prev_body > a * 0.6)
            cur_bear = bar['close'] < bar['open']
            no_new_high = bar['high'] <= prev['high'] + a * 0.3
            if in_high_zone and prev_big_bull and cur_bear and no_new_high:
                signals.iloc[i] = 'short'

    return signals


def signal_pa2_pinbar(bars, atr_period=14, zone_atr=1.5, lookback=20,
                      wick_body_ratio=2.0, min_wick_atr=0.3):
    """
    PA2: 安値圏ピンバー / 高値圏ピンバー
    
    ロング条件（下ヒゲピンバー）:
      - 直近lookback本の最安値付近にいる
      - 下ヒゲ > 実体 × wick_body_ratio
      - 下ヒゲ > ATR × min_wick_atr
      - 実体が陽線（または小さい陰線）
    
    ショート条件（上ヒゲピンバー）:
      - 直近lookback本の最高値付近にいる
      - 上ヒゲ > 実体 × wick_body_ratio
      - 上ヒゲ > ATR × min_wick_atr
    """
    atr = _calc_atr(bars, atr_period)
    swing_high, swing_low = _swing_zone(bars, lookback)
    signals = pd.Series(index=bars.index, dtype=object)

    for i in range(1, len(bars)):
        bar = bars.iloc[i]
        a = atr.iloc[i]
        if np.isnan(a) or a == 0:
            continue

        body = abs(bar['close'] - bar['open'])
        lower_wick = min(bar['open'], bar['close']) - bar['low']
        upper_wick = bar['high'] - max(bar['open'], bar['close'])

        # ロング: 安値圏での下ヒゲピンバー
        sw_low = swing_low.iloc[i]
        if not np.isnan(sw_low):
            in_low_zone = bar['low'] <= sw_low + zone_atr * a
            long_lower_wick = lower_wick > body * wick_body_ratio
            wick_big_enough = lower_wick > a * min_wick_atr
            if in_low_zone and long_lower_wick and wick_big_enough:
                signals.iloc[i] = 'long'
                continue

        # ショート: 高値圏での上ヒゲピンバー
        sw_high = swing_high.iloc[i]
        if not np.isnan(sw_high):
            in_high_zone = bar['high'] >= sw_high - zone_atr * a
            long_upper_wick = upper_wick > body * wick_body_ratio
            wick_big_enough = upper_wick > a * min_wick_atr
            if in_high_zone and long_upper_wick and wick_big_enough:
                signals.iloc[i] = 'short'

    return signals


def signal_pa3_engulf(bars, atr_period=14, zone_atr=2.0, lookback=20):
    """
    PA3: 包み足（前回陰線を包む陽線 / 前回陽線を包む陰線）
    
    ロング条件:
      - 直近lookback本の最安値付近にいる
      - 今回の陽線実体が前回の陰線実体を完全に包む
    
    ショート条件:
      - 直近lookback本の最高値付近にいる
      - 今回の陰線実体が前回の陽線実体を完全に包む
    """
    atr = _calc_atr(bars, atr_period)
    swing_high, swing_low = _swing_zone(bars, lookback)
    signals = pd.Series(index=bars.index, dtype=object)

    for i in range(1, len(bars)):
        bar = bars.iloc[i]
        prev = bars.iloc[i - 1]
        a = atr.iloc[i]
        if np.isnan(a) or a == 0:
            continue

        cur_open = bar['open']
        cur_close = bar['close']
        prev_open = prev['open']
        prev_close = prev['close']

        # ロング: 安値圏で包み足陽線
        sw_low = swing_low.iloc[i]
        if not np.isnan(sw_low):
            in_low_zone = bar['low'] <= sw_low + zone_atr * a
            prev_bear = prev_close < prev_open
            cur_bull = cur_close > cur_open
            engulf = cur_open <= prev_close and cur_close >= prev_open
            if in_low_zone and prev_bear and cur_bull and engulf:
                signals.iloc[i] = 'long'
                continue

        # ショート: 高値圏で包み足陰線
        sw_high = swing_high.iloc[i]
        if not np.isnan(sw_high):
            in_high_zone = bar['high'] >= sw_high - zone_atr * a
            prev_bull = prev_close > prev_open
            cur_bear = cur_close < cur_open
            engulf = cur_open >= prev_close and cur_close <= prev_open
            if in_high_zone and prev_bull and cur_bear and engulf:
                signals.iloc[i] = 'short'

    return signals


def signal_pa4_combined(bars, atr_period=14, zone_atr=1.5, lookback=20):
    """
    PA4: PA1 + PA2 + PA3 の複合シグナル（OR条件）
    優先順位: PA1 > PA2 > PA3
    """
    s1 = signal_pa1_reversal(bars, atr_period, zone_atr, lookback)
    s2 = signal_pa2_pinbar(bars, atr_period, zone_atr, lookback)
    s3 = signal_pa3_engulf(bars, atr_period, zone_atr, lookback)

    combined = pd.Series(index=bars.index, dtype=object)
    for i in range(len(bars)):
        if s1.iloc[i] in ['long', 'short']:
            combined.iloc[i] = s1.iloc[i]
        elif s2.iloc[i] in ['long', 'short']:
            combined.iloc[i] = s2.iloc[i]
        elif s3.iloc[i] in ['long', 'short']:
            combined.iloc[i] = s3.iloc[i]
    return combined


def signal_pa5_combined_strict(bars, atr_period=14, zone_atr=1.2, lookback=20):
    """
    PA5: 厳格版複合シグナル
    - ゾーンフィルターを厳しく（zone_atr=1.2）
    - PA1 + PA2 のみ（包み足は除外）
    - 連続シグナルを避けるため、前回シグナルから3本以上経過した場合のみ
    """
    s1 = signal_pa1_reversal(bars, atr_period, zone_atr, lookback)
    s2 = signal_pa2_pinbar(bars, atr_period, zone_atr, lookback)

    combined = pd.Series(index=bars.index, dtype=object)
    last_signal_i = -10

    for i in range(len(bars)):
        sig = None
        if s1.iloc[i] in ['long', 'short']:
            sig = s1.iloc[i]
        elif s2.iloc[i] in ['long', 'short']:
            sig = s2.iloc[i]

        if sig and (i - last_signal_i) >= 3:
            combined.iloc[i] = sig
            last_signal_i = i

    return combined


# TP計算用ヘルパー（直近高値・安値）
def calc_swing_tp(bars, idx, direction, lookback=20):
    """
    直近スウィングハイ/ローをTPとして返す。
    ロング: 直近lookback本の最高値
    ショート: 直近lookback本の最安値
    """
    start = max(0, idx - lookback)
    if direction == 'long':
        return bars['high'].iloc[start:idx].max()
    else:
        return bars['low'].iloc[start:idx].min()
