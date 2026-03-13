"""
スイングハイ / スイングロー 検出モジュール
==========================================
やがみメソッド準拠:
  - ロング時SL  → 直近の押し安値（スイングロー）を4H足から取得
  - ショート時SL → 直近の戻り高値（スイングハイ）を4H足から取得

スイングの定義:
  ある足の安値 / 高値が 前後 window 本より低い / 高い ピボットポイント

使用例:
  >>> bars_4h = broker.get_candles('XAU_USD', 'H4', count=200)
  >>> sl = get_sl_for_long(bars_4h, window=3, lookback=10)
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  ピボット検出 (ベクトル化)                                          #
# ------------------------------------------------------------------ #

def find_pivot_lows(bars: pd.DataFrame, window: int = 3) -> pd.Series:
    """
    スイングロー（押し安値）を検出する。

    ある足の安値が前後 window 本の安値より低い場合をスイングローとする。
    注: 最新 window 本はリアルタイムでは未確定なので除外される。

    Args:
        bars:   OHLCデータ
        window: ピボット判定ウィンドウ本数 (推奨: 3〜5)

    Returns:
        pd.Series[bool]: スイングローの位置 (True = スイングロー)
    """
    low = bars['low']
    n = len(low)
    pivot = pd.Series(False, index=bars.index)

    for i in range(window, n - window):
        center = low.iloc[i]
        left   = low.iloc[i - window: i].values
        right  = low.iloc[i + 1: i + window + 1].values
        if (center < left).all() and (center < right).all():
            pivot.iloc[i] = True

    return pivot


def find_pivot_highs(bars: pd.DataFrame, window: int = 3) -> pd.Series:
    """
    スイングハイ（戻り高値）を検出する。

    Args:
        bars:   OHLCデータ
        window: ピボット判定ウィンドウ本数 (推奨: 3〜5)

    Returns:
        pd.Series[bool]: スイングハイの位置 (True = スイングハイ)
    """
    high = bars['high']
    n = len(high)
    pivot = pd.Series(False, index=bars.index)

    for i in range(window, n - window):
        center = high.iloc[i]
        left   = high.iloc[i - window: i].values
        right  = high.iloc[i + 1: i + window + 1].values
        if (center > left).all() and (center > right).all():
            pivot.iloc[i] = True

    return pivot


# ------------------------------------------------------------------ #
#  ライブトレード用: 最新のスイングSLを返す                           #
# ------------------------------------------------------------------ #

def get_sl_for_long(bars_4h: pd.DataFrame,
                     window: int = 3,
                     lookback: int = 10) -> float:
    """
    ロングポジションの損切り価格（直近の押し安値）を返す。

    最近 lookback 本の4H足からスイングローを探す。
    見つからない場合は直近の最安値をフォールバックとして使用。

    Args:
        bars_4h:  4H足OHLCデータ
        window:   ピボット判定ウィンドウ (推奨: 3)
        lookback: 最大遡り本数 (推奨: 10〜15)

    Returns:
        float: SL価格

    Example:
        >>> sl = get_sl_for_long(bars, window=3, lookback=10)
        >>> units = (balance * 0.02) / (entry_price - sl)
    """
    # 最近 (lookback + window*2) 本のサブセットで検索
    search_n = lookback + window * 2
    recent = bars_4h.iloc[-search_n:] if len(bars_4h) > search_n else bars_4h

    pivot_mask = find_pivot_lows(recent, window)
    swing_lows = recent.loc[pivot_mask, 'low']

    if not swing_lows.empty:
        return float(swing_lows.iloc[-1])  # 最新のスイングロー

    # フォールバック: 直近 lookback 本の最安値
    return float(bars_4h['low'].iloc[-lookback:].min())


def get_sl_for_short(bars_4h: pd.DataFrame,
                      window: int = 3,
                      lookback: int = 10) -> float:
    """
    ショートポジションの損切り価格（直近の戻り高値）を返す。

    Args:
        bars_4h:  4H足OHLCデータ
        window:   ピボット判定ウィンドウ (推奨: 3)
        lookback: 最大遡り本数 (推奨: 10〜15)

    Returns:
        float: SL価格
    """
    search_n = lookback + window * 2
    recent = bars_4h.iloc[-search_n:] if len(bars_4h) > search_n else bars_4h

    pivot_mask = find_pivot_highs(recent, window)
    swing_highs = recent.loc[pivot_mask, 'high']

    if not swing_highs.empty:
        return float(swing_highs.iloc[-1])  # 最新のスイングハイ

    # フォールバック: 直近 lookback 本の最高値
    return float(bars_4h['high'].iloc[-lookback:].max())


# ------------------------------------------------------------------ #
#  バックテスト用: SL価格 Series を一括計算                           #
# ------------------------------------------------------------------ #

def build_swing_sl_series_long(bars_4h: pd.DataFrame,
                                 window: int = 3,
                                 lookback: int = 10) -> pd.Series:
    """
    バックテスト用: 各4H足バーでのロングSL価格を計算する (高速版)。

    ピボットローを一度検出してフォワードフィルすることでO(n)で実現。

    Returns:
        pd.Series[float]: SL価格 (NaN = 未確定)
    """
    pivot_mask = find_pivot_lows(bars_4h, window)

    # ピボット位置の安値のみ設定 → forward fill
    sl_series = pd.Series(np.nan, index=bars_4h.index)
    sl_series[pivot_mask] = bars_4h.loc[pivot_mask, 'low']
    sl_series = sl_series.ffill()

    # フォールバック: rolling min で補完
    fallback = bars_4h['low'].rolling(lookback).min().shift(1)
    sl_series = sl_series.fillna(fallback)

    return sl_series


def build_swing_sl_series_short(bars_4h: pd.DataFrame,
                                  window: int = 3,
                                  lookback: int = 10) -> pd.Series:
    """
    バックテスト用: 各4H足バーでのショートSL価格を計算する (高速版)。

    Returns:
        pd.Series[float]: SL価格 (NaN = 未確定)
    """
    pivot_mask = find_pivot_highs(bars_4h, window)

    sl_series = pd.Series(np.nan, index=bars_4h.index)
    sl_series[pivot_mask] = bars_4h.loc[pivot_mask, 'high']
    sl_series = sl_series.ffill()

    fallback = bars_4h['high'].rolling(lookback).max().shift(1)
    sl_series = sl_series.fillna(fallback)

    return sl_series
