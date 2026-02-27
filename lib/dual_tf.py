"""
デュアルタイムフレーム (4H × 15m) シグナルエンジン
====================================================
実装ルール:
  1. トレンド判断 → 4時間足 (EMA21 方向)
  2. エントリータイミング → 15分足 (DC ブレイクアウト)
  3. 4H足の実体確定後のみシグナル有効 (確定済みバーのみ参照)
  4. 順張りのみ (4H上昇 → ロングのみ / 4H下降 → ショートのみ)
  5. 1分・5分・1時間足は参照しない

設計思想:
  「4H足が確定してから15分足でエントリーを狙う」
  4H足のブレイク/トレンド確認 → 15m足がその方向にブレイクした瞬間にエントリー
"""

import pandas as pd
import numpy as np
from typing import Optional


BARS_PER_DAY = {
    '15min': 96, '15T': 96, 'M15': 96,
    '1h':    24, '1H': 24,  'H1': 24,
    '4h':     6, '4H': 6,   'H4': 6,
    '8h':     3, '8H': 3,
    '1d':     1, '1D': 1,   'D': 1,
}


# ------------------------------------------------------------------ #
#  4H トレンド判定                                                    #
# ------------------------------------------------------------------ #

def get_4h_trend(bars_4h: pd.DataFrame, ema_days: int = 21) -> str:
    """
    4H足のトレンド方向を EMA で判定する。

    判定ロジック:
      - 上昇トレンド: 終値 > EMA21 かつ EMA21 が上向き (3本前より高い)
      - 下降トレンド: 終値 < EMA21 かつ EMA21 が下向き (3本前より低い)
      - ニュートラル: 上記以外

    Args:
        bars_4h:  完成済みの4H足OHLCデータ
        ema_days: EMA日数 (21日 = 4H換算で126本)

    Returns:
        'up' | 'down' | 'neutral'
    """
    ema_n = ema_days * BARS_PER_DAY.get('4h', 6)
    close = bars_4h['close']
    ema   = close.ewm(span=ema_n, adjust=False).mean()

    last_close = close.iloc[-1]
    last_ema   = ema.iloc[-1]
    # EMAの傾き: 3本前との差で判断 (短期ノイズを除去)
    ema_slope  = ema.iloc[-1] - ema.iloc[-min(4, len(ema))]

    if last_close > last_ema and ema_slope > 0:
        return 'up'
    elif last_close < last_ema and ema_slope < 0:
        return 'down'
    return 'neutral'


def compute_4h_trend_series(bars_4h: pd.DataFrame,
                              ema_days: int = 21) -> pd.Series:
    """
    バックテスト用: 全4H足バーのトレンド Series を計算。

    Returns:
        pd.Series: 'up' | 'down' | 'neutral'
    """
    ema_n = ema_days * BARS_PER_DAY.get('4h', 6)
    close     = bars_4h['close']
    ema       = close.ewm(span=ema_n, adjust=False).mean()
    ema_slope = ema.diff(3)

    trend = pd.Series('neutral', index=bars_4h.index, dtype=object)
    trend[(close > ema) & (ema_slope > 0)] = 'up'
    trend[(close < ema) & (ema_slope < 0)] = 'down'

    return trend


# ------------------------------------------------------------------ #
#  15m エントリーシグナル                                             #
# ------------------------------------------------------------------ #

def get_15m_signal(bars_15m: pd.DataFrame,
                    trend: str,
                    dc_lookback: int = 20) -> Optional[str]:
    """
    15m足でエントリーシグナルを確認する。
    完成済みの最新バーのみを対象とする。

    ルール:
      - トレンド 'up'   → 直近 dc_lookback 本の高値を終値が上抜け → 'long'
      - トレンド 'down' → 直近 dc_lookback 本の安値を終値が下抜け → 'short'
      - 'neutral'       → シグナルなし

    Args:
        bars_15m:    完成済み15m足OHLCデータ
        trend:       4H足トレンド ('up'|'down'|'neutral')
        dc_lookback: ドンチャン期間 (本数, デフォルト20本=5時間)

    Returns:
        'long' | 'short' | None
    """
    if trend == 'neutral':
        return None
    if len(bars_15m) < dc_lookback + 2:
        return None

    close = bars_15m['close']
    high  = bars_15m['high']
    low   = bars_15m['low']

    # 前バーまでの最高値・最安値 (現在バーを除く → 実体確定後ルール)
    dc_hi = high.shift(1).rolling(dc_lookback).max()
    dc_lo = low.shift(1).rolling(dc_lookback).min()

    last_close = close.iloc[-1]
    last_dc_hi = dc_hi.iloc[-1]
    last_dc_lo = dc_lo.iloc[-1]

    if pd.isna(last_dc_hi) or pd.isna(last_dc_lo):
        return None

    if trend == 'up'   and last_close > last_dc_hi:
        return 'long'
    if trend == 'down' and last_close < last_dc_lo:
        return 'short'

    return None


# ------------------------------------------------------------------ #
#  バックテスト用: 15m シグナルを一括ベクトル計算                     #
# ------------------------------------------------------------------ #

def compute_15m_signals_vectorized(bars_15m: pd.DataFrame,
                                    trend_at_15m: pd.Series,
                                    dc_lookback: int = 20) -> pd.Series:
    """
    バックテスト用: 全15m足バーのシグナルを一括計算 (ベクトル化)。

    Args:
        bars_15m:      15m足OHLCデータ
        trend_at_15m:  15m足インデックスにアラインされた4H足トレンド Series
        dc_lookback:   ドンチャン期間

    Returns:
        pd.Series: 'long' | 'short' | None (object dtype)
    """
    close = bars_15m['close']
    high  = bars_15m['high']
    low   = bars_15m['low']

    # 前バーまでの DC チャンネル (実体確定後ルール: shift(1))
    dc_hi = high.shift(1).rolling(dc_lookback).max()
    dc_lo = low.shift(1).rolling(dc_lookback).min()

    signals = pd.Series(None, index=bars_15m.index, dtype=object)

    # ロング: 4H上昇トレンド かつ 15m DC高値ブレイク
    long_mask  = (trend_at_15m == 'up')   & (close > dc_hi)
    # ショート: 4H下降トレンド かつ 15m DC安値ブレイク
    short_mask = (trend_at_15m == 'down') & (close < dc_lo)

    signals[long_mask]  = 'long'
    signals[short_mask] = 'short'

    return signals


def align_4h_trend_to_15m(bars_4h: pd.DataFrame,
                            bars_15m: pd.DataFrame,
                            ema_days: int = 21) -> pd.Series:
    """
    4H足のトレンドを15m足のインデックスにアラインする (バックテスト用)。

    4H足確定後の次の15m足から新しいトレンドを適用する。
    (4H足確定前のブレイクを無効化するルールの実装)

    前提: bars_4h のインデックスは4H足確定時刻 (UTC)。

    Returns:
        pd.Series: 15mインデックスでのトレンド ('up'|'down'|'neutral')
    """
    # 4H足のトレンド計算 (確定バーのみ)
    trend_4h = compute_4h_trend_series(bars_4h, ema_days)

    # 15m インデックスに reindex → forward fill
    # ffill: 4H足が確定した瞬間から次の4H足確定まで同じトレンドを使用
    trend_aligned = trend_4h.reindex(
        trend_4h.index.union(bars_15m.index)
    ).ffill()

    # 15m インデックスのみ抽出
    trend_aligned = trend_aligned.reindex(bars_15m.index)

    return trend_aligned.fillna('neutral')
