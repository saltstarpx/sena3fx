"""
方向B: EMAトレンド型 4時間足フィルター
=========================================
PA1_Reversal_TightSL に4時間足のEMAトレンド方向フィルターを追加する。

RUN-002との違い:
  - RUN-002: 4時間足のスウィングゾーン（安値圏/高値圏）でフィルター
    → 円安トレンド中にリバーサル型フィルターが機能しなかった
  - RUN-003方向B: 4時間足のEMA傾きでトレンド方向を判定
    → トレンド方向にのみエントリーを許可（トレンドフォロー型）

フィルター条件:
  ロングシグナル: 4時間足 EMA_fast > EMA_slow（上昇トレンド）
  ショートシグナル: 4時間足 EMA_fast < EMA_slow（下降トレンド）

オプション:
  - EMAの傾き（slope）が一定以上の場合のみ許可（トレンド強度フィルター）
  - EMAとの乖離率フィルター（乖離しすぎている場合は除外）
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.yagami_pa import signal_pa1_reversal
from strategies.htf_filter import align_htf_to_ltf


def build_ema_trend_filter(bars_4h, fast=20, slow=50, slope_period=3, min_slope=0.0):
    """
    4時間足EMAトレンドフィルターを構築する。

    Parameters
    ----------
    bars_4h : pd.DataFrame
        4時間足OHLC
    fast : int
        短期EMA期間
    slow : int
        長期EMA期間
    slope_period : int
        EMA傾き計算期間（本数）
    min_slope : float
        最小傾き閾値（pips/本）。0.0で傾き制限なし

    Returns
    -------
    pd.DataFrame
        カラム: ['long_ok', 'short_ok', 'ema_fast', 'ema_slow', 'ema_slope']
    """
    bars = bars_4h.copy()
    ema_fast = bars['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = bars['close'].ewm(span=slow, adjust=False).mean()

    # EMA傾き（fast EMAのslope_period本前との差）
    ema_slope = ema_fast - ema_fast.shift(slope_period)

    long_ok = pd.Series(False, index=bars.index)
    short_ok = pd.Series(False, index=bars.index)

    for i in range(slow, len(bars)):
        ef = ema_fast.iloc[i]
        es = ema_slow.iloc[i]
        slope = ema_slope.iloc[i]

        if np.isnan(ef) or np.isnan(es) or np.isnan(slope):
            continue

        # ロング許可: EMA fast > slow かつ傾きが正
        if ef > es and slope >= min_slope:
            long_ok.iloc[i] = True

        # ショート許可: EMA fast < slow かつ傾きが負
        if ef < es and slope <= -min_slope:
            short_ok.iloc[i] = True

    return pd.DataFrame({
        'long_ok': long_ok,
        'short_ok': short_ok,
        'ema_fast': ema_fast,
        'ema_slow': ema_slow,
        'ema_slope': ema_slope,
    })


def signal_direction_b(
    bars_1h,
    bars_4h,
    # PA1ベースパラメータ
    zone_atr=1.5,
    lookback=20,
    # EMAパラメータ
    ema_fast=20,
    ema_slow=50,
    slope_period=3,
    min_slope=0.0,
):
    """
    方向Bシグナル生成関数。

    Parameters
    ----------
    bars_1h : pd.DataFrame
        1時間足OHLCV
    bars_4h : pd.DataFrame
        4時間足OHLCV
    zone_atr : float
        PA1のゾーン幅
    lookback : int
        PA1のルックバック
    ema_fast : int
        4時間足短期EMA
    ema_slow : int
        4時間足長期EMA
    slope_period : int
        EMA傾き計算期間
    min_slope : float
        最小傾き閾値

    Returns
    -------
    pd.Series
        シグナル ('long' / 'short' / None)
    """
    # ベースシグナル（PA1）
    raw_sig = signal_pa1_reversal(bars_1h, zone_atr=zone_atr, lookback=lookback)

    # 4時間足EMAトレンドフィルター構築
    ema_filter = build_ema_trend_filter(bars_4h, fast=ema_fast, slow=ema_slow,
                                        slope_period=slope_period, min_slope=min_slope)

    # 1時間足インデックスにアライン
    ema_aligned = align_htf_to_ltf(ema_filter, bars_1h.index)

    # フィルター適用
    filtered = pd.Series(index=bars_1h.index, dtype=object)

    for i in range(len(raw_sig)):
        sig = raw_sig.iloc[i]
        if sig not in ['long', 'short']:
            continue
        ts = raw_sig.index[i]
        try:
            row = ema_aligned.loc[ts]
            if sig == 'long' and row['long_ok']:
                filtered.iloc[i] = 'long'
            elif sig == 'short' and row['short_ok']:
                filtered.iloc[i] = 'short'
        except KeyError:
            pass

    return filtered


# ---- パラメータセット定義 ----
DIRECTION_B_PARAMS = [
    # (label, ema_fast, ema_slow, slope_period, min_slope)
    ('B1_EMA20_50_noSlope',   20, 50,  3, 0.0),
    ('B2_EMA20_50_slope',     20, 50,  3, 0.005),
    ('B3_EMA10_30_noSlope',   10, 30,  3, 0.0),
    ('B4_EMA10_30_slope',     10, 30,  3, 0.005),
    ('B5_EMA50_200_noSlope',  50, 200, 5, 0.0),
    ('B6_EMA20_100_noSlope',  20, 100, 3, 0.0),
    ('B7_EMA20_50_strongSlp', 20, 50,  5, 0.01),
    ('B8_EMA10_50_noSlope',   10, 50,  3, 0.0),
    ('B9_EMA30_100_noSlope',  30, 100, 3, 0.0),
]
