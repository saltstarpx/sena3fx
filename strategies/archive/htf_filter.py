"""
上位足（4時間足）フィルター
============================
やがみ氏の「エントリー時間軸の12倍上位足でレジサポ確認」原則を実装。

1時間足エントリーに対して4時間足のレジスタンス/サポートゾーンを確認し、
シグナルの信頼性を向上させる。

フィルター条件:
  ロングシグナル: 4時間足でサポート付近にいる（安値圏）
  ショートシグナル: 4時間足でレジスタンス付近にいる（高値圏）
"""
import numpy as np
import pandas as pd


def _calc_atr_4h(bars_4h, period=14):
    h = bars_4h['high'].values
    l = bars_4h['low'].values
    c = bars_4h['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars_4h.index).rolling(period).mean()


def build_htf_filter(bars_4h, lookback=20, zone_atr=1.5):
    """
    4時間足データからサポート/レジスタンスゾーンを構築し、
    各タイムスタンプにおける「ロング許可」「ショート許可」フラグを返す。

    Parameters
    ----------
    bars_4h : pd.DataFrame
        4時間足OHLC（インデックスはdatetimeまたはtimestamp）
    lookback : int
        スウィングゾーン判定のルックバック本数
    zone_atr : float
        ゾーン幅（ATR倍率）

    Returns
    -------
    pd.DataFrame
        カラム: ['long_ok', 'short_ok', 'htf_swing_low', 'htf_swing_high', 'htf_atr']
        インデックス: 4時間足タイムスタンプ
    """
    bars = bars_4h.copy()
    atr = _calc_atr_4h(bars)
    swing_high = bars['high'].rolling(lookback).max()
    swing_low = bars['low'].rolling(lookback).min()

    long_ok = pd.Series(False, index=bars.index)
    short_ok = pd.Series(False, index=bars.index)

    for i in range(lookback, len(bars)):
        a = atr.iloc[i]
        if np.isnan(a) or a == 0:
            continue
        price = bars['close'].iloc[i]
        sw_low = swing_low.iloc[i]
        sw_high = swing_high.iloc[i]

        # ロング許可: 4時間足の安値圏（スウィングロー + zone_atr×ATR以内）
        if not np.isnan(sw_low) and price <= sw_low + zone_atr * a:
            long_ok.iloc[i] = True

        # ショート許可: 4時間足の高値圏（スウィングハイ - zone_atr×ATR以内）
        if not np.isnan(sw_high) and price >= sw_high - zone_atr * a:
            short_ok.iloc[i] = True

    result = pd.DataFrame({
        'long_ok': long_ok,
        'short_ok': short_ok,
        'htf_swing_low': swing_low,
        'htf_swing_high': swing_high,
        'htf_atr': atr,
    })
    return result


def align_htf_to_ltf(htf_filter_df, ltf_index):
    """
    4時間足フィルターを1時間足のインデックスに前方補完でアラインする。

    Parameters
    ----------
    htf_filter_df : pd.DataFrame
        build_htf_filter() の出力
    ltf_index : pd.DatetimeIndex
        1時間足のタイムスタンプインデックス

    Returns
    -------
    pd.DataFrame
        ltf_index に合わせてリサンプルされたフィルターDF
    """
    # インデックスをUTC統一
    htf = htf_filter_df.copy()
    if htf.index.tz is None:
        htf.index = htf.index.tz_localize('UTC')
    else:
        htf.index = htf.index.tz_convert('UTC')

    ltf_idx = ltf_index
    if ltf_idx.tz is None:
        ltf_idx = ltf_idx.tz_localize('UTC')
    else:
        ltf_idx = ltf_idx.tz_convert('UTC')

    # 1時間足インデックスに reindex して前方補完
    combined_idx = htf.index.union(ltf_idx).sort_values()
    htf_reindexed = htf.reindex(combined_idx).ffill()
    aligned = htf_reindexed.reindex(ltf_idx)
    return aligned


def apply_htf_filter(ltf_signals, htf_aligned):
    """
    1時間足シグナルに4時間足フィルターを適用する。

    Parameters
    ----------
    ltf_signals : pd.Series
        1時間足シグナル（'long' / 'short' / None）
    htf_aligned : pd.DataFrame
        align_htf_to_ltf() の出力

    Returns
    -------
    pd.Series
        フィルター後のシグナル
    """
    filtered = pd.Series(index=ltf_signals.index, dtype=object)

    for i in range(len(ltf_signals)):
        sig = ltf_signals.iloc[i]
        if sig not in ['long', 'short']:
            continue
        ts = ltf_signals.index[i]
        try:
            row = htf_aligned.loc[ts]
            if sig == 'long' and row['long_ok']:
                filtered.iloc[i] = 'long'
            elif sig == 'short' and row['short_ok']:
                filtered.iloc[i] = 'short'
        except KeyError:
            pass

    return filtered
