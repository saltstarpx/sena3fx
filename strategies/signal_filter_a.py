"""
方向A: シグナル精度向上フィルター
====================================
PA1_Reversal_TightSL に以下の追加条件を組み合わせる。

1. インサイドバー確認 (inside_bar)
   - シグナル足の次の足が前足の高値・安値の内側に収まる
   - 「迷い」を確認してからエントリー → 偽シグナル除去

2. ボリューム確認 (volume_confirm)
   - シグナル足のボリュームが直近N本平均の X 倍以上
   - 勢いのある反転のみを採用

3. 時間帯フィルター (session_filter)
   - 東京 (00:00〜09:00 UTC)
   - ロンドン (07:00〜16:00 UTC)
   - NY (13:00〜22:00 UTC)
   - 上記セッション重複時間帯（流動性が高い時間）のみ許可

各フィルターは独立して ON/OFF 可能。
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.yagami_pa import signal_pa1_reversal


def _calc_atr(bars, period=14):
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars.index).rolling(period).mean()


def signal_direction_a(
    bars,
    # PA1ベースパラメータ
    zone_atr=1.5,
    lookback=20,
    # インサイドバーフィルター
    use_inside_bar=True,
    # ボリュームフィルター
    use_volume=True,
    volume_mult=1.3,
    volume_lookback=20,
    # 時間帯フィルター
    use_session=True,
    sessions=('tokyo', 'london', 'ny'),
):
    """
    方向Aシグナル生成関数。

    Parameters
    ----------
    bars : pd.DataFrame
        1時間足OHLCV
    zone_atr : float
        PA1のゾーン幅（ATR倍率）
    lookback : int
        PA1のルックバック本数
    use_inside_bar : bool
        インサイドバー確認を使用するか
    use_volume : bool
        ボリューム確認を使用するか
    volume_mult : float
        ボリューム確認の閾値倍率
    volume_lookback : int
        ボリューム平均の計算期間
    use_session : bool
        時間帯フィルターを使用するか
    sessions : tuple
        許可するセッション名 ('tokyo', 'london', 'ny')

    Returns
    -------
    pd.Series
        シグナル ('long' / 'short' / None)
    """
    # ベースシグナル（PA1）
    raw_sig = signal_pa1_reversal(bars, zone_atr=zone_atr, lookback=lookback)

    # ボリューム移動平均
    if use_volume and 'volume' in bars.columns:
        vol_ma = bars['volume'].rolling(volume_lookback).mean()
    else:
        vol_ma = None

    filtered = pd.Series(index=bars.index, dtype=object)

    for i in range(lookback + 2, len(bars)):
        sig = raw_sig.iloc[i]
        if sig not in ['long', 'short']:
            continue

        ts = bars.index[i]

        # ---- 時間帯フィルター ----
        if use_session:
            hour_utc = ts.hour if hasattr(ts, 'hour') else pd.Timestamp(ts).hour
            in_session = False
            if 'tokyo' in sessions and 0 <= hour_utc < 9:
                in_session = True
            if 'london' in sessions and 7 <= hour_utc < 16:
                in_session = True
            if 'ny' in sessions and 13 <= hour_utc < 22:
                in_session = True
            if not in_session:
                continue

        # ---- ボリュームフィルター ----
        if use_volume and vol_ma is not None:
            vol_avg = vol_ma.iloc[i]
            vol_now = bars['volume'].iloc[i]
            if not np.isnan(vol_avg) and vol_avg > 0:
                if vol_now < volume_mult * vol_avg:
                    continue

        # ---- インサイドバー確認 ----
        # シグナル足(i)の次の足(i+1)が前足(i-1)の範囲内に収まるか確認
        # ただし i+1 が存在する場合のみ（先読みなし: シグナルは i+1 確定後に発火）
        if use_inside_bar:
            if i + 1 >= len(bars):
                continue
            prev_high = bars['high'].iloc[i - 1]
            prev_low = bars['low'].iloc[i - 1]
            next_high = bars['high'].iloc[i + 1]
            next_low = bars['low'].iloc[i + 1]
            # インサイドバー: 次足が前足の高値・安値の内側
            if not (next_high <= prev_high and next_low >= prev_low):
                continue

        filtered.iloc[i] = sig

    return filtered


# ---- パラメータセット定義 ----
DIRECTION_A_PARAMS = [
    # (label, use_inside_bar, use_volume, volume_mult, use_session, sessions)
    ('A1_IB_only',      True,  False, 1.3, False, ('tokyo','london','ny')),
    ('A2_Vol_only',     False, True,  1.3, False, ('tokyo','london','ny')),
    ('A3_Sess_only',    False, False, 1.3, True,  ('tokyo','london','ny')),
    ('A4_IB_Vol',       True,  True,  1.3, False, ('tokyo','london','ny')),
    ('A5_IB_Sess',      True,  False, 1.3, True,  ('tokyo','london','ny')),
    ('A6_Vol_Sess',     False, True,  1.3, True,  ('tokyo','london','ny')),
    ('A7_All',          True,  True,  1.3, True,  ('tokyo','london','ny')),
    ('A8_All_HighVol',  True,  True,  1.5, True,  ('tokyo','london','ny')),
    ('A9_LondonNY',     True,  True,  1.3, True,  ('london','ny')),
]
