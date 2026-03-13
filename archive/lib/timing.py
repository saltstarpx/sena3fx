"""
やがみメソッド 足更新タイミング & MTF分析
==========================================
「足更新でポジションを取る（中途半端な時間にポジションを取らない）」
「上位足の色に順張り」
「上位足の更新タイミングほど初動が美味しい」
「15分足で反転の判断をしている」

実装:
  1. 足更新タイミング検出（4h/日足更新の直後）
  2. 上位足の色判定（順張り方向の決定）
  3. MTF分析（複数時間足の方向性統合）
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def detect_bar_update_timing(bars: pd.DataFrame, freq: str) -> pd.Series:
    """
    各バーが上位足の更新タイミングかどうかを判定。

    やがみ: 「自分が軸にしている時間足より上位の足の色に順張り」
    「4hをお勧め」「同時に更新される足が多い」

    freq: 'auto'の場合はバーのインターバルから推定

    Returns:
      pd.Series of bool (True = 上位足更新タイミング)
    """
    idx = bars.index
    n = len(bars)
    is_update = np.zeros(n, dtype=bool)

    # 時間足に応じた上位足更新タイミング
    if freq == '15min' or freq == '15T':
        # 15分足 → 1h, 4h更新をマーク
        for i in range(n):
            minute = idx[i].minute
            hour = idx[i].hour
            if minute == 0:  # 1h更新
                is_update[i] = True
            if minute == 0 and hour % 4 == 0:  # 4h更新
                is_update[i] = True
    elif freq == '1h' or freq == '1H':
        # 1h足 → 4h, 日足更新をマーク
        for i in range(n):
            hour = idx[i].hour
            if hour % 4 == 0:  # 4h更新
                is_update[i] = True
            if hour == 0:  # 日足更新
                is_update[i] = True
    elif freq == '4h' or freq == '4H':
        # 4h足 → 日足更新をマーク
        for i in range(n):
            hour = idx[i].hour
            if hour == 0:
                is_update[i] = True
    else:
        # デフォルト: 全てTrue
        is_update[:] = True

    return pd.Series(is_update, index=bars.index, name='bar_update')


def higher_tf_direction(bars: pd.DataFrame, freq: str) -> pd.Series:
    """
    上位足の方向（陽線/陰線）を判定。

    やがみ: 「日足が陽線なら日足の始値を背にロングを積む」
    「4hが陽線なら確定した瞬間の始値を背に次の4hの始値でロング」

    Returns:
      pd.Series: 1.0 = 上位足陽線(ロング有利), -1.0 = 上位足陰線(ショート有利), 0 = 不明
    """
    # 上位足のバーを作成
    if freq in ('1h', '1H'):
        higher_freq = '4h'
    elif freq in ('4h', '4H'):
        higher_freq = '1D'
    elif freq in ('15min', '15T'):
        higher_freq = '1h'
    else:
        higher_freq = '1D'

    higher_bars = bars['close'].resample(higher_freq).agg(
        open='first', close='last'
    )
    # NaN除去
    higher_bars = higher_bars.dropna()

    if len(higher_bars) == 0:
        return pd.Series(0.0, index=bars.index, name='htf_direction')

    # 各バーに対して、その時点での上位足の方向をマッピング
    direction = pd.Series(0.0, index=bars.index, name='htf_direction')

    for i in range(len(bars)):
        ts = bars.index[i]
        # 直近の上位足バーを探す
        mask = higher_bars.index <= ts
        if mask.any():
            last_higher = higher_bars.loc[mask].iloc[-1]
            if last_higher['close'] > last_higher['open']:
                direction.iloc[i] = 1.0
            elif last_higher['close'] < last_higher['open']:
                direction.iloc[i] = -1.0

    return direction


def mtf_confluence(bars_dict: dict) -> pd.DataFrame:
    """
    複数時間足の方向性を統合。

    やがみ: 「短期のtpに来た時の上位足を考える」
    「何時にどの辺で上位足が閉じたら次のローソク足は何色になりそうか」

    Args:
      bars_dict: {'1h': df_1h, '4h': df_4h, '1D': df_daily}

    Returns:
      DataFrame with 'mtf_score' column (-1.0 ~ +1.0)
    """
    base_freq = min(bars_dict.keys(), key=lambda k: {
        '15min': 1, '1h': 2, '4h': 3, '1D': 4
    }.get(k, 5))
    base_bars = bars_dict[base_freq]

    scores = pd.DataFrame(index=base_bars.index)
    weights = {'15min': 0.1, '1h': 0.2, '4h': 0.35, '1D': 0.35}

    for freq, df in bars_dict.items():
        w = weights.get(freq, 0.2)
        direction = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                direction.iloc[i] = 1.0
            elif df['close'].iloc[i] < df['open'].iloc[i]:
                direction.iloc[i] = -1.0

        # base_barsのインデックスにリサンプル
        direction = direction.reindex(base_bars.index, method='ffill').fillna(0)
        scores[f'dir_{freq}'] = direction * w

    scores['mtf_score'] = scores.sum(axis=1)
    return scores


def session_filter(bars: pd.DataFrame) -> pd.Series:
    """
    セッション判定。

    やがみ: 「東京時間の上昇は順張りしてロンドン時間やニューヨーク時間でショートを積む」
    requirements_summary: 「アジア時間ブレイクアウトは危険」

    Returns:
      pd.Series: 'asia', 'london', 'ny'
    """
    sessions = pd.Series('', index=bars.index, dtype=object)

    for i in range(len(bars)):
        hour = bars.index[i].hour  # UTC
        if 21 <= hour or hour < 7:
            sessions.iloc[i] = 'asia'
        elif 7 <= hour < 15:
            sessions.iloc[i] = 'london'
        else:  # 15-21
            sessions.iloc[i] = 'ny'

    return sessions
