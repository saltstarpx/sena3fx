"""
やがみメソッド レジサポ自動検出
================================
「ローソク足の髭先、髭と実体の間(つまり始値や終値)に水平線を引く」
「反発回数が多い程、強固なレジサポ」
「自分がトレードしている時間軸を大体12倍すると丁度いい」

実装:
  - 実体上端/下端(始値/終値)とヒゲ先(高値/安値)から水平レベルを抽出
  - クラスタリングで近接レベルを統合
  - タッチ回数で強度を算出
  - 上位足のレジサポも計算（MTF対応）
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


def extract_levels(bars: pd.DataFrame, tolerance_atr_mult=0.3,
                   min_touches=2, atr_period=14) -> List[dict]:
    """
    ローソク足からレジサポレベルを抽出。

    やがみの方法:
      - ヒゲ先(高値/安値)
      - 実体端(始値/終値)
    からレベル候補を抽出し、近接するものを統合。

    Returns:
      List[dict] with keys: level, touches, type('support'/'resistance'), strength
    """
    o = bars['open'].values
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    n = len(bars)

    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr_val = np.nanmean(pd.Series(tr).rolling(atr_period).mean().values[-20:])
    if np.isnan(atr_val) or atr_val == 0:
        return []

    tolerance = atr_val * tolerance_atr_mult

    # 全てのレベル候補を収集
    candidates = []
    for i in range(n):
        candidates.append(h[i])   # ヒゲ先(高値)
        candidates.append(l[i])   # ヒゲ先(安値)
        candidates.append(max(o[i], c[i]))  # 実体上端
        candidates.append(min(o[i], c[i]))  # 実体下端

    candidates = np.array(candidates)
    candidates.sort()

    # クラスタリング（近接レベルを統合）
    clusters = []
    current_cluster = [candidates[0]]

    for i in range(1, len(candidates)):
        if candidates[i] - np.mean(current_cluster) <= tolerance:
            current_cluster.append(candidates[i])
        else:
            if len(current_cluster) >= min_touches:
                clusters.append(current_cluster)
            current_cluster = [candidates[i]]
    if len(current_cluster) >= min_touches:
        clusters.append(current_cluster)

    # レベルとしてまとめる
    current_price = c[-1]
    levels = []
    for cluster in clusters:
        level = np.mean(cluster)
        touches = len(cluster)
        level_type = 'support' if level < current_price else 'resistance'
        # 強度: タッチ回数と現在価格との距離で計算
        distance = abs(level - current_price) / atr_val
        strength = min(1.0, touches / 10.0) * max(0.1, 1.0 - distance / 20.0)

        levels.append({
            'level': round(level, 4),
            'touches': touches,
            'type': level_type,
            'strength': round(strength, 4),
            'distance_atr': round(distance, 2),
        })

    # 強度順にソート
    levels.sort(key=lambda x: x['strength'], reverse=True)
    return levels


def nearest_support_resistance(bars: pd.DataFrame, current_idx: int = -1,
                               tolerance_atr_mult=0.3) -> Tuple[float, float]:
    """
    現在価格に最も近いサポートとレジスタンスのレベルを返す。
    やがみ: 「上位足ほど強固なレジサポが構築される」

    Returns:
      (nearest_support, nearest_resistance)
    """
    levels = extract_levels(bars, tolerance_atr_mult)
    price = bars['close'].values[current_idx]

    supports = [lv['level'] for lv in levels if lv['type'] == 'support']
    resistances = [lv['level'] for lv in levels if lv['type'] == 'resistance']

    nearest_sup = max(supports) if supports else price - 100
    nearest_res = min(resistances) if resistances else price + 100

    return nearest_sup, nearest_res


def is_at_level(price: float, levels: List[dict], atr: float,
                proximity_mult=0.5) -> Tuple[bool, str]:
    """
    現在価格がレジサポレベル付近にあるか判定。

    やがみ: 「レジサポの位置でエントリーする」
    Returns:
      (is_near_level, level_type)
    """
    for lv in levels:
        if abs(price - lv['level']) < atr * proximity_mult:
            return True, lv['type']
    return False, ''


def extract_levels_binned(bars: pd.DataFrame, n_bins: int = 40,
                          min_freq_pct: float = 0.06,
                          atr_period: int = 14) -> List[dict]:
    """
    価格帯ビン（クラスタ）方式でレジサポを抽出 (Proposal D)。

    従来の extract_levels は min_touches=2 の水平クラスタリングのため
    ほぼ全バーで C1 が成立（99%+）してしまう。
    本関数は価格レンジを n_bins 個のビンに分割し、
    ローソク足の価格点（高値・安値・実体端）が min_freq_pct 以上
    集中するビンのみを S/R ゾーンとして返す。
    これにより C1 成立率を 20〜60% 程度に制御できる。

    Args:
      n_bins:        価格レンジを分割するビン数（多いほど細かい）
      min_freq_pct:  有効 S/R ゾーンとして認める最低頻度（全価格点に対する割合）
      atr_period:    ATR 計算期間

    Returns:
      List[dict] with keys: level, touches, type, strength, distance_atr
    """
    o = bars['open'].values
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    n = len(bars)

    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr_val = np.nanmean(pd.Series(tr).rolling(atr_period).mean().values[-20:])
    if np.isnan(atr_val) or atr_val == 0:
        return []

    # 全価格点を収集（高値・安値・実体端）
    points = np.concatenate([
        h, l,
        np.maximum(o, c),  # 実体上端
        np.minimum(o, c),  # 実体下端
    ])
    points = points[np.isfinite(points)]
    if len(points) == 0:
        return []

    price_min = points.min()
    price_max = points.max()
    if price_max <= price_min:
        return []

    counts, bin_edges = np.histogram(points, bins=n_bins,
                                     range=(price_min, price_max))
    total_points = len(points)
    threshold = total_points * min_freq_pct
    current_price = c[-1]

    levels = []
    for idx in range(n_bins):
        if counts[idx] < threshold:
            continue
        level = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0
        touches = int(counts[idx])
        level_type = 'support' if level < current_price else 'resistance'
        distance = abs(level - current_price) / atr_val
        strength = min(1.0, touches / (total_points * 0.1)) * max(0.1, 1.0 - distance / 20.0)
        levels.append({
            'level': round(level, 4),
            'touches': touches,
            'type': level_type,
            'strength': round(strength, 4),
            'distance_atr': round(distance, 2),
        })

    levels.sort(key=lambda x: x['strength'], reverse=True)
    return levels
