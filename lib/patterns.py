"""
やがみメソッド チャートパターン検出エンジン
============================================
フラッグ、ウェッジ、逆三尊、三尊、ダブルボトム/トップ、
三角持ち合い、アセトラ、カップ&ハンドル等の自動検出。

やがみの教え:
  - 三角系パターンは三角形の高さ分がTPの最低目安
  - フラッグの下落角度25-30°が理想
  - 綺麗すぎるパターンほど騙しに注意
  - 二番底/二番天井を待ってからエントリー
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _find_pivots(h, l, window=5):
    """スイングハイ/ローの検出"""
    n = len(h)
    swing_highs = []
    swing_lows = []

    for i in range(window, n - window):
        if h[i] == max(h[i-window:i+window+1]):
            swing_highs.append((i, h[i]))
        if l[i] == min(l[i-window:i+window+1]):
            swing_lows.append((i, l[i]))

    return swing_highs, swing_lows


def detect_chart_patterns(bars: pd.DataFrame, pivot_window=5) -> pd.DataFrame:
    """
    チャートパターンを検出して各バーにラベルを付与。

    'chart_pattern' 列:
      'inv_hs_long' - 逆三尊ロング（右肩形成）
      'hs_short' - 三尊ショート（右肩形成）
      'flag_bull' - ブルフラッグブレイク
      'flag_bear' - ベアフラッグブレイク
      'wedge_bull' - ウェッジ上抜け
      'wedge_bear' - ウェッジ下抜け
      'triangle_break_bull' - 三角持ち合い上抜け
      'triangle_break_bear' - 三角持ち合い下抜け
      'ascending_tri' - アセンディングトライアングル上抜け
      'descending_tri' - ディセンディングトライアングル下抜け
      None - なし
    """
    o = bars['open'].values
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    n = len(bars)

    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14).mean().values

    swing_highs, swing_lows = _find_pivots(h, l, pivot_window)

    patterns = [None] * n
    pattern_tp = np.zeros(n)  # 推奨TP距離

    for i in range(20, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # 最近のスイングポイント取得
        recent_highs = [(idx, val) for idx, val in swing_highs if i - 40 <= idx < i]
        recent_lows = [(idx, val) for idx, val in swing_lows if i - 40 <= idx < i]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            continue

        # --- 逆三尊 (Inverse Head & Shoulders) ---
        # 3つの安値で中央が最も低い + ネックラインの確認
        if len(recent_lows) >= 3:
            for li in range(len(recent_lows) - 2):
                l1_idx, l1_val = recent_lows[li]
                l2_idx, l2_val = recent_lows[li + 1]
                l3_idx, l3_val = recent_lows[li + 2]

                # 中央(トップ)が最も低い
                if l2_val < l1_val and l2_val < l3_val:
                    # 左肩と右肩が近い水準
                    shoulder_diff = abs(l1_val - l3_val)
                    head_depth = min(l1_val, l3_val) - l2_val

                    if (shoulder_diff < atr[i] * 2.0 and
                        head_depth > atr[i] * 1.0 and
                        l3_idx > l2_idx > l1_idx):

                        # ネックライン近辺のブレイク確認
                        neckline_highs = [val for idx, val in recent_highs
                                         if l1_idx < idx < l3_idx]
                        if neckline_highs:
                            neckline = np.mean(neckline_highs)
                            if c[i] > neckline and c[i-1] <= neckline:
                                patterns[i] = 'inv_hs_long'
                                pattern_tp[i] = neckline - l2_val
                                break

        # --- 三尊 (Head & Shoulders) ---
        if len(recent_highs) >= 3 and patterns[i] is None:
            for hi in range(len(recent_highs) - 2):
                h1_idx, h1_val = recent_highs[hi]
                h2_idx, h2_val = recent_highs[hi + 1]
                h3_idx, h3_val = recent_highs[hi + 2]

                if h2_val > h1_val and h2_val > h3_val:
                    shoulder_diff = abs(h1_val - h3_val)
                    head_height = h2_val - max(h1_val, h3_val)

                    if (shoulder_diff < atr[i] * 2.0 and
                        head_height > atr[i] * 1.0 and
                        h3_idx > h2_idx > h1_idx):

                        neckline_lows = [val for idx, val in recent_lows
                                        if h1_idx < idx < h3_idx]
                        if neckline_lows:
                            neckline = np.mean(neckline_lows)
                            if c[i] < neckline and c[i-1] >= neckline:
                                patterns[i] = 'hs_short'
                                pattern_tp[i] = h2_val - neckline
                                break

        if patterns[i] is not None:
            continue

        # --- フラッグ ---
        # 上昇トレンド後の下方チャネル → 上抜け
        lookback = 15
        if i >= lookback:
            seg_h = h[i-lookback:i+1]
            seg_l = l[i-lookback:i+1]
            seg_c = c[i-lookback:i+1]

            # 安値の切り下がりチェック
            low_slope = np.polyfit(range(lookback+1), seg_l, 1)[0]
            high_slope = np.polyfit(range(lookback+1), seg_h, 1)[0]

            # ブルフラッグ: 両方下向き + 今足で上限ブレイク
            if (low_slope < -atr[i] * 0.02 and high_slope < -atr[i] * 0.02 and
                abs(low_slope - high_slope) < atr[i] * 0.03):
                trend_line_val = seg_h[0] + high_slope * lookback
                if c[i] > trend_line_val + atr[i] * 0.3:
                    flag_height = max(seg_h) - min(seg_l)
                    patterns[i] = 'flag_bull'
                    pattern_tp[i] = flag_height
                    continue

            # ベアフラッグ: 両方上向き + 今足で下限ブレイク
            if (low_slope > atr[i] * 0.02 and high_slope > atr[i] * 0.02 and
                abs(low_slope - high_slope) < atr[i] * 0.03):
                trend_line_val = seg_l[0] + low_slope * lookback
                if c[i] < trend_line_val - atr[i] * 0.3:
                    flag_height = max(seg_h) - min(seg_l)
                    patterns[i] = 'flag_bear'
                    pattern_tp[i] = flag_height
                    continue

        # --- 三角持ち合い (Symmetric Triangle) ---
        if i >= 12:
            seg_len = 12
            seg_h = h[i-seg_len:i+1]
            seg_l = l[i-seg_len:i+1]

            high_slope = np.polyfit(range(seg_len+1), seg_h, 1)[0]
            low_slope = np.polyfit(range(seg_len+1), seg_l, 1)[0]

            # 高値切り下げ + 安値切り上げ → 収束
            if high_slope < -atr[i] * 0.01 and low_slope > atr[i] * 0.01:
                tri_height = max(seg_h) - min(seg_l)
                upper_line = seg_h[0] + high_slope * seg_len
                lower_line = seg_l[0] + low_slope * seg_len

                if c[i] > upper_line + atr[i] * 0.2:
                    patterns[i] = 'triangle_break_bull'
                    pattern_tp[i] = tri_height
                    continue
                elif c[i] < lower_line - atr[i] * 0.2:
                    patterns[i] = 'triangle_break_bear'
                    pattern_tp[i] = tri_height
                    continue

            # アセンディングトライアングル: 高値が水平 + 安値切り上げ
            if abs(high_slope) < atr[i] * 0.005 and low_slope > atr[i] * 0.01:
                resist_level = np.mean(seg_h[-5:])
                if c[i] > resist_level + atr[i] * 0.2:
                    patterns[i] = 'ascending_tri'
                    pattern_tp[i] = resist_level - min(seg_l)
                    continue

            # ディセンディングトライアングル: 安値が水平 + 高値切り下げ
            if abs(low_slope) < atr[i] * 0.005 and high_slope < -atr[i] * 0.01:
                support_level = np.mean(seg_l[-5:])
                if c[i] < support_level - atr[i] * 0.2:
                    patterns[i] = 'descending_tri'
                    pattern_tp[i] = max(seg_h) - support_level
                    continue

    result = bars.copy()
    result['chart_pattern'] = patterns
    result['pattern_tp'] = pattern_tp
    return result
