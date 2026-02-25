"""
やがみメソッド ローソク足分析エンジン
=====================================
「ローソク足の本」「ローソク足の本2」「ポジり方の本」の内容を完全実装。

実装パターン:
  [単体ローソク足]
    ① 大陽線/大陰線 - 逆張り厳禁
    ② 中陽線/中陰線 - モブ（シグナルなし）
    ③ 包み足(Engulfing) - 転換点の鉄板
    ④ 上下ヒゲピンバー(Doji) - トレンドレス、ポジ禁止
    ⑤ 下ヒゲ陽線/上ヒゲ陰線 - 底打ち/天井サイン
    ⑥ ピンバー強陽線/強陰線 - 強いが消耗注意
    ⑦ 売り押し目の陽線/買い押し目の陰線 - 次足更新で逆張り

  [プライスアクション]
    - リバーサルロー/ハイ（最強の反転シグナル）
    - 一定価格以下ヒゲ（実体揃い）
    - 無限スラストアップ/ダウン
    - 上ヒゲ埋めムーブ（wick fill）
    - インサイドバー

  [ポジり方の本]
    - 二番底/二番天井を待つ
    - 15分足で逆の色の足が出てから逆張り
    - 足更新タイミングでエントリー
    - 建値ストップの適切な判定
"""
import numpy as np
import pandas as pd


def body(o, c):
    """実体の大きさ(絶対値)"""
    return np.abs(c - o)


def upper_wick(h, o, c):
    """上ヒゲの長さ"""
    return h - np.maximum(o, c)


def lower_wick(l, o, c):
    """下ヒゲの長さ"""
    return np.minimum(o, c) - l


def candle_range(h, l):
    """ローソク足の全体幅"""
    return h - l


def is_bullish(o, c):
    return c > o


def is_bearish(o, c):
    return c < o


# ==============================================================
# 単体ローソク足の強弱判定
# ==============================================================

def detect_single_candle(bars: pd.DataFrame, atr_period=14) -> pd.DataFrame:
    """
    各バーに対して単体ローソク足の分類を付与する。

    返り値の 'candle_type' 列:
      'big_bull', 'big_bear' - 大陽線/大陰線（①）
      'mid_bull', 'mid_bear' - 中陽線/中陰線（②）
      'engulf_bull', 'engulf_bear' - 包み足（③）
      'doji' - 上下ヒゲピンバー（④）
      'hammer', 'inv_hammer' - 下ヒゲ陽線/上ヒゲ陰線（⑤）
      'pinbar_bull', 'pinbar_bear' - ピンバー強線（⑥）
      'pullback_bull', 'pullback_bear' - 押し目足（⑦）
      'neutral' - その他
    """
    o = bars['open'].values
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values

    n = len(bars)
    bd = body(o, c)
    cr = candle_range(h, l)
    uw = upper_wick(h, o, c)
    lw = lower_wick(l, o, c)

    # ATR計算
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)),
        np.abs(l - np.roll(c, 1))
    ))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(atr_period).mean().values

    types = np.full(n, 'neutral', dtype=object)
    strength = np.zeros(n)  # -1(超弱気) ~ +1(超強気)

    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0 or cr[i] == 0:
            continue

        body_ratio = bd[i] / cr[i]  # 実体/全幅
        body_atr = bd[i] / atr[i]   # 実体/ATR
        uw_ratio = uw[i] / cr[i] if cr[i] > 0 else 0
        lw_ratio = lw[i] / cr[i] if cr[i] > 0 else 0
        bull = c[i] > o[i]
        bear = c[i] < o[i]

        # ④ Doji / 上下ヒゲピンバー: 実体が小さく上下ヒゲが両方ある
        if body_ratio < 0.2 and uw_ratio > 0.3 and lw_ratio > 0.3:
            types[i] = 'doji'
            strength[i] = 0.0
            continue

        # ① 大陽線/大陰線: 実体がATR比で大きい + 実体比率高い
        if body_atr > 1.5 and body_ratio > 0.65:
            if bull:
                types[i] = 'big_bull'
                strength[i] = min(1.0, body_atr / 3.0)
            else:
                types[i] = 'big_bear'
                strength[i] = -min(1.0, body_atr / 3.0)
            continue

        # ③ 包み足(Engulfing): 前回足の高値安値を包む
        prev_body = bd[i - 1]
        if (bull and c[i] > max(o[i-1], c[i-1]) and o[i] < min(o[i-1], c[i-1])
                and bd[i] > prev_body * 1.1):
            types[i] = 'engulf_bull'
            strength[i] = 0.7
            continue
        if (bear and c[i] < min(o[i-1], c[i-1]) and o[i] > max(o[i-1], c[i-1])
                and bd[i] > prev_body * 1.1):
            types[i] = 'engulf_bear'
            strength[i] = -0.7
            continue

        # ⑤ 下ヒゲ陽線(Hammer) / 上ヒゲ陰線(Inverted Hammer)
        if bull and lw_ratio > 0.55 and uw_ratio < 0.15 and body_ratio > 0.15:
            types[i] = 'hammer'
            strength[i] = 0.6
            continue
        if bear and uw_ratio > 0.55 and lw_ratio < 0.15 and body_ratio > 0.15:
            types[i] = 'inv_hammer'
            strength[i] = -0.6
            continue

        # ⑥ ピンバー強陽線/強陰線: ヒゲが長いが実体もそこそこ
        if bull and lw_ratio > 0.45 and body_ratio > 0.25 and body_atr > 0.8:
            types[i] = 'pinbar_bull'
            strength[i] = 0.5
            continue
        if bear and uw_ratio > 0.45 and body_ratio > 0.25 and body_atr > 0.8:
            types[i] = 'pinbar_bear'
            strength[i] = -0.5
            continue

        # ⑦ 売り押し目の陽線 / 買い押し目の陰線
        # 下降トレンド中の弱い陽線 → 次足更新でショート有効
        if i >= 3:
            recent_bears = sum(1 for j in range(i-3, i) if c[j] < o[j])
            if bull and recent_bears >= 2 and body_atr < 0.6:
                types[i] = 'pullback_bull'  # 下降中の弱い陽線→ショート押し目
                strength[i] = -0.3  # ショート寄り
                continue
            recent_bulls = sum(1 for j in range(i-3, i) if c[j] > o[j])
            if bear and recent_bulls >= 2 and body_atr < 0.6:
                types[i] = 'pullback_bear'  # 上昇中の弱い陰線→ロング押し目
                strength[i] = 0.3
                continue

        # ② 中陽線/中陰線（モブ）
        if bull:
            types[i] = 'mid_bull'
            strength[i] = 0.1
        elif bear:
            types[i] = 'mid_bear'
            strength[i] = -0.1

    result = bars.copy()
    result['candle_type'] = types
    result['candle_strength'] = strength
    return result


# ==============================================================
# プライスアクション検出
# ==============================================================

def detect_price_action(bars: pd.DataFrame, atr_period=14) -> pd.DataFrame:
    """
    複数足にまたがるプライスアクションを検出。

    'pa_signal' 列:
      'reversal_low' - リバーサルロー（最強の底打ち）
      'reversal_high' - リバーサルハイ（最強の天井）
      'double_bottom' - ダブルボトム2番底
      'double_top' - ダブルトップ2番天井
      'wick_fill_bull' - 上ヒゲ埋めムーブ（強気）
      'wick_fill_bear' - 下ヒゲ埋めムーブ（弱気）
      'body_align_support' - 実体揃いサポート
      'body_align_resist' - 実体揃いレジスタンス
      'inside_bar' - インサイドバー
      'thrust_up' - 無限スラストアップ
      'thrust_down' - 無限スラストダウン
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
    atr = pd.Series(tr).rolling(atr_period).mean().values

    pa_signals = [None] * n
    pa_strength = np.zeros(n)

    for i in range(4, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # --- リバーサルロー ---
        # 急落→V字回復: 大陰線の後に大陽線が来て安値を大幅に回復
        if (c[i-1] < o[i-1] and  # 前足が陰線
            body(o[i-1], c[i-1]) > atr[i] * 1.2 and  # 前足が大陰線
            c[i] > o[i] and  # 今足が陽線
            body(o[i], c[i]) > atr[i] * 1.0 and  # 今足もそこそこ大きい
            c[i] > (o[i-1] + c[i-1]) / 2):  # 前足の中間以上まで回復
            pa_signals[i] = 'reversal_low'
            pa_strength[i] = 0.9
            continue

        # --- リバーサルハイ ---
        if (c[i-1] > o[i-1] and
            body(o[i-1], c[i-1]) > atr[i] * 1.2 and
            c[i] < o[i] and
            body(o[i], c[i]) > atr[i] * 1.0 and
            c[i] < (o[i-1] + c[i-1]) / 2):
            pa_signals[i] = 'reversal_high'
            pa_strength[i] = -0.9
            continue

        # --- ダブルボトム ---
        # 過去10本の最安値付近を2回テスト
        lookback = min(i, 20)
        recent_lows = l[i-lookback:i+1]
        min_low = np.min(recent_lows)
        if (abs(l[i] - min_low) < atr[i] * 0.5 and  # 今足が最安値付近
            c[i] > o[i] and  # 陽線で反発
            lower_wick(l[i], o[i], c[i]) > body(o[i], c[i]) * 0.5):  # 下ヒゲあり
            # 過去にも同レベルの安値があるか
            for j in range(max(0, i-lookback), i-2):
                if abs(l[j] - l[i]) < atr[i] * 1.0 and l[j] < l[j-1] if j > 0 else True:
                    # 間に反発があるか
                    mid_high = np.max(h[j:i])
                    if mid_high > min_low + atr[i] * 1.5:
                        pa_signals[i] = 'double_bottom'
                        pa_strength[i] = 0.8
                        break
            if pa_signals[i]:
                continue

        # --- ダブルトップ ---
        recent_highs = h[i-lookback:i+1]
        max_high = np.max(recent_highs)
        if (abs(h[i] - max_high) < atr[i] * 0.5 and
            c[i] < o[i] and
            upper_wick(h[i], o[i], c[i]) > body(o[i], c[i]) * 0.5):
            for j in range(max(0, i-lookback), i-2):
                if abs(h[j] - h[i]) < atr[i] * 1.0:
                    mid_low = np.min(l[j:i])
                    if max_high - mid_low > atr[i] * 1.5:
                        pa_signals[i] = 'double_top'
                        pa_strength[i] = -0.8
                        break
            if pa_signals[i]:
                continue

        # --- ヒゲ埋めムーブ ---
        # 前足の上ヒゲを今足の実体が埋める（やがみ: 弱気信号→ポジフラットか持ち替え）
        if i >= 2:
            prev_upper_wick_top = h[i-1]
            prev_body_top = max(o[i-1], c[i-1])
            if (prev_upper_wick_top - prev_body_top > atr[i] * 0.3 and  # 前足に上ヒゲ
                c[i] > prev_body_top and  # 今足の実体が前足の実体上端を超える
                min(o[i], c[i]) < prev_upper_wick_top):  # ヒゲを実体で埋めてる
                pa_signals[i] = 'wick_fill_bull'
                pa_strength[i] = 0.4
                continue

            prev_lower_wick_bottom = l[i-1]
            prev_body_bottom = min(o[i-1], c[i-1])
            if (prev_body_bottom - prev_lower_wick_bottom > atr[i] * 0.3 and
                c[i] < prev_body_bottom and
                max(o[i], c[i]) > prev_lower_wick_bottom):
                # やがみ: 「髭を埋めにくるムーブは即ポジションをフラットにする」
                pa_signals[i] = 'wick_fill_bear'
                pa_strength[i] = -0.4
                continue

        # --- 実体揃い(Body Alignment) ---
        # 3本以上の実体の下端/上端が揃っている → 強力なサポート/レジスタンス
        if i >= 3:
            body_lows = [min(o[j], c[j]) for j in range(i-3, i+1)]
            body_highs = [max(o[j], c[j]) for j in range(i-3, i+1)]
            low_range = max(body_lows) - min(body_lows)
            high_range = max(body_highs) - min(body_highs)

            if low_range < atr[i] * 0.3 and c[i] > o[i]:
                pa_signals[i] = 'body_align_support'
                pa_strength[i] = 0.6
                continue
            if high_range < atr[i] * 0.3 and c[i] < o[i]:
                pa_signals[i] = 'body_align_resist'
                pa_strength[i] = -0.6
                continue

        # --- インサイドバー ---
        if h[i] <= h[i-1] and l[i] >= l[i-1]:
            pa_signals[i] = 'inside_bar'
            pa_strength[i] = 0.0  # 方向性なし、パターン待ち
            continue

        # --- 無限スラストアップ/ダウン ---
        if i >= 4:
            consecutive_bull = all(c[j] > o[j] and l[j] > l[j-1] for j in range(i-3, i+1))
            if consecutive_bull:
                pa_signals[i] = 'thrust_up'
                pa_strength[i] = 0.3  # 強いが後乗り注意
                continue
            consecutive_bear = all(c[j] < o[j] and h[j] < h[j-1] for j in range(i-3, i+1))
            if consecutive_bear:
                pa_signals[i] = 'thrust_down'
                pa_strength[i] = -0.3
                continue

    result = bars.copy() if 'candle_type' not in bars.columns else bars
    result['pa_signal'] = pa_signals
    result['pa_strength'] = pa_strength
    return result


# ==============================================================
# Doji連続検出（やがみ: 4本以上連続でトレンドレス判定）
# ==============================================================

def detect_trendless(bars: pd.DataFrame, doji_count=4) -> pd.Series:
    """
    上下ヒゲピンバー(doji)が連続している区間をトレンドレスとしてマーク。
    やがみ: 「1〜4h程度の時間軸4本以上連続して出現するとポジション撤収」
    """
    o = bars['open'].values
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    n = len(bars)

    cr = candle_range(h, l)
    bd = body(o, c)
    is_doji = np.zeros(n, dtype=bool)

    for i in range(n):
        if cr[i] > 0:
            body_r = bd[i] / cr[i]
            uw_r = upper_wick(h[i], o[i], c[i]) / cr[i]
            lw_r = lower_wick(l[i], o[i], c[i]) / cr[i]
            if body_r < 0.3 and uw_r > 0.25 and lw_r > 0.25:
                is_doji[i] = True

    trendless = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if is_doji[i]:
            count += 1
        else:
            count = 0
        if count >= doji_count:
            trendless[i] = True
            # 遡ってマーク
            for j in range(max(0, i - count + 1), i):
                trendless[j] = True

    return pd.Series(trendless, index=bars.index, name='trendless')
