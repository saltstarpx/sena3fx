"""
botterコミュニティ知見フィルター集
=====================================
仮想通貨botter Advent Calendar 2024/2025 および richmanbtcチュートリアルから
吸収した手法をXAUUSD向けに実装。

参考手法:
  1. ボラティリティレジームフィルター
     - ATR比率で「静かすぎ / 適正 / 荒れすぎ」を判定
     - 出典: 消えたエッジの話 (botter Advent Calendar 2024 #22)

  2. トレンドレジームフィルター (MA傾き・EMA200)
     - EMA200方向への順張りのみ許可
     - 出典: ラリーウィリアムズ式 × botter コミュニティ変種

  3. マルチタイムフレームモメンタムフィルター
     - 4h・1d リターンの方向一致チェック
     - 出典: richmanbtc MLBot 特徴量設計思想

  4. 時刻アノマリーフィルター
     - ロンドン/NY オープン前後の高優位時間帯のみエントリー
     - 出典: botter Advent Calendar 2024 時刻アノマリー記事
"""
import numpy as np
import pandas as pd


# ===== 1. ボラティリティレジームフィルター =====

def volatility_regime(bars: pd.DataFrame,
                      atr_period: int = 14,
                      ma_period: int = 50,
                      low_thresh: float = 0.6,
                      high_thresh: float = 2.2) -> pd.Series:
    """
    ATR比率で現在のボラティリティレジームを判定。

    richmanbtcスタイル + botter Advent 2024:
    - ATR(14) / ATR(14).MA(50) を正規化比率として使用
    - 比率 < low_thresh  → レンジ相場 (スプレッドに溶ける)
    - 比率 > high_thresh → ニュース/スパイク (損切り多発)
    - それ以外          → 適正ボラ → トレード可

    Returns:
        pd.Series of bool: True = トレード可能ボラ帯
    """
    h, l, c = bars['high'].values, bars['low'].values, bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr, index=bars.index).rolling(atr_period).mean()

    # ATR比率 = 現在のATR / その長期平均
    atr_ma = atr.rolling(ma_period).mean()
    atr_ratio = atr / atr_ma

    tradeable = (atr_ratio >= low_thresh) & (atr_ratio <= high_thresh)
    return tradeable.fillna(False)


def get_atr_ratio(bars: pd.DataFrame,
                  atr_period: int = 14,
                  ma_period: int = 50) -> pd.Series:
    """ATR比率の生値を返す（分析用）"""
    h, l, c = bars['high'].values, bars['low'].values, bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr, index=bars.index).rolling(atr_period).mean()
    atr_ma = atr.rolling(ma_period).mean()
    return (atr / atr_ma).fillna(0)


# ===== 2. トレンドレジームフィルター =====

def trend_regime(bars: pd.DataFrame,
                 fast_ema: int = 50,
                 slow_ema: int = 200,
                 slope_lookback: int = 5) -> pd.Series:
    """
    EMAの傾きと位置でトレンドレジームを判定。

    botter コミュニティ推奨:
    - close > EMA(200) かつ EMA(200) が上昇中 → ロング優勢 (+1)
    - close < EMA(200) かつ EMA(200) が下降中 → ショート優勢 (-1)
    - それ以外 → 中立 (0)

    EMA(50) vs EMA(200) のゴールデンクロス/デッドクロスも考慮。

    Returns:
        pd.Series: +1 (ロング優勢) / -1 (ショート優勢) / 0 (中立)
    """
    close = bars['close']

    ema_fast = close.ewm(span=fast_ema, adjust=False).mean()
    ema_slow = close.ewm(span=slow_ema, adjust=False).mean()

    # EMA(200)の傾き（slope_lookback本前との比較）
    ema_slow_slope = ema_slow - ema_slow.shift(slope_lookback)

    regime = pd.Series(0, index=bars.index, dtype=int)

    # ロング優勢: close > EMA200 かつ EMA200上昇 かつ EMA50 > EMA200
    long_cond = (close > ema_slow) & (ema_slow_slope > 0) & (ema_fast > ema_slow)
    # ショート優勢: close < EMA200 かつ EMA200下降 かつ EMA50 < EMA200
    short_cond = (close < ema_slow) & (ema_slow_slope < 0) & (ema_fast < ema_slow)

    regime[long_cond] = 1
    regime[short_cond] = -1

    return regime


def trend_regime_simple(bars: pd.DataFrame,
                        ema_period: int = 200) -> pd.Series:
    """
    シンプル版: EMA(200)の上下で判定。
    計算コストが低いため、リアルタイム用途向け。

    Returns:
        pd.Series: +1 / -1 / 0
    """
    close = bars['close']
    ema = close.ewm(span=ema_period, adjust=False).mean()
    slope = ema - ema.shift(3)

    regime = pd.Series(0, index=bars.index, dtype=int)
    regime[close > ema] = 1
    regime[close < ema] = -1
    return regime


# ===== 3. マルチタイムフレームモメンタムフィルター =====

def mtf_momentum(bars: pd.DataFrame,
                 periods_4h_in_bars: int = 4,
                 periods_1d_in_bars: int = 24) -> pd.DataFrame:
    """
    richmanbtc MLBot 特徴量設計からの転用:
    return_4h, return_1d, return_3d を計算し、
    方向の一致度でモメンタムスコアを生成。

    1h足データの場合:
      - return_4h  = 過去4本のリターン
      - return_1d  = 過去24本のリターン
      - return_3d  = 過去72本のリターン

    Returns:
        pd.DataFrame: 'ret_4h', 'ret_1d', 'ret_3d', 'momentum_score'
    """
    close = bars['close']

    ret_4h = (close - close.shift(periods_4h_in_bars)) / close.shift(periods_4h_in_bars)
    ret_1d = (close - close.shift(periods_1d_in_bars)) / close.shift(periods_1d_in_bars)
    ret_3d = (close - close.shift(periods_1d_in_bars * 3)) / close.shift(periods_1d_in_bars * 3)

    # 方向スコア: 各TFの方向が揃うほど高スコア
    def sign(s):
        return s.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 加重スコア（短期ほど重要）
    momentum_score = sign(ret_4h) * 0.5 + sign(ret_1d) * 0.3 + sign(ret_3d) * 0.2

    return pd.DataFrame({
        'ret_4h': ret_4h,
        'ret_1d': ret_1d,
        'ret_3d': ret_3d,
        'momentum_score': momentum_score,
    }, index=bars.index)


def momentum_filter(bars: pd.DataFrame,
                    direction: str,
                    min_score: float = 0.3,
                    periods_4h: int = 4,
                    periods_1d: int = 24) -> pd.Series:
    """
    モメンタムが指定方向と一致する場合のみTrueを返す。

    Args:
        direction: 'long' または 'short'
        min_score: 最低スコア（0.3 = 少なくとも4hと1dが一致）

    Returns:
        pd.Series of bool
    """
    mtf = mtf_momentum(bars, periods_4h, periods_1d)
    score = mtf['momentum_score']

    if direction == 'long':
        return score >= min_score
    else:
        return score <= -min_score


# ===== 4. 時刻アノマリーフィルター =====

# XAUUSD 統計的高優位時間帯 (UTC)
# ロンドンオープン前後: 6-10h
# NYオープン前後: 12-16h
# 避けるべき時間: アジア深夜 1-5h, 週末前 Fri 18h+
GOLD_PRIME_HOURS = frozenset([6, 7, 8, 9, 13, 14, 15])
GOLD_AVOID_HOURS = frozenset([1, 2, 3, 4, 22, 23])  # 深夜アジア + 閉場直前


def time_anomaly_filter(bars: pd.DataFrame,
                        prime_hours: frozenset = GOLD_PRIME_HOURS,
                        avoid_weekends: bool = True) -> pd.Series:
    """
    時刻アノマリーフィルター。

    botter Advent Calendar 2024 時刻アノマリー記事より:
    - 市場が最もアクティブな時間帯(ロンドン・NY オープン)に限定
    - アジア深夜や週末前後を回避

    Returns:
        pd.Series of bool: True = エントリー許可時間帯
    """
    idx = bars.index
    hours = idx.hour
    weekdays = idx.dayofweek  # 0=月, 4=金, 5=土, 6=日

    # プライム時間帯に含まれるか
    in_prime = pd.Series(
        [h in prime_hours for h in hours],
        index=bars.index
    )

    # 回避時間帯に含まれるか
    in_avoid = pd.Series(
        [h in GOLD_AVOID_HOURS for h in hours],
        index=bars.index
    )

    # 週末フィルター（土日はゴールドほぼ休場）
    if avoid_weekends:
        is_weekend = pd.Series(weekdays >= 5, index=bars.index)
        # 金曜21時以降も危険
        is_fri_night = pd.Series(
            [(wd == 4 and h >= 20) for wd, h in zip(weekdays, hours)],
            index=bars.index
        )
        return in_prime & ~in_avoid & ~is_weekend & ~is_fri_night

    return in_prime & ~in_avoid


def time_score(bars: pd.DataFrame) -> pd.Series:
    """
    時刻ごとのスコア（分析用）。-1.0 ~ +1.0
    """
    idx = bars.index
    scores = pd.Series(0.0, index=bars.index)

    for i, ts in enumerate(idx):
        h = ts.hour
        wd = ts.dayofweek
        if wd >= 5:
            scores.iloc[i] = -1.0
        elif h in GOLD_PRIME_HOURS:
            scores.iloc[i] = 1.0
        elif h in GOLD_AVOID_HOURS:
            scores.iloc[i] = -0.5
        else:
            scores.iloc[i] = 0.0

    return scores


# ===== 5. 複合フィルタースコア =====

def composite_filter_score(bars: pd.DataFrame,
                            direction: str = None) -> pd.Series:
    """
    全フィルターを統合した複合スコア。

    スコア計算:
      - ボラティリティ適正: +1
      - トレンドレジーム一致: +2 (最重要)
      - モメンタム一致: +1
      - プライム時間帯: +1
      合計 0~5

    Returns:
        pd.Series: int 0~5
    """
    score = pd.Series(0, index=bars.index, dtype=int)

    # ボラティリティ
    vol_ok = volatility_regime(bars)
    score += vol_ok.astype(int)

    # トレンドレジーム
    t_regime = trend_regime_simple(bars)

    # プライム時間
    t_prime = time_anomaly_filter(bars).astype(int)
    score += t_prime

    if direction == 'long':
        score += (t_regime == 1).astype(int) * 2
        # モメンタム
        mtf = mtf_momentum(bars)
        score += (mtf['momentum_score'] >= 0.3).astype(int)

    elif direction == 'short':
        score += (t_regime == -1).astype(int) * 2
        mtf = mtf_momentum(bars)
        score += (mtf['momentum_score'] <= -0.3).astype(int)

    else:
        # 方向未指定: トレンドがある方向に加点
        score += (t_regime != 0).astype(int) * 2

    return score
