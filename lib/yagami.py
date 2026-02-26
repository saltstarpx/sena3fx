"""
やがみメソッド 5条件統合シグナルエンジン
=========================================
「全部の項目を複合的に考える」
「この中で3つか4つ以上条件を満たしているところだけエントリー」

5条件:
  1. ローソク足のレジサポの位置
  2. ローソク足の強弱
  3. プライスアクション
  4. チャートパターン
  5. ローソク足更新タイミング

評価:
  A評価(高品質): 4-5条件 → エントリー
  B評価(許容): 3条件 → 慎重にエントリー
  C評価(養分): 2条件以下 → エントリー禁止

追加フィルター（ポジり方の本）:
  - 二番底/二番天井を待つ
  - 急落急騰の逆張りは15分足で逆の色が出てから
  - トレンドレス（doji連続）はノーポジ
  - アジア時間のブレイクアウトは危険
"""
import numpy as np
import pandas as pd
from .candle import detect_single_candle, detect_price_action, detect_trendless
from .patterns import detect_chart_patterns
from .levels import extract_levels, is_at_level
from .timing import detect_bar_update_timing, session_filter


def analyze_bars(bars: pd.DataFrame, freq: str = '1h',
                 higher_bars: pd.DataFrame = None) -> pd.DataFrame:
    """
    全分析を統合してバーデータに列を追加。
    """
    # 1. ローソク足単体分析
    df = detect_single_candle(bars)
    # 2. プライスアクション
    df = detect_price_action(df)
    # 3. チャートパターン
    df = detect_chart_patterns(df)
    # 4. トレンドレス検出
    df['trendless'] = detect_trendless(bars)
    # 5. 足更新タイミング
    df['bar_update'] = detect_bar_update_timing(bars, freq)
    # 6. セッション
    df['session'] = session_filter(bars)

    return df


def yagami_signal(bars: pd.DataFrame, freq: str = '1h',
                  higher_bars: pd.DataFrame = None,
                  min_grade: str = 'B') -> pd.Series:
    """
    やがみメソッド5条件統合シグナル生成。

    Args:
      bars: OHLCデータ
      freq: 時間足
      higher_bars: 上位足データ（オプション）
      min_grade: 最低エントリーグレード ('A', 'B', 'C')

    Returns:
      pd.Series: 'long', 'short', None
    """
    df = analyze_bars(bars, freq, higher_bars)

    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    # ATR
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(14).mean().values

    signals = pd.Series(index=df.index, dtype=object)
    grades = pd.Series('', index=df.index, dtype=object)
    scores = pd.Series(0, index=df.index, dtype=int)

    # レジサポレベルは一度計算（20本ごとに更新）
    levels_cache = None
    levels_update_idx = -100

    min_score = {'A': 4, 'B': 3, 'C': 2}.get(min_grade, 3)

    for i in range(20, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        # トレンドレスはスキップ（やがみ: 「ポジション撤収」）
        if df['trendless'].iloc[i]:
            continue

        # アジア時間のブレイクアウトは危険（requirements_summary）
        session = df['session'].iloc[i]

        # レジサポ更新
        if i - levels_update_idx >= 10:
            levels_cache = extract_levels(df.iloc[max(0, i-100):i+1])
            levels_update_idx = i

        # ===== 5条件評価 =====
        score = 0
        direction = 0  # +1=ロング, -1=ショート

        # 条件1: レジサポの位置
        at_level, level_type = is_at_level(c[i], levels_cache, atr[i], 0.8)
        if at_level:
            score += 1
            if level_type == 'support':
                direction += 1
            elif level_type == 'resistance':
                direction -= 1

        # 条件2: ローソク足の強弱
        ctype = df['candle_type'].iloc[i]
        cstrength = df['candle_strength'].iloc[i]

        strong_bull_types = ('big_bull', 'engulf_bull', 'hammer', 'pinbar_bull')
        strong_bear_types = ('big_bear', 'engulf_bear', 'inv_hammer', 'pinbar_bear')
        if ctype in strong_bull_types:
            score += 1
            direction += 1
        elif ctype in strong_bear_types:
            score += 1
            direction -= 1
        elif ctype == 'pullback_bull':
            # 下降中の弱い陽線 → やがみ: 「次の足更新と同時にショート」
            direction -= 1
            score += 1
        elif ctype == 'pullback_bear':
            direction += 1
            score += 1

        # 条件3: プライスアクション
        pa = df['pa_signal'].iloc[i]
        pa_str = df['pa_strength'].iloc[i]

        if pa in ('reversal_low', 'double_bottom', 'body_align_support'):
            score += 1
            direction += 1
        elif pa in ('reversal_high', 'double_top', 'body_align_resist'):
            score += 1
            direction -= 1
        elif pa == 'wick_fill_bear':
            # やがみ: 「髭を埋めにくるムーブは即ポジションをフラットにするか持ち替え」
            score += 1
            direction -= 1
        elif pa == 'wick_fill_bull':
            score += 1
            direction += 1

        # 条件4: チャートパターン
        cp = df['chart_pattern'].iloc[i]
        if cp in ('inv_hs_long', 'flag_bull', 'wedge_bull',
                  'triangle_break_bull', 'ascending_tri'):
            score += 1
            direction += 1
        elif cp in ('hs_short', 'flag_bear', 'wedge_bear',
                    'triangle_break_bear', 'descending_tri'):
            score += 1
            direction -= 1

        # 条件5: 足更新タイミング
        if df['bar_update'].iloc[i]:
            score += 1
            # 上位足の方向に加点
            if higher_bars is not None and len(higher_bars) > 0:
                # 直近の上位足の色
                mask = higher_bars.index <= df.index[i]
                if mask.any():
                    last_htf = higher_bars.loc[mask].iloc[-1]
                    if last_htf['close'] > last_htf['open']:
                        direction += 0.5
                    elif last_htf['close'] < last_htf['open']:
                        direction -= 0.5

        # ===== グレード判定 =====
        scores.iloc[i] = score
        if score >= 4:
            grades.iloc[i] = 'A'
        elif score >= 3:
            grades.iloc[i] = 'B'
        else:
            grades.iloc[i] = 'C'

        # ===== シグナル生成 =====
        if score >= min_score:
            # アジア時間ブレイクアウトフィルター
            if session == 'asia' and cp and 'break' in str(cp):
                continue  # やがみ+requirements: アジア時間ブレイクアウトは危険

            # 大陽線/大陰線への逆張り禁止（やがみ: 「逆らうと死にます」）
            if ctype == 'big_bull' and direction < 0:
                continue
            if ctype == 'big_bear' and direction > 0:
                continue

            if direction > 0:
                signals.iloc[i] = 'long'
            elif direction < 0:
                signals.iloc[i] = 'short'

    return signals


def yagami_signal_with_details(bars: pd.DataFrame, freq: str = '1h',
                                higher_bars: pd.DataFrame = None,
                                min_grade: str = 'B') -> pd.DataFrame:
    """
    詳細情報付きのシグナル生成。バックテスト結果の分析用。
    """
    df = analyze_bars(bars, freq, higher_bars)
    sigs = yagami_signal(bars, freq, higher_bars, min_grade)
    df['signal'] = sigs
    return df


# ==============================================================
# 戦略バリエーション
# ==============================================================

def sig_yagami_A(freq='1h'):
    """A評価のみ（4-5条件）の厳選エントリー"""
    def _f(bars):
        return yagami_signal(bars, freq, min_grade='A')
    return _f


def sig_yagami_B(freq='1h'):
    """B評価以上（3条件以上）のエントリー"""
    def _f(bars):
        return yagami_signal(bars, freq, min_grade='B')
    return _f


def sig_yagami_reversal_only(freq='1h'):
    """リバーサルロー/ハイのみのエントリー（最強シグナル）"""
    def _f(bars):
        df = analyze_bars(bars, freq)
        signals = pd.Series(index=bars.index, dtype=object)
        for i in range(len(df)):
            pa = df['pa_signal'].iloc[i]
            if pa == 'reversal_low':
                signals.iloc[i] = 'long'
            elif pa == 'reversal_high':
                signals.iloc[i] = 'short'
        return signals
    return _f


def sig_yagami_double_bottom(freq='1h'):
    """ダブルボトム/トップ + 足更新タイミングのエントリー"""
    def _f(bars):
        df = analyze_bars(bars, freq)
        signals = pd.Series(index=bars.index, dtype=object)
        for i in range(len(df)):
            pa = df['pa_signal'].iloc[i]
            update = df['bar_update'].iloc[i]
            if pa == 'double_bottom' and update:
                signals.iloc[i] = 'long'
            elif pa == 'double_top' and update:
                signals.iloc[i] = 'short'
        return signals
    return _f


def sig_yagami_pattern_break(freq='1h'):
    """チャートパターンブレイク + レジサポ + 足更新のエントリー"""
    def _f(bars):
        df = analyze_bars(bars, freq)
        atr = pd.Series(
            np.maximum(df['high'] - df['low'],
                      np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                                np.abs(df['low'] - df['close'].shift(1))))
        ).rolling(14).mean()

        signals = pd.Series(index=bars.index, dtype=object)
        levels_cache = None

        for i in range(20, len(df)):
            cp = df['chart_pattern'].iloc[i]
            update = df['bar_update'].iloc[i]
            session = df['session'].iloc[i]

            if not cp or not update:
                continue
            if session == 'asia' and 'break' in str(cp):
                continue

            if i % 10 == 0:
                levels_cache = extract_levels(df.iloc[max(0, i-100):i+1])

            at_lv, _ = is_at_level(df['close'].iloc[i], levels_cache,
                                   atr.iloc[i] if not np.isnan(atr.iloc[i]) else 1.0)

            if cp in ('inv_hs_long', 'flag_bull', 'wedge_bull',
                      'triangle_break_bull', 'ascending_tri'):
                signals.iloc[i] = 'long'
            elif cp in ('hs_short', 'flag_bear', 'wedge_bear',
                        'triangle_break_bear', 'descending_tri'):
                signals.iloc[i] = 'short'

        return signals
    return _f


def sig_yagami_london_ny(freq='1h'):
    """ロンドン/NYセッション限定のやがみB評価以上"""
    def _f(bars):
        df = analyze_bars(bars, freq)
        full_signals = yagami_signal(bars, freq, min_grade='B')
        signals = pd.Series(index=bars.index, dtype=object)
        for i in range(len(df)):
            if df['session'].iloc[i] in ('london', 'ny') and full_signals.iloc[i]:
                signals.iloc[i] = full_signals.iloc[i]
        return signals
    return _f


# ==============================================================
# botter Advent Calendar 知見統合強化戦略
# ==============================================================

def sig_yagami_filtered(freq='1h', min_grade='B',
                        use_vol_regime=True,
                        use_trend_regime=True,
                        use_time_filter=True,
                        use_momentum=True):
    """
    botter Advent Calendar 2024/2025 知見を統合した強化版シグナル。

    追加フィルター:
      1. ボラティリティレジーム (ATR比率 0.6~2.2)
         - 「消えたエッジの話」(botter AC 2024 #22) より
      2. トレンドレジーム (EMA200方向)
         - ラリーウィリアムズ式 × botter コミュニティ
      3. 時刻アノマリー (ロンドン/NYオープン限定)
         - botter AC 2024 時刻アノマリー記事より
      4. MTFモメンタム (4h/1d リターン方向一致)
         - richmanbtc MLBot 特徴量設計思想より
    """
    from .filters import (volatility_regime, trend_regime_simple,
                          time_anomaly_filter, mtf_momentum)

    def _f(bars):
        # やがみベースシグナル
        base_signals = yagami_signal(bars, freq, min_grade=min_grade)

        # フィルター計算
        vol_ok = volatility_regime(bars) if use_vol_regime else pd.Series(True, index=bars.index)
        t_regime = trend_regime_simple(bars) if use_trend_regime else pd.Series(0, index=bars.index)
        time_ok = time_anomaly_filter(bars) if use_time_filter else pd.Series(True, index=bars.index)
        mtf = mtf_momentum(bars) if use_momentum else None

        signals = pd.Series(index=bars.index, dtype=object)

        for i in range(len(bars)):
            sig = base_signals.iloc[i]
            if not sig:
                continue

            # ボラティリティフィルター
            if use_vol_regime and not vol_ok.iloc[i]:
                continue

            # 時刻フィルター
            if use_time_filter and not time_ok.iloc[i]:
                continue

            # トレンドレジームフィルター（逆張り禁止）
            regime = t_regime.iloc[i]
            if use_trend_regime:
                if sig == 'long' and regime == -1:
                    continue  # ダウントレンドでロング禁止
                if sig == 'short' and regime == 1:
                    continue  # アップトレンドでショート禁止

            # MTFモメンタムフィルター
            if use_momentum and mtf is not None:
                mom_score = mtf['momentum_score'].iloc[i]
                if sig == 'long' and mom_score < -0.3:
                    continue  # モメンタムが逆
                if sig == 'short' and mom_score > 0.3:
                    continue

            signals.iloc[i] = sig

        return signals
    return _f


def sig_yagami_vol_regime(freq='1h'):
    """ボラティリティ適正帯のみのやがみB評価"""
    return sig_yagami_filtered(freq, min_grade='B',
                               use_vol_regime=True,
                               use_trend_regime=False,
                               use_time_filter=False,
                               use_momentum=False)


def sig_yagami_trend_regime(freq='1h'):
    """EMA200トレンド方向のみのやがみB評価"""
    return sig_yagami_filtered(freq, min_grade='B',
                               use_vol_regime=False,
                               use_trend_regime=True,
                               use_time_filter=False,
                               use_momentum=False)


def sig_yagami_prime_time(freq='1h'):
    """ロンドン/NYオープン時刻アノマリーフィルター + やがみB"""
    return sig_yagami_filtered(freq, min_grade='B',
                               use_vol_regime=False,
                               use_trend_regime=False,
                               use_time_filter=True,
                               use_momentum=False)


def sig_yagami_full_filter(freq='1h'):
    """全フィルター統合 (最厳選・高品質エントリー)"""
    return sig_yagami_filtered(freq, min_grade='B',
                               use_vol_regime=True,
                               use_trend_regime=True,
                               use_time_filter=True,
                               use_momentum=True)


def sig_yagami_A_full_filter(freq='1h'):
    """A評価 + 全フィルター統合 (最高品質)"""
    return sig_yagami_filtered(freq, min_grade='A',
                               use_vol_regime=True,
                               use_trend_regime=True,
                               use_time_filter=True,
                               use_momentum=True)

