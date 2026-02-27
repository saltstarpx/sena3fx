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


# ==============================================================
# MTFカスケード戦略 (4H方向 → 1H確認 → 15min タイミング)
# ==============================================================

def sig_yagami_mtf_cascade(bars_dict: dict, min_grade: str = 'B'):
    """
    MTFカスケードシグナル。
    やがみ: 「1H/4Hをみて15分でエントリーポイントを模索、1分足でエントリー」

    仕組み:
      Step 1 (4H): EMA200 + 傾きでトレンド方向バイアスを決定
      Step 2 (1H): やがみ5条件シグナルで方向確認
      Step 3 (base): 反転ローソク足 / プライスアクションでタイミング確認

    Args:
        bars_dict: {'4h': df_4h, '1h': df_1h (任意)}
                   上位足OHLCデータ。
        min_grade: 1H やがみシグナルの最低グレード ('A'/'B')

    Returns:
        signal_func(base_bars) -> pd.Series
        backtest例: engine.run(bars_1h, sig_yagami_mtf_cascade({'4h': bars_4h}))
                    engine.run(bars_1h, sig_yagami_mtf_cascade({'4h': bars_4h}),
                               htf_bars=bars_4h)  # SLも4H基準に
    """
    from .filters import time_anomaly_filter
    from .candle import detect_single_candle, detect_price_action

    bars_4h = bars_dict.get('4h')
    bars_1h_ext = bars_dict.get('1h')  # 外部1Hデータ（任意）

    def _f(base_bars):
        n = len(base_bars)
        signals = pd.Series(index=base_bars.index, dtype=object)
        if n < 30:
            return signals

        # --- Step 1: 4H EMA200 トレンド方向の事前計算 ---
        htf_regime = pd.Series(0, index=base_bars.index, dtype=int)
        if bars_4h is not None and len(bars_4h) >= 50:
            ema200 = bars_4h['close'].ewm(span=200, adjust=False).mean()
            slope5 = ema200.diff(5)
            dir_4h = pd.Series(0, index=bars_4h.index, dtype=int)
            dir_4h.loc[(bars_4h['close'] > ema200) & (slope5 > 0)] = 1
            dir_4h.loc[(bars_4h['close'] < ema200) & (slope5 < 0)] = -1
            htf_regime = dir_4h.reindex(
                base_bars.index, method='ffill').fillna(0).astype(int)

        # --- Step 2: 1H やがみシグナルの事前計算 ---
        # 外部1Hデータがあればそちらを使用、なければ base_bars を 1H として扱う
        bars_1h_use = bars_1h_ext if bars_1h_ext is not None else base_bars
        sigs_1h = yagami_signal(bars_1h_use, freq='1h', min_grade=min_grade)
        # base_bars インデックスに ffill（最新の1Hシグナルを維持）
        aligned_1h = sigs_1h.reindex(base_bars.index, method='ffill')

        # --- Step 3: 時刻フィルター ---
        time_ok = time_anomaly_filter(base_bars)

        # --- Step 4: base_bars のローソク足品質（15min/1H タイミング確認） ---
        df = detect_single_candle(base_bars)
        df = detect_price_action(df)

        for i in range(20, n):
            sig_1h = aligned_1h.iloc[i]
            if sig_1h not in ('long', 'short'):
                continue

            if not time_ok.iloc[i]:
                continue

            # 4H バイアスフィルター（逆張り禁止）
            bias = htf_regime.iloc[i]
            if bias == 1 and sig_1h == 'short':
                continue
            if bias == -1 and sig_1h == 'long':
                continue

            # 15min タイミング確認: 反転ローソク足 or プライスアクション
            ctype = df['candle_type'].iloc[i]
            pa = df['pa_signal'].iloc[i]

            bull_confirm = ctype in ('big_bull', 'engulf_bull', 'hammer', 'pinbar_bull')
            bear_confirm = ctype in ('big_bear', 'engulf_bear', 'inv_hammer', 'pinbar_bear')
            pa_bull = pa in ('reversal_low', 'double_bottom',
                             'wick_fill_bull', 'body_align_support')
            pa_bear = pa in ('reversal_high', 'double_top',
                             'wick_fill_bear', 'body_align_resist')

            if sig_1h == 'long' and (bull_confirm or pa_bull):
                signals.iloc[i] = 'long'
            elif sig_1h == 'short' and (bear_confirm or pa_bear):
                signals.iloc[i] = 'short'

        return signals

    return _f


def sig_yagami_mtf_4h_1h(freq='1h'):
    """
    シンプルMTF: base_bars自体を1Hとして使い、
    同データから4H相当のEMA方向を計算してフィルタリング。
    bars_dict不要のスタンドアロン版。
    """
    from .filters import trend_regime_simple, time_anomaly_filter
    from .candle import detect_single_candle, detect_price_action

    def _f(base_bars):
        n = len(base_bars)
        signals = pd.Series(index=base_bars.index, dtype=object)
        if n < 50:
            return signals

        # 1H やがみシグナル
        base_sigs = yagami_signal(base_bars, freq=freq, min_grade='B')

        # 4H相当のEMA200方向（1Hデータを4H resampleして近似）
        bars_4h_approx = base_bars.resample('4h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        ).dropna()
        if len(bars_4h_approx) >= 50:
            ema200 = bars_4h_approx['close'].ewm(span=200, adjust=False).mean()
            slope = ema200.diff(5)
            dir_4h = pd.Series(0, index=bars_4h_approx.index, dtype=int)
            dir_4h.loc[(bars_4h_approx['close'] > ema200) & (slope > 0)] = 1
            dir_4h.loc[(bars_4h_approx['close'] < ema200) & (slope < 0)] = -1
            regime_4h = dir_4h.reindex(
                base_bars.index, method='ffill').fillna(0).astype(int)
        else:
            regime_4h = pd.Series(0, index=base_bars.index, dtype=int)

        time_ok = time_anomaly_filter(base_bars)

        for i in range(n):
            sig = base_sigs.iloc[i]
            if sig not in ('long', 'short'):
                continue
            if not time_ok.iloc[i]:
                continue
            bias = regime_4h.iloc[i]
            if bias == 1 and sig == 'short':
                continue
            if bias == -1 and sig == 'long':
                continue
            signals.iloc[i] = sig

        return signals

    return _f


# ==============================================================
# ブレイクアウト戦略 (レジサポ転換確認エントリー)
# ==============================================================

def sig_yagami_breakout(freq: str = '1h',
                        retest_window: int = 20,
                        retest_atr_mult: float = 0.4,
                        min_level_touches: int = 3,
                        confirm_candle: bool = True):
    """
    ブレイクアウト + レジサポ転換エントリー。
    やがみ: 「ブレイクアウトも必ず取ってください」
    「アジア時間のブレイクアウトは見送り」

    手順:
      1. S/Rレベルブレイク検出（終値が level ± ATR*0.2 を超える）
      2. ブレイク後 retest_window 本以内に価格が戻る（レジサポ転換）
      3. 確認ローソク足（反転系）でエントリー

    Args:
        retest_window: リテスト待機バー数（デフォルト20本）
        retest_atr_mult: リテスト許容距離 ATR倍数（デフォルト0.4）
        min_level_touches: 有効レベル最低タッチ数（デフォルト3）
        confirm_candle: 確認ローソク足を要求するか（Falseにすると即エントリー）
    """
    from .candle import detect_single_candle, detect_price_action
    from .timing import session_filter

    def _f(bars):
        n = len(bars)
        signals = pd.Series(index=bars.index, dtype=object)
        if n < 50:
            return signals

        # ATR
        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr, index=bars.index).rolling(14).mean()

        df = detect_single_candle(bars)
        df = detect_price_action(df)
        session = session_filter(bars)

        levels_cache = []
        levels_update_idx = -100
        # アクティブブレイク追跡: {key: {dir, level, break_idx}}
        active_breaks = {}

        for i in range(30, n):
            a = atr.iloc[i]
            if np.isnan(a) or a == 0:
                continue
            close_i = c[i]
            in_asia = session.iloc[i] == 'asia'

            # レジサポレベル更新（20本ごと）
            if i - levels_update_idx >= 20:
                levels_cache = extract_levels(
                    bars.iloc[max(0, i - 120):i + 1],
                    min_touches=min_level_touches
                )
                levels_update_idx = i

            # === ブレイクアウト検出（アジア時間除外） ===
            if not in_asia and levels_cache:
                for lv in levels_cache:
                    lvl = lv['level']
                    key = round(lvl, 2)
                    if key in active_breaks:
                        continue

                    # ブルブレイク: レジスタンスを終値上抜け
                    if lv['type'] == 'resistance' and close_i > lvl + a * 0.2:
                        active_breaks[key] = {
                            'dir': 'bull', 'level': lvl, 'break_idx': i
                        }
                    # ベアブレイク: サポートを終値下抜け
                    elif lv['type'] == 'support' and close_i < lvl - a * 0.2:
                        active_breaks[key] = {
                            'dir': 'bear', 'level': lvl, 'break_idx': i
                        }

            # === レジサポ転換リテスト判定 ===
            expired = []
            for key, brk in active_breaks.items():
                bars_since = i - brk['break_idx']

                # 待機ウィンドウ超過 → 無効化
                if bars_since > retest_window:
                    expired.append(key)
                    continue

                # ブレイク直後1本はスキップ
                if bars_since < 2:
                    continue

                lvl = brk['level']

                # リテスト: 価格がブレイクレベルに接近
                if abs(close_i - lvl) > a * retest_atr_mult:
                    continue

                ctype = df['candle_type'].iloc[i]
                pa = df['pa_signal'].iloc[i]

                if brk['dir'] == 'bull':
                    # 旧レジスタンス → 新サポート: ロング
                    # 価格がレベルの上側にあること
                    if close_i < lvl - a * 0.15:
                        continue
                    bull_ok = (not confirm_candle or
                               ctype in ('big_bull', 'engulf_bull', 'hammer',
                                         'pinbar_bull', 'pullback_bull') or
                               pa in ('reversal_low', 'wick_fill_bull', 'double_bottom'))
                    if bull_ok:
                        signals.iloc[i] = 'long'
                        expired.append(key)

                elif brk['dir'] == 'bear':
                    # 旧サポート → 新レジスタンス: ショート
                    if close_i > lvl + a * 0.15:
                        continue
                    bear_ok = (not confirm_candle or
                               ctype in ('big_bear', 'engulf_bear', 'inv_hammer',
                                         'pinbar_bear', 'pullback_bear') or
                               pa in ('reversal_high', 'wick_fill_bear', 'double_top'))
                    if bear_ok:
                        signals.iloc[i] = 'short'
                        expired.append(key)

            for key in expired:
                active_breaks.pop(key, None)

        return signals

    return _f


def sig_yagami_breakout_filtered(freq: str = '1h'):
    """
    ブレイクアウト + 全フィルター統合版。
    ボラティリティ・時刻・トレンドレジームの3フィルターを追加。
    """
    from .filters import volatility_regime, trend_regime_simple, time_anomaly_filter

    breakout_base = sig_yagami_breakout(freq, retest_window=20,
                                         retest_atr_mult=0.4,
                                         min_level_touches=3)

    def _f(bars):
        base_sigs = breakout_base(bars)

        vol_ok = volatility_regime(bars)
        t_regime = trend_regime_simple(bars)
        time_ok = time_anomaly_filter(bars)

        signals = pd.Series(index=bars.index, dtype=object)
        for i in range(len(bars)):
            sig = base_sigs.iloc[i]
            if sig not in ('long', 'short'):
                continue
            if not vol_ok.iloc[i]:
                continue
            if not time_ok.iloc[i]:
                continue
            regime = t_regime.iloc[i]
            if sig == 'long' and regime == -1:
                continue
            if sig == 'short' and regime == 1:
                continue
            signals.iloc[i] = sig

        return signals

    return _f


# ==============================================================
# マエダイメソッド: 背を近くして大きな値動きを捕まえる
# ==============================================================

def sig_maedai_breakout(freq: str = '1h',
                        lookback: int = 20,
                        atr_confirm: float = 0.0,
                        session_filter_on: bool = True):
    """
    マエダイ式ドンチャン・ブレイクアウト。

    「大きな時間軸のブレイクを背を近くして何度も挑戦して
     大きい値動きを取っていく」

    仕組み:
      - 直近 lookback 本の高値/安値 (Donchian Channel) を使用
      - 終値が上限を超えた → ロング
      - 終値が下限を割った → ショート
      - SL は ATR×0.8 (tight) — エンジン側 default_sl_atr=0.8 で設定
      - TP は ATR×10 以上の大きな利益を狙う — エンジン側 default_tp_atr=10.0

    Args:
        lookback: ドンチャン期間 (デフォルト20本)
        atr_confirm: ブレイク確認ATR倍数 (0=なし, 0.3=ATR×0.3以上のブレイク)
        session_filter_on: Trueの場合アジア時間を除外
    """
    from .timing import session_filter as _session_filter

    def _f(bars):
        n = len(bars)
        signals = pd.Series(index=bars.index, dtype=object)
        if n < lookback + 5:
            return signals

        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values

        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr, index=bars.index).rolling(14).mean()

        if session_filter_on:
            sess = _session_filter(bars)

        # 前バーまでの N 本高値/安値 (shift(1) で現在バーを含まない)
        rolling_high = bars['high'].rolling(lookback).max().shift(1)
        rolling_low  = bars['low'].rolling(lookback).min().shift(1)

        for i in range(lookback + 1, n):
            if np.isnan(rolling_high.iloc[i]) or np.isnan(rolling_low.iloc[i]):
                continue

            a = atr.iloc[i]
            if np.isnan(a) or a == 0:
                continue

            close_i = c[i]
            ch_high = rolling_high.iloc[i]
            ch_low  = rolling_low.iloc[i]

            if session_filter_on and sess.iloc[i] == 'asia':
                continue

            if atr_confirm > 0:
                if close_i > ch_high and (close_i - ch_high) >= atr_confirm * a:
                    signals.iloc[i] = 'long'
                elif close_i < ch_low and (ch_low - close_i) >= atr_confirm * a:
                    signals.iloc[i] = 'short'
            else:
                if close_i > ch_high:
                    signals.iloc[i] = 'long'
                elif close_i < ch_low:
                    signals.iloc[i] = 'short'

        return signals

    return _f


def sig_maedai_htf_breakout(lookback_htf: int = 10,
                             lookback_ltf: int = 5,
                             session_filter_on: bool = True):
    """
    マエダイ式 MTF ブレイクアウト。

    「1H/4Hをみて大きなブレイクを確認 → 同方向でタイトなエントリー」

    仕組み:
      - 4H足 (1Hから近似): lookback_htf 本ドンチャンでトレンド方向決定
      - 1H足: その方向に lookback_ltf 本のブレイクで精密エントリー
    """
    from .timing import session_filter as _sf

    def _f(base_bars):
        n = len(base_bars)
        signals = pd.Series(index=base_bars.index, dtype=object)
        if n < (lookback_htf * 4) + lookback_ltf + 10:
            return signals

        # 4H bars from 1H
        bars_4h = base_bars.resample('4h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        ).dropna()

        if len(bars_4h) < lookback_htf + 3:
            return signals

        dc_high_4h = bars_4h['high'].rolling(lookback_htf).max().shift(1)
        dc_low_4h  = bars_4h['low'].rolling(lookback_htf).min().shift(1)

        htf_dir = pd.Series(0, index=bars_4h.index, dtype=int)
        for j in range(lookback_htf + 1, len(bars_4h)):
            if pd.isna(dc_high_4h.iloc[j]):
                continue
            c4 = bars_4h['close'].iloc[j]
            if c4 > dc_high_4h.iloc[j]:
                htf_dir.iloc[j] = 1
            elif c4 < dc_low_4h.iloc[j]:
                htf_dir.iloc[j] = -1

        htf_dir_1h = htf_dir.reindex(base_bars.index, method='ffill').fillna(0).astype(int)

        # 1H timing breakout
        dc_high_1h = base_bars['high'].rolling(lookback_ltf).max().shift(1)
        dc_low_1h  = base_bars['low'].rolling(lookback_ltf).min().shift(1)

        sess = _sf(base_bars)

        for i in range(lookback_htf * 4 + lookback_ltf + 1, n):
            if pd.isna(dc_high_1h.iloc[i]):
                continue
            if session_filter_on and sess.iloc[i] == 'asia':
                continue

            htf_b = htf_dir_1h.iloc[i]
            close_i = base_bars['close'].iloc[i]

            if htf_b == 1 and close_i > dc_high_1h.iloc[i]:
                signals.iloc[i] = 'long'
            elif htf_b == -1 and close_i < dc_low_1h.iloc[i]:
                signals.iloc[i] = 'short'

        return signals

    return _f
