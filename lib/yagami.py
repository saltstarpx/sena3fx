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


# ==============================================================
# マエダイメソッド v2: レンジ品質 + 後ノリ + パターン類似
# ==============================================================

def _calc_range_quality(bars, idx, lookback=20):
    """
    ブレイク前のレンジ品質スコア (0.0 〜 1.0)。

    スコア構成:
      - ATR圧縮率 (short_atr / long_atr が低い = コイリング)
      - レンジ継続期間 (長いほど蓄積エネルギー大)
      - レンジの狭さ (高値-安値 / ATR が小さい)

    高スコア = 強いブレイク候補。
    """
    start = max(0, idx - lookback)
    sub = bars.iloc[start:idx + 1]
    if len(sub) < 10:
        return 0.5  # デフォルト

    h = sub['high'].values
    l = sub['low'].values
    c = sub['close'].values

    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr_s = float(pd.Series(tr).rolling(5).mean().iloc[-1])   # 短期ATR
    atr_l = float(pd.Series(tr).rolling(14).mean().iloc[-1])  # 長期ATR
    if atr_l == 0:
        return 0.5

    # ATR圧縮スコア: 短期が長期より小さいほど高評価
    compression_score = max(0.0, 1.0 - atr_s / atr_l)

    # レンジ幅スコア: 高値-安値 / (ATR × sqrt(lookback)) で正規化
    # sqrt(lookback) でlookback長さに関わらず公平なスコアに
    range_width = max(h) - min(l)
    expected_range = atr_l * (lookback ** 0.5) * 0.8  # 期待レンジ幅
    range_score = max(0.0, 1.0 - (range_width / max(expected_range, atr_l)))

    # レンジ継続スコア: もし均衡状態が長く続いているほど高評価
    # 終値がレンジの中央±20%に収まっているバー数をカウント
    mid = (max(h) + min(l)) / 2
    half = (max(h) - min(l)) * 0.3
    in_range = np.sum(np.abs(c - mid) < half)
    duration_score = min(1.0, in_range / max(lookback * 0.5, 1))

    score = (compression_score * 0.4 + range_score * 0.3 + duration_score * 0.3)
    return float(np.clip(score, 0.0, 1.0))


def sig_maedai_breakout_v2(freq: str = '1h',
                            lookback: int = 20,
                            entry_mode: str = 'retest',
                            require_compression: bool = True,
                            compression_ratio: float = 0.92,
                            min_range_score: float = 0.20,
                            use_patterns: bool = True,
                            retest_tolerance: float = 0.4,
                            retest_window: int = 10,
                            pullback_window: int = 6,
                            session_filter_on: bool = True):
    """
    マエダイ式 最適化ブレイクアウト v2。

    「後ノリでも行ける」「背が近い」「過去のパターンに似ている」

    改善点:
      1. レンジ品質スコア: ATR圧縮率 + 継続期間でブレイクエネルギーを評価
      2. エントリーモード:
           'immediate' — ブレイクバー即エントリー
           'next_bar'  — 次バー同方向確認後エントリー (ダマシ削減)
           'retest'    — ブレイク後リテスト待ちエントリー (タイトSL)
           'pullback'  — 後ノリ: ブレイク後の押し/戻りで追従エントリー
      3. チャートパターン確認: flag/wedge/triangle/ascending_tri で信頼度加算

    Args:
        entry_mode: 'immediate' | 'next_bar' | 'retest' | 'pullback'
        require_compression: ブレイク前ATR圧縮を必須とするか
        compression_ratio: short_atr/long_atr の閾値 (デフォルト0.85)
        min_range_score: 必要な最低レンジ品質スコア (0.0〜1.0)
        use_patterns: チャートパターンによるスコア加算
        retest_tolerance: リテスト許容距離 (ATR倍数)
        retest_window: retest/pullbackモードの待機バー数
        pullback_window: pullbackモードの押し目待ち期間
    """
    from .candle import detect_single_candle, detect_price_action
    from .patterns import detect_chart_patterns
    from .timing import session_filter as _sf

    BULL_PATTERNS = ('flag_bull', 'wedge_bull', 'triangle_break_bull',
                     'ascending_tri', 'inv_hs_long')
    BEAR_PATTERNS = ('flag_bear', 'wedge_bear', 'triangle_break_bear',
                     'descending_tri', 'hs_short')

    def _f(bars):
        n = len(bars)
        signals = pd.Series(index=bars.index, dtype=object)
        if n < lookback + 20:
            return signals

        # === 事前計算 ===
        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values

        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr_s = pd.Series(tr, index=bars.index).rolling(5).mean()
        atr_l = pd.Series(tr, index=bars.index).rolling(20).mean()
        atr14 = pd.Series(tr, index=bars.index).rolling(14).mean()

        # チャートパターン (use_patterns=True の場合)
        if use_patterns:
            df_pat = detect_chart_patterns(bars)
            patterns = df_pat['chart_pattern'].values
        else:
            patterns = np.array([None] * n)

        # ローソク足品質
        df_c = detect_single_candle(bars)
        df_c = detect_price_action(df_c)

        if session_filter_on:
            sess = _sf(bars)

        # Donchian (前バーまで)
        rolling_high = bars['high'].rolling(lookback).max().shift(1)
        rolling_low  = bars['low'].rolling(lookback).min().shift(1)

        # ATR圧縮判定
        # NOTE: ブレイクバー自身ではなく 1〜3 本前で判定
        # (ブレイクバーは急騰/急落でATR(5)が跳ね上がるため)
        def is_compressed(i):
            if not require_compression:
                return True
            # ブレイク前 (i-2 ≈ レンジ最後のバー) で圧縮状態を確認
            check_i = max(0, i - 2)
            a_s = atr_s.iloc[check_i]
            a_l = atr_l.iloc[check_i]
            if np.isnan(a_s) or np.isnan(a_l) or a_l == 0:
                return False
            return (a_s / a_l) <= compression_ratio

        # パターンスコア加算
        def pattern_confirms(i, direction):
            if not use_patterns:
                return True, 0.0
            pat = patterns[i] if i < len(patterns) else None
            if direction == 'long' and pat in BULL_PATTERNS:
                return True, 1.0
            if direction == 'short' and pat in BEAR_PATTERNS:
                return True, 1.0
            # パターンなしでも通過 (スコア0)
            return True, 0.0

        # アクティブブレイク追跡
        # {key: {dir, level, break_idx, break_atr, bar_break_high, bar_break_low}}
        active_breaks = {}

        for i in range(lookback + 1, n):
            a = atr14.iloc[i]
            if np.isnan(a) or a == 0:
                continue
            if np.isnan(rolling_high.iloc[i]):
                continue
            if session_filter_on and sess.iloc[i] == 'asia':
                continue

            close_i = c[i]
            ch_high = rolling_high.iloc[i]
            ch_low  = rolling_low.iloc[i]

            # === ブレイクアウト検出 ===
            new_break = None
            if close_i > ch_high:
                rq = _calc_range_quality(bars, i - 1, lookback)
                if rq >= min_range_score and is_compressed(i):
                    new_break = ('bull', ch_high, i, a,
                                 bars['high'].iloc[i], bars['low'].iloc[i])
            elif close_i < ch_low:
                rq = _calc_range_quality(bars, i - 1, lookback)
                if rq >= min_range_score and is_compressed(i):
                    new_break = ('bear', ch_low, i, a,
                                 bars['high'].iloc[i], bars['low'].iloc[i])

            # immediate: ブレイクバーで即エントリー
            if new_break and entry_mode == 'immediate':
                direction = 'long' if new_break[0] == 'bull' else 'short'
                _, pat_score = pattern_confirms(i, direction)
                signals.iloc[i] = direction

            # next_bar: 追跡リストに追加して次バーで確認
            elif new_break and entry_mode in ('next_bar', 'retest', 'pullback'):
                brk_dir, brk_level, brk_idx, brk_atr, brk_bar_high, brk_bar_low = new_break
                key = (round(brk_level, 1), brk_idx)
                active_breaks[key] = {
                    'dir': brk_dir,
                    'level': brk_level,
                    'break_idx': brk_idx,
                    'break_atr': brk_atr,
                    'bar_high': brk_bar_high,
                    'bar_low': brk_bar_low,
                }

            # === アクティブブレイクの後処理 ===
            expired = []
            for key, brk in active_breaks.items():
                bars_since = i - brk['break_idx']
                if bars_since == 0:
                    continue

                win = retest_window if entry_mode in ('retest',) else pullback_window
                if bars_since > win:
                    expired.append(key)
                    continue

                brk_dir  = brk['dir']
                brk_lv   = brk['level']
                brk_atr  = brk['break_atr']
                direction = 'long' if brk_dir == 'bull' else 'short'
                ctype = df_c['candle_type'].iloc[i]
                pa    = df_c['pa_signal'].iloc[i]

                if entry_mode == 'next_bar':
                    # 次バーが同方向に動いていれば即エントリー
                    if bars_since == 1:
                        if brk_dir == 'bull' and close_i > brk['bar_high'] * 0.999:
                            signals.iloc[i] = 'long'
                            expired.append(key)
                        elif brk_dir == 'bear' and close_i < brk['bar_low'] * 1.001:
                            signals.iloc[i] = 'short'
                            expired.append(key)

                elif entry_mode == 'retest':
                    # レベルにリテストが来た時に確認ローソク足でエントリー
                    dist = abs(close_i - brk_lv)
                    if dist <= brk_atr * retest_tolerance:
                        # 価格がブレイクの反対側に逝かないこと
                        if brk_dir == 'bull' and close_i < brk_lv - brk_atr * 0.2:
                            expired.append(key)
                            continue
                        if brk_dir == 'bear' and close_i > brk_lv + brk_atr * 0.2:
                            expired.append(key)
                            continue

                        bull_ok = ctype in ('big_bull', 'engulf_bull', 'hammer',
                                            'pinbar_bull', 'pullback_bull') or \
                                  pa in ('reversal_low', 'wick_fill_bull', 'double_bottom')
                        bear_ok = ctype in ('big_bear', 'engulf_bear', 'inv_hammer',
                                            'pinbar_bear', 'pullback_bear') or \
                                  pa in ('reversal_high', 'wick_fill_bear', 'double_top')
                        if brk_dir == 'bull' and bull_ok:
                            signals.iloc[i] = 'long'
                            expired.append(key)
                        elif brk_dir == 'bear' and bear_ok:
                            signals.iloc[i] = 'short'
                            expired.append(key)

                elif entry_mode == 'pullback':
                    # 後ノリ: ブレイク後に一度押し/戻りが入ったら追従エントリー
                    # 押し目の最安値 (ブレイク後の最安/最高) を取ってから
                    # 再び方向に動き出したらエントリー
                    recent_slice = bars.iloc[brk['break_idx']:i + 1]
                    if len(recent_slice) < 3:
                        continue

                    if brk_dir == 'bull':
                        # ブレイク後に一度安値をつけてから上昇再開
                        recent_low = recent_slice['low'].min()
                        # 押しが入っている (最安値がブレイクバーのcloseより低い)
                        had_pullback = recent_low < bars['close'].iloc[brk['break_idx']]
                        # 現在は再上昇 (直近2本が上昇)
                        rising = close_i > bars['close'].iloc[i - 1]
                        if had_pullback and rising:
                            # SLを押し目安値近くに置けるので背が近い
                            bull_ok = ctype in ('big_bull', 'engulf_bull', 'hammer',
                                                'pinbar_bull') or \
                                      pa in ('reversal_low', 'wick_fill_bull')
                            if bull_ok or bars_since >= 2:
                                signals.iloc[i] = 'long'
                                expired.append(key)
                    else:
                        # ブレイク後に一度高値をつけてから下降再開
                        recent_high = recent_slice['high'].max()
                        had_pullback = recent_high > bars['close'].iloc[brk['break_idx']]
                        falling = close_i < bars['close'].iloc[i - 1]
                        if had_pullback and falling:
                            bear_ok = ctype in ('big_bear', 'engulf_bear', 'inv_hammer',
                                                'pinbar_bear') or \
                                      pa in ('reversal_high', 'wick_fill_bear')
                            if bear_ok or bars_since >= 2:
                                signals.iloc[i] = 'short'
                                expired.append(key)

            for key in expired:
                active_breaks.pop(key, None)

        return signals

    return _f


def sig_maedai_best(freq: str = '1h'):
    """
    マエダイ方式 推奨設定:
    - retest モード (後ノリ + タイトSL)
    - ATR圧縮フィルター ON
    - チャートパターン確認 ON
    - 4H方向フィルター込み
    """
    from .filters import time_anomaly_filter

    breakout_fn = sig_maedai_breakout_v2(
        freq=freq,
        lookback=20,
        entry_mode='retest',
        require_compression=True,
        compression_ratio=0.85,
        min_range_score=0.3,
        use_patterns=True,
        retest_tolerance=0.5,
        retest_window=10,
        session_filter_on=True,
    )

    # 4H方向フィルター
    def _f(base_bars):
        base_sigs = breakout_fn(base_bars)
        time_ok = time_anomaly_filter(base_bars)

        # 4H approx
        bars_4h = base_bars.resample('4h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        ).dropna()
        if len(bars_4h) >= 50:
            ema200 = bars_4h['close'].ewm(span=200, adjust=False).mean()
            slope  = ema200.diff(5)
            dir_4h = pd.Series(0, index=bars_4h.index, dtype=int)
            dir_4h.loc[(bars_4h['close'] > ema200) & (slope > 0)] = 1
            dir_4h.loc[(bars_4h['close'] < ema200) & (slope < 0)] = -1
            regime = dir_4h.reindex(base_bars.index, method='ffill').fillna(0).astype(int)
        else:
            regime = pd.Series(0, index=base_bars.index, dtype=int)

        signals = pd.Series(index=base_bars.index, dtype=object)
        for i in range(len(base_bars)):
            sig = base_sigs.iloc[i]
            if sig not in ('long', 'short'):
                continue
            if not time_ok.iloc[i]:
                continue
            b = regime.iloc[i]
            if b == 1 and sig == 'short':
                continue
            if b == -1 and sig == 'long':
                continue
            signals.iloc[i] = sig

        return signals

    return _f


def sig_maedai_htf_pullback(lookback_htf: int = 10,
                              pullback_bars: int = 5,
                              session_filter_on: bool = True):
    """
    マエダイ式 HTF方向 × 後ノリ (最強の組み合わせ)。

    仕組み:
      1. 4H Donchian ブレイクで方向確定 (大きな足のブレイク)
      2. 1Hで押し目/戻りが入るのを待つ (後ノリ・タイトSL)
      3. 押し目から再び方向に動いた瞬間でエントリー
         → SL = 押し目の安値/高値 (非常に背が近い)

    やがみ: 「二番底/二番天井を待ってからエントリー」に近い思想
    マエダイ: 「後ノリでも行ける場合がある」
    """
    from .candle import detect_single_candle, detect_price_action
    from .timing import session_filter as _sf

    def _f(base_bars):
        n = len(base_bars)
        signals = pd.Series(index=base_bars.index, dtype=object)
        if n < lookback_htf * 4 + pullback_bars + 10:
            return signals

        # 4H から Donchian 方向
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

        # 1H ローソク足品質
        df_c = detect_single_candle(base_bars)
        df_c = detect_price_action(df_c)

        # ATR
        hv = base_bars['high'].values
        lv = base_bars['low'].values
        cv = base_bars['close'].values
        tr = np.maximum(hv - lv, np.maximum(
            np.abs(hv - np.roll(cv, 1)), np.abs(lv - np.roll(cv, 1))))
        tr[0] = hv[0] - lv[0]
        atr14 = pd.Series(tr, index=base_bars.index).rolling(14).mean()

        sess = _sf(base_bars)

        for i in range(lookback_htf * 4 + pullback_bars + 1, n):
            if session_filter_on and sess.iloc[i] == 'asia':
                continue

            htf_b = htf_dir_1h.iloc[i]
            if htf_b == 0:
                continue

            a = atr14.iloc[i]
            if np.isnan(a) or a == 0:
                continue

            close_i = base_bars['close'].iloc[i]
            ctype = df_c['candle_type'].iloc[i]
            pa    = df_c['pa_signal'].iloc[i]

            # 直近 pullback_bars のローを取得 (押し目確認)
            recent = base_bars.iloc[i - pullback_bars:i + 1]
            recent_lows  = recent['low'].values
            recent_highs = recent['high'].values

            if htf_b == 1:  # 4H 上昇方向 → 押し目ロング
                # 押し目: 直近にいくつか安値更新があった
                had_pullback = recent_lows.min() < base_bars['close'].iloc[i - pullback_bars]
                # 押し目から回復: 現在が直近 N 本の高値を上抜け
                pullback_low = recent_lows.min()
                recovering = close_i > np.percentile(recent['close'].values, 60)
                bull_candle = ctype in ('big_bull', 'engulf_bull', 'hammer',
                                        'pinbar_bull', 'pullback_bull') or \
                              pa in ('reversal_low', 'double_bottom', 'wick_fill_bull')

                if had_pullback and recovering and bull_candle:
                    signals.iloc[i] = 'long'

            else:  # htf_b == -1: 4H 下降方向 → 戻り売り
                had_pullback = recent_highs.max() > base_bars['close'].iloc[i - pullback_bars]
                recovering = close_i < np.percentile(recent['close'].values, 40)
                bear_candle = ctype in ('big_bear', 'engulf_bear', 'inv_hammer',
                                        'pinbar_bear', 'pullback_bear') or \
                              pa in ('reversal_high', 'double_top', 'wick_fill_bear')

                if had_pullback and recovering and bear_candle:
                    signals.iloc[i] = 'short'

        return signals

    return _f


# ──────────────────────────────────────────────────────
# マエダイ D1 戦略: ドンチャン30 + EMA200 トレンドフィルター
# ──────────────────────────────────────────────────────

def sig_maedai_d1_dc30(lookback=30, ema_period=200, session_filter_on=False):
    """
    D1足 ドンチャンチャンネルブレイク + EMA200トレンドフィルター。

    マエダイメソッドを日足レベルで実装:
    - 大きな足でのみトレード (D1 = 真のトレンド判定)
    - EMA200 で長期方向を確認 (上抜け→ロング方向のみ、下抜け→ショート方向のみ)
    - 30日高値/安値ブレイクでエントリー

    推奨エンジン設定:
    - exit_on_signal=False (SL/TPのみ決済)
    - trail_start_atr=4, trail_dist_atr=3
    - SL=ATR×0.8以上, TP=ATR×6以上

    Args:
        lookback: ドンチャンチャンネル期間 (デフォルト30日)
        ema_period: トレンドフィルターEMA期間 (デフォルト200日)
        session_filter_on: セッションフィルター (D1は無効化を推奨)
    """
    def _f(bars):
        c = bars['close']
        h = bars['high']
        l = bars['low']

        # ドンチャンチャンネル (前日までの高値/安値)
        dc_hi = h.shift(1).rolling(lookback).max()
        dc_lo = l.shift(1).rolling(lookback).min()

        # EMA200 トレンドフィルター
        ema = c.ewm(span=ema_period, adjust=False).mean()

        signals = pd.Series('flat', index=bars.index)

        # ロング: 高値ブレイク + EMA200上抜け
        long_mask  = (c > dc_hi) & (c > ema)
        # ショート: 安値ブレイク + EMA200下抜け
        short_mask = (c < dc_lo) & (c < ema)

        signals[long_mask]  = 'long'
        signals[short_mask] = 'short'

        return signals

    return _f


def sig_maedai_d1_dc_multi(lookback=30, ema_period=200,
                            confirm_close=True):
    """
    D1足 ドンチャン + EMA200 + クローズ確認。

    confirm_close=True: ブレイクの翌日にも同方向クローズで確認
    (ダマシブレイク削減)
    """
    def _f(bars):
        c = bars['close']
        h = bars['high']
        l = bars['low']

        dc_hi = h.shift(1).rolling(lookback).max()
        dc_lo = l.shift(1).rolling(lookback).min()
        ema   = c.ewm(span=ema_period, adjust=False).mean()

        raw_long  = (c > dc_hi) & (c > ema)
        raw_short = (c < dc_lo) & (c < ema)

        signals = pd.Series('flat', index=bars.index)

        if confirm_close:
            # 翌日もブレイク方向維持で確認 (1バーラグ)
            confirm_long  = raw_long  & raw_long.shift(1).fillna(False)
            confirm_short = raw_short & raw_short.shift(1).fillna(False)
            signals[confirm_long]  = 'long'
            signals[confirm_short] = 'short'
        else:
            signals[raw_long]  = 'long'
            signals[raw_short] = 'short'

        return signals

    return _f


# ──────────────────────────────────────────────────────
# 汎用マルチTF: ドンチャン(日数指定) + D1 EMA200 トレンドフィルター
# ──────────────────────────────────────────────────────

def sig_maedai_dc_ema_tf(freq='4h', lookback_days=30, ema_days=200,
                          confirm_bars=1, atr_confirm=False):
    """
    任意時間軸 × ドンチャン(日数指定) + D1 EMA200 トレンドフィルター。

    lookback_days で「何日間の高値/安値を抜けたらエントリー」を指定し、
    内部でbars数に変換。D1 EMA200 は常に日足基準で計算。

    Args:
        freq: '4h' | '8h' | '12h' | '1d' など
        lookback_days: 高値/安値ブレイクの参照期間(日数)
        ema_days: トレンドフィルターEMAの期間(日数)
        confirm_bars: ブレイク確認バー数 (1=翌バーも同方向で確認)
        atr_confirm: Trueでブレイクバーが平均より大きいATRか確認
    """
    BARS_PER_DAY = {'1h': 24, '2h': 12, '4h': 6, '6h': 4,
                    '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd = BARS_PER_DAY.get(freq, 1)
        lb  = max(5, lookback_days * bpd)      # ドンチャン期間 (bars)
        ema_n = max(20, ema_days * bpd)         # EMA期間 (bars)

        c = bars['close']
        h = bars['high']
        l = bars['low']

        # ドンチャン (前バー高値/安値 = 当日ブレイクをリアルタイム検出)
        dc_hi = h.shift(1).rolling(lb).max()
        dc_lo = l.shift(1).rolling(lb).min()

        # EMA (同足で計算; D1相当の長期バイアス)
        ema = c.ewm(span=ema_n, adjust=False).mean()

        raw_long  = (c > dc_hi) & (c > ema)
        raw_short = (c < dc_lo) & (c < ema)

        # ATR確認 (ブレイクバーが直近ATRより大きい動き)
        if atr_confirm:
            atr_ser = (h - l).rolling(14).mean()
            bar_range = h - l
            raw_long  = raw_long  & (bar_range > atr_ser * 0.8)
            raw_short = raw_short & (bar_range > atr_ser * 0.8)

        signals = pd.Series('flat', index=bars.index)

        if confirm_bars >= 1:
            # confirm_bars 本連続でブレイク方向を確認
            mask_long  = raw_long.copy()
            mask_short = raw_short.copy()
            for lag in range(1, confirm_bars + 1):
                mask_long  = mask_long  & raw_long.shift(lag).fillna(False)
                mask_short = mask_short & raw_short.shift(lag).fillna(False)
            signals[mask_long]  = 'long'
            signals[mask_short] = 'short'
        else:
            signals[raw_long]  = 'long'
            signals[raw_short] = 'short'

        return signals

    return _f


# ──────────────────────────────────────────────────────
# RSI 押し目エントリー (ユーザー取引履歴から逆算)
# ──────────────────────────────────────────────────────

def sig_rsi_pullback_tf(freq='4h', ema_days=200, rsi_period=14,
                         rsi_oversold=45, rsi_overbought=55):
    """
    EMA200 上昇トレンド中の RSI 押し目エントリー。

    ユーザー取引履歴分析から逆算した戦略:
    - 金相場の上昇トレンド中、押し目 (RSI 40-50 圏) を買う
    - ブレイクアウトだけでなく「押し目での拾い」も行う
    - EMA200 上抜けを維持している間のみロング方向

    Long条件:
      - close > EMA200 (上昇トレンド維持)
      - RSI が rsi_oversold 以下から上抜け (押し目から回復)

    Short条件:
      - close < EMA200 (下降トレンド)
      - RSI が rsi_overbought 以上から下抜け (戻りから売り)
      ※ long_biased=True のエンジンではショートは大幅下落時のみ許可

    Args:
        freq: 時間軸 ('4h' | '8h' | '12h' | '1d' 等)
        ema_days: EMAの日数指定 (デフォルト200日)
        rsi_period: RSI算出期間 (デフォルト14)
        rsi_oversold: この値以下から上抜けでロングシグナル (デフォルト45)
        rsi_overbought: この値以上から下抜けでショートシグナル (デフォルト55)
    """
    BARS_PER_DAY = {'1h': 24, '2h': 12, '4h': 6, '6h': 4,
                    '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd = BARS_PER_DAY.get(freq, 1)
        ema_n = max(20, ema_days * bpd)

        c = bars['close']

        # EMA200 トレンドフィルター
        ema = c.ewm(span=ema_n, adjust=False).mean()

        # RSI (Wilder平滑化)
        delta    = c.diff()
        avg_gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # RSIクロス検出 (前バー≤閾値 & 現バー>閾値)
        rsi_cross_up   = (rsi.shift(1) <= rsi_oversold)  & (rsi > rsi_oversold)
        rsi_cross_down = (rsi.shift(1) >= rsi_overbought) & (rsi < rsi_overbought)

        long_mask  = (c > ema) & rsi_cross_up
        short_mask = (c < ema) & rsi_cross_down

        signals = pd.Series('flat', index=bars.index)
        signals[long_mask]  = 'long'
        signals[short_mask] = 'short'

        return signals

    return _f


# ──────────────────────────────────────────────────────
# DC + ADX + RSI 複合フィルター戦略
# ──────────────────────────────────────────────────────

def sig_dc_adx_rsi_tf(freq='4h', lookback_days=15, ema_days=200,
                       adx_period=14, adx_min=20,
                       rsi_max_long=70, rsi_min_short=30,
                       confirm_bars=1):
    """
    ドンチャンブレイク + ADXトレンド強度フィルター + RSI 過買い/過売りフィルター。

    ユーザー実績: "もっと使えるインジがあればそれを使ってもいいです"

    既存の DC+EMA 戦略に2つのフィルターを追加:
    1. ADX(14) > adx_min: トレンド相場でのみブレイクアウトに乗る
       → ADX低い = レンジ相場 → ダマシブレイクが多い → スキップ
    2. RSI フィルター:
       - ロング: RSI > rsi_max_long (過買い) なら見送り
       - ショート: RSI < rsi_min_short (過売り) なら見送り
       → 既に動きすぎた後のブレイクは危険

    Args:
        freq: 時間軸
        lookback_days: ドンチャンの参照期間(日数)
        ema_days: EMAの期間(日数)
        adx_period: ADXの算出期間
        adx_min: この値以上のADXでのみエントリー (デフォルト20)
        rsi_max_long: ロング時RSI上限 (デフォルト70)
        rsi_min_short: ショート時RSI下限 (デフォルト30)
        confirm_bars: ブレイク確認バー数
    """
    BARS_PER_DAY = {'1h': 24, '2h': 12, '4h': 6, '6h': 4,
                    '8h': 3, '12h': 2, '1d': 1}

    def _calc_adx_series(bars, period):
        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values

        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]

        up   = np.diff(h, prepend=h[0])
        down = -np.diff(l, prepend=l[0])
        dm_plus  = np.where((up > down) & (up > 0), up, 0.0)
        dm_minus = np.where((down > up) & (down > 0), down, 0.0)

        atr_s  = pd.Series(tr).ewm(com=period - 1, adjust=False).mean()
        dmp_s  = pd.Series(dm_plus).ewm(com=period - 1, adjust=False).mean()
        dmm_s  = pd.Series(dm_minus).ewm(com=period - 1, adjust=False).mean()

        di_plus  = 100 * dmp_s / atr_s.replace(0, 1e-10)
        di_minus = 100 * dmm_s / atr_s.replace(0, 1e-10)
        dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-10)
        adx = dx.ewm(com=period - 1, adjust=False).mean()

        return pd.Series(adx.values, index=bars.index)

    def _f(bars):
        bpd   = BARS_PER_DAY.get(freq, 1)
        lb    = max(5, lookback_days * bpd)
        ema_n = max(20, ema_days * bpd)

        c = bars['close']
        h = bars['high']
        l = bars['low']

        dc_hi = h.shift(1).rolling(lb).max()
        dc_lo = l.shift(1).rolling(lb).min()

        ema = c.ewm(span=ema_n, adjust=False).mean()
        adx = _calc_adx_series(bars, adx_period)

        delta    = c.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        raw_long  = (c > dc_hi) & (c > ema) & (adx >= adx_min) & (rsi <= rsi_max_long)
        raw_short = (c < dc_lo) & (c < ema) & (adx >= adx_min) & (rsi >= rsi_min_short)

        signals = pd.Series('flat', index=bars.index)

        if confirm_bars >= 1:
            mask_long  = raw_long.copy()
            mask_short = raw_short.copy()
            for lag in range(1, confirm_bars + 1):
                mask_long  = mask_long  & raw_long.shift(lag).fillna(False)
                mask_short = mask_short & raw_short.shift(lag).fillna(False)
            signals[mask_long]  = 'long'
            signals[mask_short] = 'short'
        else:
            signals[raw_long]  = 'long'
            signals[raw_short] = 'short'

        return signals

    return _f


# ──────────────────────────────────────────────────────
# マエダイ×やがみ 複合シグナル (ユーザー提案: OR統合+両方で厚くする)
# ──────────────────────────────────────────────────────

def sig_maedai_yagami_union(freq='4h', lookback_days=15, ema_days=200,
                              confirm_bars=2, rsi_oversold=45):
    """
    マエダイ(DCブレイク) OR RSI押し目 のいずれか点灯でエントリー。

    ユーザー提案: 「どちらかが反応したらポジションを取って、
                   両方反応したら厚くはればいいんじゃないか」

    実装:
    1. DCブレイク+EMA200 (マエダイ式) が点灯 → エントリー
    2. RSI押し目+EMA200 が点灯            → エントリー
    3. どちらか1つで通常エントリー (OR)
    4. 「厚くはる」はエンジンのピラミッド設定で自動対応
       (含み益 2ATR ごとに自動追加 → 強トレンド時に自然に積み上がる)
    """
    BARS_PER_DAY = {'1h': 24, '2h': 12, '4h': 6, '6h': 4,
                    '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd   = BARS_PER_DAY.get(freq, 1)
        lb    = max(5, lookback_days * bpd)
        ema_n = max(20, ema_days * bpd)

        c = bars['close']
        h = bars['high']
        l = bars['low']

        # ── マエダイ: DCブレイク + EMA200 ──
        dc_hi = h.shift(1).rolling(lb).max()
        dc_lo = l.shift(1).rolling(lb).min()
        ema   = c.ewm(span=ema_n, adjust=False).mean()

        raw_dc_long  = (c > dc_hi) & (c > ema)
        raw_dc_short = (c < dc_lo) & (c < ema)

        if confirm_bars >= 1:
            ml, ms = raw_dc_long.copy(), raw_dc_short.copy()
            for lag in range(1, confirm_bars + 1):
                ml = ml & raw_dc_long.shift(lag).fillna(False)
                ms = ms & raw_dc_short.shift(lag).fillna(False)
            maedai_long  = ml
            maedai_short = ms
        else:
            maedai_long  = raw_dc_long
            maedai_short = raw_dc_short

        # ── RSI押し目: EMA上 + RSI反発 ──
        delta    = c.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, 1e-10)))

        rsi_long  = (c > ema) & (rsi.shift(1) <= rsi_oversold)       & (rsi > rsi_oversold)
        rsi_short = (c < ema) & (rsi.shift(1) >= (100 - rsi_oversold)) & (rsi < (100 - rsi_oversold))

        # ── OR統合 (どちらか反応でエントリー) ──
        signals = pd.Series('flat', index=bars.index)
        signals[maedai_long  | rsi_long]  = 'long'
        signals[maedai_short | rsi_short] = 'short'

        return signals

    return _f


# ──────────────────────────────────────────────────────
# 時刻フィルター (指令1・指令5)
# ──────────────────────────────────────────────────────

def _apply_time_filters(signals, bars,
                        ny_session_only=False,
                        block_noon_jst=False,
                        block_saturday=False):
    """
    シグナルに時刻フィルターを適用するユーティリティ。

    ユーザー分析から:
    - NYセッション(JST 21-06時) は利益の53%, PF=3.05
    - JST 12時台はPF=0.94 (損失パターン)
    - 土曜日はPF=0.46 (明確な損失パターン)

    Args:
        ny_session_only: TrueでNYセッション(UTC 12-21時)のみエントリー
        block_noon_jst: TrueでJST 12時台(UTC 03:00-03:59)ブロック
        block_saturday: TrueでJST土曜日(UTC金曜15時以降+土曜)ブロック
    """
    idx = bars.index
    filtered = signals.copy()

    # UTC→JST変換 (JST = UTC+9)
    # UTC hour での判定: JST時間 = (UTC_hour + 9) % 24
    utc_hours = idx.hour

    if ny_session_only:
        # NYセッション: JST 21:00-翌06:00 = UTC 12:00-21:00
        # (サマータイム無視の近似値)
        ny_mask = (utc_hours >= 12) & (utc_hours < 21)
        filtered[~ny_mask] = 'flat'

    if block_noon_jst:
        # JST 12:00-12:59 = UTC 03:00-03:59
        noon_mask = (utc_hours == 3)
        filtered[noon_mask] = 'flat'

    if block_saturday:
        # JST 土曜日 = UTC 金曜15:00以降〜土曜14:59
        # 簡略: UTC weekday==4 (金曜) hour>=15 または weekday==5 (土曜)
        dow = idx.weekday
        sat_mask = ((dow == 4) & (utc_hours >= 15)) | (dow == 5)
        filtered[sat_mask] = 'flat'

    return filtered


def sig_rsi_pullback_filtered(freq='4h', ema_days=200, rsi_period=14,
                               rsi_oversold=45, rsi_overbought=55,
                               ny_session_only=True,
                               block_noon_jst=True,
                               block_saturday=True):
    """
    RSI押し目エントリー + ユーザー実績フィルター全適用版。

    ユーザーの「勝ちの設計図」に基づく全フィルター:
    1. NYセッション(UTC 12-21時)のみ (利益53%, PF=3.05)
    2. JST 12時台ブロック (PF=0.94の損失パターン)
    3. 土曜日ブロック (PF=0.46の明確な損失)
    4. EMA200 + RSI押し目 (押し目買い)
    """
    base_fn = sig_rsi_pullback_tf(freq=freq, ema_days=ema_days,
                                   rsi_period=rsi_period,
                                   rsi_oversold=rsi_oversold,
                                   rsi_overbought=rsi_overbought)

    def _f(bars):
        signals = base_fn(bars)
        return _apply_time_filters(signals, bars,
                                   ny_session_only=ny_session_only,
                                   block_noon_jst=block_noon_jst,
                                   block_saturday=block_saturday)
    return _f


def sig_rsi_short_tf(freq='4h', ema_days=200, rsi_period=14,
                      rsi_overbought=55,
                      ny_session_only=True,
                      block_noon_jst=True,
                      block_saturday=True):
    """
    RSI戻り売り戦略 (指令4: ショート専用)。

    ユーザー実績: ショートPF=12.49 (ロングPF=1.86を圧倒)

    エントリー条件:
    - close < EMA200 (下降トレンド)
    - RSIが rsi_overbought 以上から下抜け (戻りから売り)
    - NYセッション限定 + 禁止時間帯スキップ

    ロングシグナルは出さない (純粋ショート戦略)。
    エンジン側は long_biased=False で実行すること。
    """
    BARS_PER_DAY = {'1h': 24, '2h': 12, '4h': 6, '6h': 4,
                    '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd   = BARS_PER_DAY.get(freq, 1)
        ema_n = max(20, ema_days * bpd)
        c = bars['close']

        ema   = c.ewm(span=ema_n, adjust=False).mean()
        delta = c.diff()
        avg_gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        rsi_cross_down = (rsi.shift(1) >= rsi_overbought) & (rsi < rsi_overbought)
        short_mask = (c < ema) & rsi_cross_down

        signals = pd.Series('flat', index=bars.index)
        signals[short_mask] = 'short'

        return _apply_time_filters(signals, bars,
                                   ny_session_only=ny_session_only,
                                   block_noon_jst=block_noon_jst,
                                   block_saturday=block_saturday)
    return _f


def sig_dc_filtered(freq='4h', lookback_days=15, ema_days=200,
                    confirm_bars=2,
                    ny_session_only=False,
                    block_noon_jst=True,
                    block_saturday=True):
    """
    DC+EMA200ブレイクアウト + 禁止時間帯フィルター。

    既存のsig_maedai_dc_ema_tfにユーザーの禁止フィルターを追加。
    ny_session_only=Falseで「禁止時間帯だけ除外」のモードも可能。
    """
    base_fn = sig_maedai_dc_ema_tf(freq=freq, lookback_days=lookback_days,
                                    ema_days=ema_days, confirm_bars=confirm_bars)

    def _f(bars):
        signals = base_fn(bars)
        return _apply_time_filters(signals, bars,
                                   ny_session_only=ny_session_only,
                                   block_noon_jst=block_noon_jst,
                                   block_saturday=block_saturday)
    return _f


# ==============================================================
# 改善指令 v2.0 Mission1: 弱気ダイバージェンス + 主要レジスタンス ショート戦略
# ==============================================================

def sig_bearish_divergence_short(freq='8h',
                                  div_lookback=25,
                                  div_pivot_bars=3,
                                  res_lookback=100,
                                  res_atr_mult=1.5,
                                  rsi_period=14,
                                  ema_days=200,
                                  block_noon_jst=True,
                                  block_saturday=True):
    """
    弱気ダイバージェンス + 主要レジスタンス + MTFフィルター ショート戦略
    (改善指令 v2.0 Mission1)

    従来の単純RSI逆張りショートの問題点:
      - 金相場の長期上昇トレンドでは EMA200 下に価格が来ることが少ない
      - 単純な RSI 閾値超えはダマシが多く、長期バックテストで機能しない

    本戦略の3つの改善 (ヒント1〜3 対応):
    1. 弱気ダイバージェンス [ヒント1]:
       価格は高値を更新しているが RSI は切り下げている → 真の反転シグナル
       検出: 直近 div_lookback 本の中に2つのローカルハイを探し、
             h[new] >= h[old]*0.998 かつ rsi[new] < rsi[old] - 2.0 で確認

    2. 主要レジスタンス付近 [ヒント2]:
       過去 res_lookback 本の高値上位18%水準付近でのみショートエントリー
       → 「週足・日足の重要レジスタンスライン付近での反発」を模倣

    3. MTFフィルター [ヒント3]:
       同足 EMA200 の傾きが急上昇 (> ATR*0.12/bar) なら見送り
       → 「日足が強い上昇トレンドなら4時間足ショートは見送る」

    + 弱気ローソク足 or RSI高水準ダイバージェンスでエントリータイミング確認

    Args:
        freq: 時間軸 ('4h' | '8h' | '12h' 推奨)
        div_lookback: ダイバージェンス検索範囲 (バー数)
        div_pivot_bars: ローカルハイの前後確認本数
        res_lookback: 主要レジスタンス計算の参照期間 (バー数)
        res_atr_mult: レジスタンス付近判定のATR倍率
        rsi_period: RSI 算出期間
        ema_days: EMA 長期期間 (日数)
        block_noon_jst: JST 12時台ブロック
        block_saturday: 土曜ブロック
    """
    BARS_PER_DAY = {'1h': 24, '2h': 12, '4h': 6, '6h': 4,
                    '8h': 3, '12h': 2, '1d': 1}

    def _is_local_high(arr, i, half_win):
        """arr[i] が前後 half_win 本の中の最大値か（ゆるい判定）"""
        lo = max(0, i - half_win)
        hi = min(len(arr), i + half_win + 1)
        return arr[i] >= np.max(arr[lo:hi]) * 0.9995

    def _f(bars):
        n = len(bars)
        signals = pd.Series('flat', index=bars.index)
        min_start = max(div_lookback + div_pivot_bars * 2 + 5, res_lookback + 20, 30)
        if n < min_start + 10:
            return signals

        h = bars['high'].values
        l = bars['low'].values
        c = bars['close'].values

        # ATR (14期間)
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd.Series(tr, index=bars.index).rolling(14).mean().values

        # EMA200 (MTFフィルター)
        bpd = BARS_PER_DAY.get(freq, 1)
        ema_n = max(20, ema_days * bpd)
        close_s = bars['close']
        ema = close_s.ewm(span=ema_n, adjust=False).mean().values

        # RSI (Wilder平滑化)
        delta    = close_s.diff()
        avg_gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean().values
        avg_loss = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean().values
        rs  = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        # 弱気ローソク足の確認
        from .candle import detect_single_candle
        df_c = detect_single_candle(bars)
        candle_types = df_c['candle_type'].values
        BEAR_TYPES = frozenset(('big_bear', 'engulf_bear', 'inv_hammer',
                                'pinbar_bear', 'pullback_bear'))

        for i in range(min_start, n):
            a = atr[i]
            if np.isnan(a) or a == 0:
                continue

            # ── フィルター1: MTF — 「日足が強い上昇トレンドなら4時間足ショートは見送る」 ──
            # EMA200 が上向き (直近10本) かつ価格が EMA を明確に上回る場合はスキップ
            if i >= 10:
                ema_slope = (ema[i] - ema[i - 10]) / 10.0
                price_above_ema = c[i] > ema[i]
                # 強い上昇: 価格が EMA 上 + EMA が上向き + 傾きが急
                if price_above_ema and ema_slope > a * 0.08:
                    continue  # 強い上昇トレンド中はショート見送り

            # ── フィルター2: 主要レジスタンス付近か ──
            look_start = max(0, i - res_lookback)
            recent_highs = h[look_start:i]
            if len(recent_highs) < 15:
                continue
            # 上位15%の高値水準 = 主要レジスタンスゾーン
            resistance_lvl = np.percentile(recent_highs, 85)
            near_res = (c[i] >= resistance_lvl - a * res_atr_mult * 0.8) and \
                       (c[i] <= resistance_lvl + a * res_atr_mult)
            if not near_res:
                continue

            # ── フィルター3: RSI が過熱圏 (ショートの前提条件) ──
            if rsi[i] < 60:
                continue

            # ── 条件4: 弱気ダイバージェンス検出 ──
            if not _is_local_high(h, i, div_pivot_bars):
                continue

            div_found = False
            prev_j    = -1
            for j in range(i - div_pivot_bars * 2,
                           max(div_pivot_bars, i - div_lookback - 1), -1):
                if j < 0 or j == i:
                    continue
                if not _is_local_high(h, j, div_pivot_bars):
                    continue
                # 弱気ダイバージェンス (強化):
                #   価格: h[i] >= h[j]*0.998 (高値が同水準以上)
                #   RSI : rsi[i] < rsi[j] - 4.0 (RSI が明確に切り下げ)
                #   前回: rsi[j] > 55 (前回も過熱圏)
                if (h[i] >= h[j] * 0.998
                        and rsi[i] < rsi[j] - 4.0
                        and rsi[j] > 55):
                    div_found = True
                    prev_j    = j
                    break

            if not div_found:
                continue

            # ── 条件5: 弱気確認ローソク足 または 強いダイバージェンス ──
            bear_candle = candle_types[i] in BEAR_TYPES
            rsi_gap     = rsi[prev_j] - rsi[i] if prev_j >= 0 else 0
            strong_div  = (rsi[i] > 65) and (rsi_gap >= 8.0)
            if not (bear_candle or strong_div):
                continue

            signals.iloc[i] = 'short'

        return _apply_time_filters(signals, bars,
                                   ny_session_only=False,
                                   block_noon_jst=block_noon_jst,
                                   block_saturday=block_saturday)

    return _f


# ══════════════════════════════════════════════════════════════════════
# 積極的エントリー戦略群 ― 年間100回以上エントリー目標
# 哲学: エントリーしないことが最大の失敗。
#        負けトレードはデータ。エントリーしない機会損失は取り返せない。
# ══════════════════════════════════════════════════════════════════════

def sig_rsi_momentum(freq='4h', ema_days=21, rsi_period=14,
                     rsi_long_thresh=40, rsi_short_thresh=60,
                     ema_filter=True):
    """
    RSIモメンタム押し目エントリー (積極版)。
    年間100回以上エントリーを目指す。

    設計思想:
    - EMA200(旧)→EMA21(新): より短期のトレンド追従で機会を増やす
    - RSI40以下からの反転でロング (RSI45→RSI40→RSI45 でも取る)
    - 連続シグナルも許可 (持ち続けること自体をポジションの正当化に)
    - セッションフィルターなし (アジア時間の押し目も逃さない)

    Args:
        freq: '4h' | '8h' 推奨
        ema_days: トレンド確認EMA日数 (21日=短期, 50日=中期)
        rsi_long_thresh: この値以下から上抜けでロング (40)
        rsi_short_thresh: この値以上から下抜けでショート (60)
        ema_filter: Falseでトレンドフィルターなし (全方向対応)
    """
    BARS_PER_DAY = {'1h': 24, '4h': 6, '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd   = BARS_PER_DAY.get(freq, 6)
        ema_n = max(5, ema_days * bpd)

        c     = bars['close']
        ema   = c.ewm(span=ema_n, adjust=False).mean()

        delta    = c.diff()
        avg_gain = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
        rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, 1e-10)))

        rsi_cross_up   = (rsi.shift(1) <= rsi_long_thresh)  & (rsi > rsi_long_thresh)
        rsi_cross_down = (rsi.shift(1) >= rsi_short_thresh) & (rsi < rsi_short_thresh)

        if ema_filter:
            long_mask  = rsi_cross_up   & (c > ema)
            short_mask = rsi_cross_down & (c < ema)
        else:
            long_mask  = rsi_cross_up
            short_mask = rsi_cross_down

        signals = pd.Series('flat', index=bars.index)
        signals[long_mask]  = 'long'
        signals[short_mask] = 'short'
        return signals

    return _f


def sig_ema_bounce(freq='4h', ema_days=21, atr_touch_mult=0.5):
    """
    EMA21/50 実体タッチ → 翌足確認でロング。
    上昇トレンド中にEMAへの実際の押し目（lowがEMAを下回る）を捕捉。
    年間20〜50回エントリー想定。

    条件:
    1. EMAが上向き（5本前より上）
    2. 当バーのlowがEMAを下回る (実際の押し目タッチ)
    3. 当バーのcloseがEMAを上回る (即回復 = ピンバー的反発)
    4. 次バーも陽線 (close > open) で確認
    """
    BARS_PER_DAY = {'1h': 24, '4h': 6, '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd   = BARS_PER_DAY.get(freq, 6)
        ema_n = max(5, ema_days * bpd)

        c   = bars['close']
        h   = bars['high']
        l   = bars['low']
        o   = bars['open']
        ema = c.ewm(span=ema_n, adjust=False).mean()

        # ATR14
        tr  = pd.concat([h - l,
                         (h - c.shift(1)).abs(),
                         (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        ema_rising  = ema > ema.shift(5)
        ema_falling = ema < ema.shift(5)

        # ロング: lowがEMAを刺してcloseがEMA上で回復 (前バーの押し目)
        prev_touched_ema = (l.shift(1) <= ema.shift(1)) & (c.shift(1) >= ema.shift(1))
        bull_confirm     = c > o
        signals_long     = prev_touched_ema & bull_confirm & ema_rising & (c > ema)

        # ショート: highがEMAを刺してcloseがEMA下で回復
        prev_touched_ema_s = (h.shift(1) >= ema.shift(1)) & (c.shift(1) <= ema.shift(1))
        bear_confirm       = c < o
        signals_short      = prev_touched_ema_s & bear_confirm & ema_falling & (c < ema)

        signals = pd.Series('flat', index=bars.index)
        signals[signals_long]  = 'long'
        signals[signals_short] = 'short'
        return signals

    return _f


def sig_dc_fast(freq='4h', lookback_days=5, ema_filter=False,
                ema_days=50, confirm_bars=0):
    """
    超短期ドンチャンブレイク (5〜10日)。
    EMAフィルターなし・即エントリーで最大頻度を確保。
    年間60〜100回エントリー想定。

    Args:
        lookback_days: 5 or 7 推奨 (短いほど頻繁)
        ema_filter: Falseでトレンドフィルターなし
        confirm_bars: 0=即エントリー
    """
    BARS_PER_DAY = {'1h': 24, '4h': 6, '8h': 3, '12h': 2, '1d': 1}

    def _f(bars):
        bpd = BARS_PER_DAY.get(freq, 6)
        lb  = max(3, lookback_days * bpd)

        c = bars['close']
        h = bars['high']
        l = bars['low']

        dc_hi = h.shift(1).rolling(lb).max()
        dc_lo = l.shift(1).rolling(lb).min()

        raw_long  = c > dc_hi
        raw_short = c < dc_lo

        if ema_filter:
            ema_n = max(5, ema_days * bpd)
            ema   = c.ewm(span=ema_n, adjust=False).mean()
            raw_long  = raw_long  & (c > ema)
            raw_short = raw_short & (c < ema)

        signals = pd.Series('flat', index=bars.index)

        if confirm_bars >= 1:
            ml, ms = raw_long.copy(), raw_short.copy()
            for lag in range(1, confirm_bars + 1):
                ml = ml & raw_long.shift(lag).fillna(False)
                ms = ms & raw_short.shift(lag).fillna(False)
            signals[ml] = 'long'
            signals[ms] = 'short'
        else:
            signals[raw_long]  = 'long'
            signals[raw_short] = 'short'

        return signals

    return _f


def sig_aggressive_union(freq='4h', ema_days=21, lookback_days_dc=7,
                          rsi_thresh=42):
    """
    積極的複合シグナル (RSI押し目 OR 短期DCブレイク)。
    どちらか一方が点灯すればエントリー。
    年間120〜180回エントリー想定 (最も積極的なモード)。

    やがみメソッドの哲学:
    「エントリーしないことが最大の失敗」
    「失敗とは裁量でエントリーをやめること」
    → できる限りシグナルを拾いにいく。
    """
    rsi_fn = sig_rsi_momentum(freq=freq, ema_days=ema_days,
                               rsi_long_thresh=rsi_thresh, ema_filter=True)
    dc_fn  = sig_dc_fast(freq=freq, lookback_days=lookback_days_dc,
                          ema_filter=False, confirm_bars=0)

    def _f(bars):
        s_rsi = rsi_fn(bars)
        s_dc  = dc_fn(bars)

        # OR統合: どちらかが long/short ならそれを採用 (rsi優先)
        combined = s_rsi.copy()
        # DC シグナルがある && RSI がフラットな箇所を補完
        combined[(s_dc == 'long')  & (combined == 'flat')] = 'long'
        combined[(s_dc == 'short') & (combined == 'flat')] = 'short'
        return combined

    return _f
