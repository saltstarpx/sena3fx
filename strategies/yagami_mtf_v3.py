"""
やがみ式3層MTF戦略 v3.0
========================
RUN-010 (Manus) の検証結果に基づく3層ロジック:
  ① H4: 環境認識と「場所」の特定 (壁の検出, トレンド方向)
  ② H1: パターン認識と「タイミング」の探索 (実体揃い, IB蓄積)
  ③ M15: 精密な執行と「背」の確定 (ボラ収束→放れ, タイトSL)

※ RUN-010はUSDJPY 1分足で検証。本実装ではXAUUSD M15/H1/H4で適用。
   1分足データが未入手のため、M15を最小執行足とする。
"""
import numpy as np
import pandas as pd
from lib.candle import (detect_single_candle, detect_price_action,
                        body, candle_range, upper_wick, lower_wick)


# ==============================================================
# ① H4: 環境認識 — トレンドと「壁」の特定
# ==============================================================

def h4_environment(bars_4h, lookback=20):
    """
    H4環境認識。

    Returns:
        DataFrame with columns:
        - h4_trend: 1 (up), -1 (down), 0 (neutral)
        - h4_wall_high: 直近20本の最高値
        - h4_wall_low: 直近20本の最安値
        - h4_atr: ATR(14)
    """
    h = bars_4h['high'].values
    l = bars_4h['low'].values
    c = bars_4h['close'].values
    o = bars_4h['open'].values
    n = len(bars_4h)

    # ATR
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr, index=bars_4h.index).rolling(14).mean()

    # Wall High / Wall Low (直近lookback本)
    wall_high = pd.Series(h, index=bars_4h.index).rolling(lookback).max()
    wall_low = pd.Series(l, index=bars_4h.index).rolling(lookback).min()

    # トレンド判定: 直近lookback本の高値・安値の切り上げ/切り下げ
    trend = pd.Series(0, index=bars_4h.index, dtype=int)
    for i in range(lookback, n):
        highs_recent = h[i - lookback:i + 1]
        lows_recent = l[i - lookback:i + 1]

        # 高値と安値の傾き (線形回帰の符号)
        x = np.arange(lookback + 1)
        h_slope = np.polyfit(x, highs_recent, 1)[0]
        l_slope = np.polyfit(x, lows_recent, 1)[0]

        if h_slope > 0 and l_slope > 0:
            trend.iloc[i] = 1   # uptrend
        elif h_slope < 0 and l_slope < 0:
            trend.iloc[i] = -1  # downtrend
        else:
            trend.iloc[i] = 0   # neutral

    return pd.DataFrame({
        'h4_trend': trend,
        'h4_wall_high': wall_high,
        'h4_wall_low': wall_low,
        'h4_atr': atr,
    }, index=bars_4h.index)


# ==============================================================
# ② H1: パターン認識 — 実体揃い・IB蓄積・エントリーゾーン
# ==============================================================

def h1_pattern(bars_h1, h4_env_aligned):
    """
    H1パターン認識。

    h4_env_aligned: H4環境をH1インデックスにffillしたDataFrame

    Returns:
        DataFrame with columns:
        - h1_body_align: True if 直近3本の実体stddev < ATR*0.20
        - h1_inside_bar: True if current bar is inside previous
        - h1_in_entry_zone: True if price is within 2*H4_ATR of wall
        - h1_pattern_ready: True if body_align or IB蓄積 (within entry zone)
    """
    o = bars_h1['open'].values
    h = bars_h1['high'].values
    l = bars_h1['low'].values
    c = bars_h1['close'].values
    n = len(bars_h1)

    # H1 ATR
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    h1_atr = pd.Series(tr, index=bars_h1.index).rolling(14).mean()

    body_align = pd.Series(False, index=bars_h1.index)
    inside_bar = pd.Series(False, index=bars_h1.index)
    in_entry_zone = pd.Series(False, index=bars_h1.index)
    pattern_ready = pd.Series(False, index=bars_h1.index)

    for i in range(3, n):
        a = h1_atr.iloc[i]
        if np.isnan(a) or a == 0:
            continue

        # --- 実体の揃い (Body Alignment) ---
        # 直近3本の終値の標準偏差 < ATR * 0.20
        closes_3 = c[i - 2:i + 1]
        if np.std(closes_3) < a * 0.20:
            body_align.iloc[i] = True

        # --- インサイドバー ---
        if h[i] <= h[i - 1] and l[i] >= l[i - 1]:
            inside_bar.iloc[i] = True

        # --- エントリーゾーン: H4壁から 2.0 * H4_ATR 以内 ---
        h4_wall_hi = h4_env_aligned['h4_wall_high'].iloc[i]
        h4_wall_lo = h4_env_aligned['h4_wall_low'].iloc[i]
        h4_atr_val = h4_env_aligned['h4_atr'].iloc[i]

        if not np.isnan(h4_wall_hi) and not np.isnan(h4_atr_val) and h4_atr_val > 0:
            near_wall_hi = abs(c[i] - h4_wall_hi) < 2.0 * h4_atr_val
            near_wall_lo = abs(c[i] - h4_wall_lo) < 2.0 * h4_atr_val
            if near_wall_hi or near_wall_lo:
                in_entry_zone.iloc[i] = True

        # --- パターン準備完了 ---
        if in_entry_zone.iloc[i] and (body_align.iloc[i] or inside_bar.iloc[i]):
            pattern_ready.iloc[i] = True

    return pd.DataFrame({
        'h1_body_align': body_align,
        'h1_inside_bar': inside_bar,
        'h1_in_entry_zone': in_entry_zone,
        'h1_pattern_ready': pattern_ready,
        'h1_atr': h1_atr,
    }, index=bars_h1.index)


# ==============================================================
# ③ M15: 精密な執行 — ボラ収束→放れ, タイトSL
# ==============================================================

def m15_execution_signals(bars_m15, h4_env_aligned, h1_pat_aligned):
    """
    M15執行シグナル。

    M15のボラティリティ収束(横軸形成) → 放れの確認でエントリー。
    SLは執行足の安値/高値 + 小バッファ。

    Returns:
        pd.Series of 'long', 'short', or None
    """
    o = bars_m15['open'].values
    h = bars_m15['high'].values
    l = bars_m15['low'].values
    c = bars_m15['close'].values
    n = len(bars_m15)

    # M15 ATR
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    m15_atr = pd.Series(tr, index=bars_m15.index).rolling(14).mean()

    signals = pd.Series(index=bars_m15.index, dtype=object)

    for i in range(5, n):
        a = m15_atr.iloc[i]
        if np.isnan(a) or a == 0:
            continue

        # --- 上位足フィルター ---
        h4_trend = h4_env_aligned['h4_trend'].iloc[i]
        h4_wall_hi = h4_env_aligned['h4_wall_high'].iloc[i]
        h4_wall_lo = h4_env_aligned['h4_wall_low'].iloc[i]
        h4_atr_val = h4_env_aligned['h4_atr'].iloc[i]
        h1_ready = h1_pat_aligned['h1_pattern_ready'].iloc[i]

        # H4 neutralの場合はスキップ
        if h4_trend == 0:
            continue

        # H1パターン未成立ならスキップ
        if not h1_ready:
            continue

        # --- M15ボラティリティ収束 (厳格版) ---
        ranges_5 = [h[j] - l[j] for j in range(i - 4, i + 1)]
        avg_range_5 = np.mean(ranges_5)
        current_range = h[i] - l[i]

        # 収束条件: 前足が平均の0.5倍以下 → 今足が「放れ」
        # (今足自体が収束している場合はまだエントリーしない)
        if i < 2:
            continue
        prev_range = h[i - 1] - l[i - 1]
        prev_contracted = prev_range <= avg_range_5 * 0.5
        if not prev_contracted:
            continue
        # 今足は収束から「放れた」足であること
        if current_range <= avg_range_5 * 0.5:
            continue  # まだ収束中、待つ

        # --- 放れの確認 ---
        is_bull_candle = c[i] > o[i]
        is_bear_candle = c[i] < o[i]

        # --- RRチェック ---
        if np.isnan(h4_wall_hi) or np.isnan(h4_wall_lo) or np.isnan(h4_atr_val):
            continue

        if h4_trend == 1 and is_bull_candle:
            # Long: SL = 今足の安値, TP = H4壁High
            sl_dist = c[i] - l[i] + a * 0.1  # 小バッファ
            tp_dist = h4_wall_hi - c[i]
            if sl_dist > 0 and tp_dist / sl_dist >= 1.5:
                signals.iloc[i] = 'long'

        elif h4_trend == -1 and is_bear_candle:
            # Short: SL = 今足の高値, TP = H4壁Low
            sl_dist = h[i] - c[i] + a * 0.1
            tp_dist = c[i] - h4_wall_lo
            if sl_dist > 0 and tp_dist / sl_dist >= 1.5:
                signals.iloc[i] = 'short'

    return signals


# ==============================================================
# 統合シグナル関数 (BacktestEngine互換)
# ==============================================================

def sig_yagami_mtf_v3(bars_4h, bars_h1=None, rr_min=1.5,
                       h4_lookback=20, align_tol=0.20, vol_contraction=0.6):
    """
    やがみ式3層MTF戦略 v3.0。

    BacktestEngine互換: engine.run(bars_m15, sig_func) で呼び出し。

    Args:
        bars_4h: 4H OHLCデータ
        bars_h1: 1H OHLCデータ (Noneなら bars_m15 から1Hリサンプル)
        rr_min: 最低RR比 (default 1.5)
        h4_lookback: H4壁検出lookback (default 20)
        align_tol: H1実体揃い許容値 ATR倍 (default 0.20)
        vol_contraction: M15ボラ収束閾値 (default 0.6)
    """
    # H4環境を事前計算
    h4_env = h4_environment(bars_4h, lookback=h4_lookback)

    def _f(bars_m15):
        n = len(bars_m15)
        signals = pd.Series(index=bars_m15.index, dtype=object)
        if n < 30:
            return signals

        # H1データ: 提供されなければM15から1Hリサンプル
        if bars_h1 is not None:
            h1_data = bars_h1
        else:
            h1_data = bars_m15.resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()

        # H4環境をM15にffill
        h4_aligned = h4_env.reindex(bars_m15.index, method='ffill')

        # H1パターンをM15にffill
        h4_for_h1 = h4_env.reindex(h1_data.index, method='ffill')
        h1_pat = h1_pattern(h1_data, h4_for_h1)
        h1_aligned = h1_pat.reindex(bars_m15.index, method='ffill')

        # M15執行シグナル
        return m15_execution_signals(bars_m15, h4_aligned, h1_aligned)

    return _f


def sig_yagami_mtf_v3_h1(bars_4h, rr_min=1.5, h4_lookback=20):
    """
    3層MTF v3.0 — H1執行版。
    M15データがない場合に H4(環境) → H1(パターン+執行) の2層で近似。

    v3.0の核心ロジック:
    - ボラ収束(横軸形成) + 実体揃い(壁) + エントリーゾーン(H4壁近傍)
    - タイトSL: エンジン側で default_sl_atr=0.5 を推奨
    - 高RR: H4壁を目標 (TP)

    BacktestEngine互換: engine.run(bars_h1, sig_func) で呼び出し。
    """
    h4_env = h4_environment(bars_4h, lookback=h4_lookback)

    def _f(bars_h1):
        n = len(bars_h1)
        signals = pd.Series(index=bars_h1.index, dtype=object)
        if n < 30:
            return signals

        o = bars_h1['open'].values
        h = bars_h1['high'].values
        l = bars_h1['low'].values
        c = bars_h1['close'].values

        # H1 ATR
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        h1_atr = pd.Series(tr, index=bars_h1.index).rolling(14).mean()

        # H4環境をH1にffill
        h4_aligned = h4_env.reindex(bars_h1.index, method='ffill')

        for i in range(20, n):
            a = h1_atr.iloc[i]
            if np.isnan(a) or a == 0:
                continue

            h4_trend = h4_aligned['h4_trend'].iloc[i]
            h4_wall_hi = h4_aligned['h4_wall_high'].iloc[i]
            h4_wall_lo = h4_aligned['h4_wall_low'].iloc[i]
            h4_atr_val = h4_aligned['h4_atr'].iloc[i]

            if h4_trend == 0:
                continue
            if np.isnan(h4_wall_hi) or np.isnan(h4_atr_val) or h4_atr_val == 0:
                continue

            # エントリーゾーン (壁から2*H4_ATR以内)
            near_hi = abs(c[i] - h4_wall_hi) < 2.0 * h4_atr_val
            near_lo = abs(c[i] - h4_wall_lo) < 2.0 * h4_atr_val
            if not (near_hi or near_lo):
                continue

            # 実体揃い (直近3本) — 厳格版: ATR*0.15
            if i >= 3:
                closes_3 = c[i - 2:i + 1]
                body_aligned = np.std(closes_3) < a * 0.15
            else:
                body_aligned = False

            # インサイドバー (連続2本以上)
            ib_count = 0
            for k in range(i, max(i - 5, 0), -1):
                if k > 0 and h[k] <= h[k - 1] and l[k] >= l[k - 1]:
                    ib_count += 1
                else:
                    break
            ib_accumulated = ib_count >= 2

            # 少なくとも1つのパターン条件が必要
            if not (body_aligned or ib_accumulated):
                continue

            # ボラ収束 (直近5本平均の0.5倍以下 — 厳格化)
            if i >= 5:
                ranges_5 = [h[j] - l[j] for j in range(i - 4, i + 1)]
                avg_range = np.mean(ranges_5)
                curr_range = h[i] - l[i]
                # 今足自体が収束しているか、前足が収束→今足放れ
                vol_now = curr_range <= avg_range * 0.5
                vol_prev_break = False
                if not vol_now and i >= 2:
                    prev_range = h[i - 1] - l[i - 1]
                    if prev_range <= avg_range * 0.5:
                        vol_prev_break = True
                if not (vol_now or vol_prev_break):
                    continue

            # 方向 + RRチェック (執行足のヒゲ先=背)
            is_bull = c[i] > o[i]
            is_bear = c[i] < o[i]

            if h4_trend == 1 and is_bull:
                # タイトSL: 今足の安値を背にする
                sl_dist = c[i] - l[i] + a * 0.05  # 極小バッファ
                tp_dist = h4_wall_hi - c[i]
                if sl_dist > 0 and tp_dist / sl_dist >= rr_min:
                    signals.iloc[i] = 'long'

            elif h4_trend == -1 and is_bear:
                sl_dist = h[i] - c[i] + a * 0.05
                tp_dist = c[i] - h4_wall_lo
                if sl_dist > 0 and tp_dist / sl_dist >= rr_min:
                    signals.iloc[i] = 'short'

        return signals

    return _f
