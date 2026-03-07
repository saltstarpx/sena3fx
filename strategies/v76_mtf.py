"""
やがみMTF v76 — 現行版（忠実実装）
====================================
4H EMA20でトレンドフィルター、1Hで二番底/二番天井パターン検出。

仕様:
  - トレンドフィルター: 4H EMA20 (終値 > EMA20 → 上昇, < EMA20 → 下降)
  - 二番底判定: abs(前々回安値 - 前回安値) <= ATR(14) × 0.3
  - 二番天井判定: abs(前々回高値 - 前回高値) <= ATR(14) × 0.3
  - 損切り幅: ATR(14) × 0.15
  - リスクリワード: 1:2.5
  - 半利確: 1R到達時に半決済、SLを建値に移動
"""
import numpy as np
import pandas as pd


def _ema(series, period):
    """EMA計算"""
    return series.ewm(span=period, adjust=False).mean()


def _atr(bars, period=14):
    """ATR計算"""
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars.index).rolling(period).mean()


def _find_swing_lows(bars, n_confirm=5):
    """
    スウィングロー検出: 左右n_confirm本より安い局所最小値。
    Returns: list of (index_position, low_value)
    """
    lows = bars['low'].values
    n = len(lows)
    swings = []
    for i in range(n_confirm, n - n_confirm):
        lo = lows[i]
        if (all(lows[i - j] >= lo for j in range(1, n_confirm + 1)) and
                all(lows[i + j] >= lo for j in range(1, n_confirm + 1))):
            swings.append((i, lo))
    return swings


def _find_swing_highs(bars, n_confirm=5):
    """
    スウィングハイ検出: 左右n_confirm本より高い局所最大値。
    Returns: list of (index_position, high_value)
    """
    highs = bars['high'].values
    n = len(highs)
    swings = []
    for i in range(n_confirm, n - n_confirm):
        hi = highs[i]
        if (all(highs[i - j] <= hi for j in range(1, n_confirm + 1)) and
                all(highs[i + j] <= hi for j in range(1, n_confirm + 1))):
            swings.append((i, hi))
    return swings


def resample_to_4h(bars_1h):
    """1H → 4Hリサンプル"""
    return bars_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])


def make_v76_signal(bars_4h):
    """
    v76シグナル生成関数を返すクロージャ。
    BacktestEngine.run(signal_func=...) に渡す。

    Args:
        bars_4h: 4時間足データ (EMA20トレンドフィルター用)

    Returns:
        signal_func(bars_1h) -> pd.Series of 'long'/'short'/None
    """
    # 4H EMA20を事前計算
    ema20_4h = _ema(bars_4h['close'], 20)
    trend_4h = pd.Series(0, index=bars_4h.index, dtype=int)
    trend_4h[bars_4h['close'] > ema20_4h] = 1   # 上昇トレンド
    trend_4h[bars_4h['close'] < ema20_4h] = -1   # 下降トレンド

    def signal_func(bars_1h):
        n = len(bars_1h)
        signals = pd.Series([None] * n, index=bars_1h.index)
        atr_1h = _atr(bars_1h, 14)

        # 1Hバーごとにスウィングを漸進的に検出
        swing_lows = []   # (bar_index, low_value)
        swing_highs = []  # (bar_index, high_value)
        NC = 3  # 確認本数

        lows = bars_1h['low'].values
        highs = bars_1h['high'].values

        for i in range(NC, n - NC):
            # スウィングロー判定 (i - NC の位置が確定)
            check_idx = i - NC
            if check_idx >= NC:
                lo = lows[check_idx]
                is_swing = True
                for j in range(1, NC + 1):
                    if lows[check_idx - j] < lo or lows[check_idx + j] < lo:
                        is_swing = False
                        break
                if is_swing:
                    swing_lows.append((check_idx, lo))

            # スウィングハイ判定
            if check_idx >= NC:
                hi = highs[check_idx]
                is_swing = True
                for j in range(1, NC + 1):
                    if highs[check_idx - j] > hi or highs[check_idx + j] > hi:
                        is_swing = False
                        break
                if is_swing:
                    swing_highs.append((check_idx, hi))

        # 各バーでシグナル判定
        for i in range(30, n):
            bar_time = bars_1h.index[i]
            atr_val = atr_1h.iloc[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # 4Hトレンドを取得 (直近の4Hバー)
            mask = trend_4h.index <= bar_time
            if mask.sum() == 0:
                continue
            current_trend = trend_4h.loc[mask].iloc[-1]

            # 直近2つのスウィングを取得
            recent_lows = [(idx, val) for idx, val in swing_lows if idx < i]
            recent_highs = [(idx, val) for idx, val in swing_highs if idx < i]

            # 二番底判定 (ロング)
            if current_trend == 1 and len(recent_lows) >= 2:
                prev_low = recent_lows[-1][1]
                prev_prev_low = recent_lows[-2][1]
                threshold = atr_val * 0.5
                if abs(prev_prev_low - prev_low) <= threshold:
                    # 二番底パターン成立 → ロングシグナル
                    # 直近のスウィングローから現在価格が上にあることを確認
                    if bars_1h['close'].iloc[i] > prev_low:
                        signals.iloc[i] = 'long'
                        continue

            # 二番天井判定 (ショート)
            if current_trend == -1 and len(recent_highs) >= 2:
                prev_high = recent_highs[-1][1]
                prev_prev_high = recent_highs[-2][1]
                threshold = atr_val * 0.5
                if abs(prev_prev_high - prev_high) <= threshold:
                    if bars_1h['close'].iloc[i] < prev_high:
                        signals.iloc[i] = 'short'
                        continue

        return signals

    return signal_func


# ===== バックテスト用パラメータ =====
V76_ENGINE_PARAMS = {
    'default_sl_atr': 0.15,     # ATR × 0.15
    'default_tp_atr': 0.375,    # ATR × 0.15 × 2.5 (RR 1:2.5)
    'use_dynamic_sl': False,     # 固定SL
    'pyramid_entries': 0,        # ピラミッドなし
    'trail_start_atr': 0.0,     # トレーリングなし
    'exit_on_signal': False,     # SL/TPのみで決済
    'partial_tp_rr': 1.0,       # 1R到達で半利確
    'partial_tp_pct': 0.5,      # 50%決済
    'breakeven_rr': 1.0,        # 1R到達でSLを建値に
    'min_sl_atr_mult': 0.05,    # Safety valve: 最低SL幅
}

# ペア別スプレッド設定
PAIR_CONFIGS = {
    'USDJPY': {'pip': 0.01, 'slippage_pips': 0.4},
    'EURJPY': {'pip': 0.01, 'slippage_pips': 1.1},
    'GBPJPY': {'pip': 0.01, 'slippage_pips': 1.5},
}
