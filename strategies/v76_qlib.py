"""
やがみMTF v76改qlib — 定量改善版
==================================
v76の二番底/二番天井ロジックをベースに、定量的フィルターで精度向上。

v76からの改善点:
  1. ADXフィルター: トレンド強度 > 20 でのみエントリー（レンジ相場回避）
  2. RSI確認: 二番底時にRSI < 40 (売られすぎ確認), 二番天井時にRSI > 60
  3. ボラティリティレジーム: ATR比率で低ボラ期を回避
  4. 確認足: 二番底/天井後に陽線/陰線確認を要求
  5. 適応的SL: 固定ATR倍率 → スウィングロー/ハイベースの動的SL
  6. RR改善: 1:3.0 (v76の1:2.5から引き上げ)
  7. EMA20+EMA50のゴールデン/デッドクロスでトレンド確認を強化
"""
import numpy as np
import pandas as pd


def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _atr(bars, period=14):
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr, index=bars.index).rolling(period).mean()


def _rsi(series, period=14):
    """RSI計算"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _adx(bars, period=14):
    """ADX計算"""
    h = bars['high'].values
    l = bars['low'].values
    c = bars['close'].values
    n = len(bars)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = h[i] - h[i-1]
        down = l[i-1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    tr[0] = h[0] - l[0]
    atr = pd.Series(tr, index=bars.index).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=bars.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=bars.index).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def resample_to_4h(bars_1h):
    """1H → 4Hリサンプル"""
    return bars_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])


def make_v76_qlib_signal(bars_4h):
    """
    v76改qlib シグナル生成クロージャ。

    改善: ADX, RSI, ボラレジーム, 確認足, 動的SL, EMAクロス

    Args:
        bars_4h: 4時間足データ

    Returns:
        signal_func(bars_1h) -> pd.Series of 'long'/'short'/None
    """
    # 4H: EMA20 + EMA50 トレンド判定
    ema20_4h = _ema(bars_4h['close'], 20)
    ema50_4h = _ema(bars_4h['close'], 50)

    # 4H: 複合トレンド (EMA20 > EMA50 かつ 終値 > EMA20 → 強い上昇)
    trend_4h = pd.Series(0, index=bars_4h.index, dtype=int)
    strong_up = (bars_4h['close'] > ema20_4h) & (ema20_4h > ema50_4h)
    strong_down = (bars_4h['close'] < ema20_4h) & (ema20_4h < ema50_4h)
    trend_4h[strong_up] = 1
    trend_4h[strong_down] = -1

    # 4H: ADX
    adx_4h, _, _ = _adx(bars_4h, 14)

    def signal_func(bars_1h):
        n = len(bars_1h)
        signals = pd.Series([None] * n, index=bars_1h.index)
        atr_1h = _atr(bars_1h, 14)
        rsi_1h = _rsi(bars_1h['close'], 14)

        # ボラティリティレジーム: ATR の SMA(50) 比率
        atr_sma = atr_1h.rolling(50).mean()
        vol_ratio = atr_1h / (atr_sma + 1e-10)

        # 1H ADX
        adx_1h, _, _ = _adx(bars_1h, 14)

        # スウィング検出
        NC = 3
        lows = bars_1h['low'].values
        highs = bars_1h['high'].values
        closes = bars_1h['close'].values
        opens = bars_1h['open'].values

        swing_lows = []
        swing_highs = []

        for i in range(NC, n - NC):
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

                hi = highs[check_idx]
                is_swing = True
                for j in range(1, NC + 1):
                    if highs[check_idx - j] > hi or highs[check_idx + j] > hi:
                        is_swing = False
                        break
                if is_swing:
                    swing_highs.append((check_idx, hi))

        for i in range(50, n):
            bar_time = bars_1h.index[i]
            atr_val = atr_1h.iloc[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            # === フィルター1: ボラティリティレジーム ===
            vr = vol_ratio.iloc[i]
            if np.isnan(vr) or vr < 0.6:
                continue  # 低ボラ期は見送り

            # === フィルター2: 1H ADX > 20 ===
            adx_val = adx_1h.iloc[i]
            if np.isnan(adx_val) or adx_val < 20:
                continue  # トレンドレスは見送り

            # 4Hトレンド取得
            mask = trend_4h.index <= bar_time
            if mask.sum() == 0:
                continue
            current_trend = trend_4h.loc[mask].iloc[-1]

            # 4H ADX取得
            adx4h_mask = adx_4h.index <= bar_time
            if adx4h_mask.sum() > 0:
                adx4h_val = adx_4h.loc[adx4h_mask].iloc[-1]
                if np.isnan(adx4h_val) or adx4h_val < 20:
                    continue  # 4Hもトレンドレスなら見送り

            # 直近スウィング取得
            recent_lows = [(idx, val) for idx, val in swing_lows if idx < i]
            recent_highs = [(idx, val) for idx, val in swing_highs if idx < i]

            # === 二番底判定 (ロング) ===
            if current_trend == 1 and len(recent_lows) >= 2:
                prev_low = recent_lows[-1][1]
                prev_prev_low = recent_lows[-2][1]
                threshold = atr_val * 0.5

                if abs(prev_prev_low - prev_low) <= threshold:
                    # フィルター3: RSI < 40 (売られすぎ反転)
                    rsi_val = rsi_1h.iloc[i]
                    if np.isnan(rsi_val) or rsi_val > 40:
                        continue

                    # フィルター4: 確認足 (陽線で反転を確認)
                    if closes[i] <= opens[i]:
                        continue  # 陰線なら見送り

                    # 価格がスウィングローより上にある
                    if closes[i] > prev_low:
                        signals.iloc[i] = 'long'
                        continue

            # === 二番天井判定 (ショート) ===
            if current_trend == -1 and len(recent_highs) >= 2:
                prev_high = recent_highs[-1][1]
                prev_prev_high = recent_highs[-2][1]
                threshold = atr_val * 0.5

                if abs(prev_prev_high - prev_high) <= threshold:
                    # フィルター3: RSI > 60 (買われすぎ反転)
                    rsi_val = rsi_1h.iloc[i]
                    if np.isnan(rsi_val) or rsi_val < 60:
                        continue

                    # フィルター4: 確認足 (陰線で反転を確認)
                    if closes[i] >= opens[i]:
                        continue  # 陽線なら見送り

                    if closes[i] < prev_high:
                        signals.iloc[i] = 'short'
                        continue

        return signals

    return signal_func


# ===== バックテスト用パラメータ =====
V76_QLIB_ENGINE_PARAMS = {
    'default_sl_atr': 0.25,      # 改善: ATR × 0.25 (v76の0.15から広げて狩られにくく)
    'default_tp_atr': 0.75,      # ATR × 0.25 × 3.0 (RR 1:3.0 に改善)
    'use_dynamic_sl': True,       # 改善: 動的SL (スウィングローベース)
    'sl_min_atr': 0.2,           # SL最低幅
    'dynamic_rr': 3.0,           # RR 1:3.0
    'pyramid_entries': 0,
    'trail_start_atr': 2.0,      # 改善: 2ATRで含み益トレーリング発動
    'trail_dist_atr': 1.0,       # 改善: 1ATR距離でトレール
    'exit_on_signal': False,
    'partial_tp_rr': 1.0,        # 1R到達で半利確
    'partial_tp_pct': 0.5,
    'breakeven_rr': 1.0,         # 1R到達でSLを建値に
    'min_sl_atr_mult': 0.1,
}

PAIR_CONFIGS = {
    'USDJPY': {'pip': 0.01, 'slippage_pips': 0.4},
    'EURJPY': {'pip': 0.01, 'slippage_pips': 1.1},
    'GBPJPY': {'pip': 0.01, 'slippage_pips': 1.5},
}
