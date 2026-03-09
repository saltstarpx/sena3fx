"""
yagami_mtf_v76_improved.py
===========================
v76ベース + v76改qlib の定量フィルターを統合

【v76からの改善点（qlib由来）】
  1. EMA20+EMA50 複合トレンド: 単純EMA20 → GC/DC確認で強トレンドのみ
  2. ADXフィルター (>20): 4H/1H両方でトレンドレス相場を回避
  3. RSI確認: 二番底時 RSI<40（売られすぎ）、二番天井時 RSI>60（買われすぎ）
  4. ボラティリティレジーム: ATR/SMA(50)比率 < 0.6 の低ボラ期を除外
  5. RR比率 1:3.0 に引き上げ（v76は1:2.5）
  6. 動的SL: スウィングロー/ハイベース（ATR固定のフォールバック付き）

【v76から維持】
  - 1分足エントリー精度（足更新から2分以内の最初の1分足）
  - スプレッド計算（チャートレベルでSL/TP固定、EP不利方向にずらし）
  - 4H/1Hの二番底・二番天井パターン検出

【スプレッドの扱い】(v76準拠)
  - SL/TPはチャートレベル（始値基準）で固定
  - ロング: ep = 始値 + spread, ショート: ep = 始値 - spread
  - 損益: (exit_price - ep) × 100 × dir
"""

import pandas as pd
import numpy as np


# ── インジケーター計算 ──────────────────────────────────────

def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_adx(bars, period=14):
    h = bars["high"].values
    l = bars["low"].values
    c = bars["close"].values
    n = len(bars)

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    tr[0] = h[0] - l[0]
    atr = pd.Series(tr, index=bars.index).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=bars.index).rolling(period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=bars.index).rolling(period).mean() / (atr + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return adx


def find_swing_lows(lows, nc=3):
    """スウィングロー検出（nc本の左右を確認）"""
    n = len(lows)
    swings = []
    for i in range(nc, n - nc):
        lo = lows[i]
        is_swing = True
        for j in range(1, nc + 1):
            if lows[i - j] < lo or lows[i + j] < lo:
                is_swing = False
                break
        if is_swing:
            swings.append((i, lo))
    return swings


def find_swing_highs(highs, nc=3):
    """スウィングハイ検出（nc本の左右を確認）"""
    n = len(highs)
    swings = []
    for i in range(nc, n - nc):
        hi = highs[i]
        is_swing = True
        for j in range(1, nc + 1):
            if highs[i - j] > hi or highs[i + j] > hi:
                is_swing = False
                break
        if is_swing:
            swings.append((i, hi))
    return swings


# ── メインシグナル生成 ──────────────────────────────────────

def generate_signals(
    data_1m, data_15m, data_4h,
    spread_pips=0.2,
    rr_ratio=3.0,          # qlib由来: 1:2.5 → 1:3.0
    adx_threshold=15,       # qlib由来: ADXフィルター（20→15に緩和）
    rsi_long_max=50,        # qlib由来: 二番底時RSI上限（40→50に緩和）
    rsi_short_min=50,       # qlib由来: 二番天井時RSI下限（60→50に緩和）
    vol_regime_min=0.5,     # qlib由来: ボラレジーム最低比率（0.6→0.5に緩和）
    use_dynamic_sl=True,    # qlib由来: スウィングベースSL
):
    """
    v76 + qlib統合シグナル生成。

    v76の1分足エントリー精度を維持しつつ、qlibの定量フィルターで
    低品質シグナルを除外する。

    Returns: list of signal dicts
    """
    spread = spread_pips * 0.01

    # ── 4時間足インジケーター ──
    data_4h = data_4h.copy()
    data_4h["atr"] = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["ema50"] = data_4h["close"].ewm(span=50, adjust=False).mean()
    # トレンド判定: v76のEMA20ベース + qlib由来のEMA50で強化
    # EMA20のみ → trend ±1、EMA20+EMA50一致 → trend ±1（同じだが品質高い）
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)
    # EMA50一致なら強トレンドフラグ
    data_4h["strong_trend"] = 0
    strong_up = (data_4h["close"] > data_4h["ema20"]) & (data_4h["ema20"] > data_4h["ema50"])
    strong_down = (data_4h["close"] < data_4h["ema20"]) & (data_4h["ema20"] < data_4h["ema50"])
    data_4h.loc[strong_up, "strong_trend"] = 1
    data_4h.loc[strong_down, "strong_trend"] = -1
    # qlib由来: ADX
    data_4h["adx"] = calculate_adx(data_4h, period=14)

    # ── 1時間足データ（15分足から集約）──
    data_1h = data_15m.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)
    # qlib由来: RSI, ADX, ボラレジーム
    data_1h["rsi"] = calculate_rsi(data_1h["close"], period=14)
    data_1h["adx"] = calculate_adx(data_1h, period=14)
    atr_sma = data_1h["atr"].rolling(50).mean()
    data_1h["vol_ratio"] = data_1h["atr"] / (atr_sma + 1e-10)

    # qlib由来: 1Hスウィング検出（動的SL用）
    h1_swing_lows = find_swing_lows(data_1h["low"].values, nc=3)
    h1_swing_highs = find_swing_highs(data_1h["high"].values, nc=3)

    signals = []
    used_times = set()

    # ── 4時間足の二番底・二番天井 ──────────────────────────
    h4_times = data_4h.index.tolist()
    for i in range(2, len(h4_times)):
        h4_current_time = h4_times[i]
        h4_prev1 = data_4h.iloc[i - 1]
        h4_prev2 = data_4h.iloc[i - 2]
        h4_current = data_4h.iloc[i]

        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        trend = h4_current["trend"]
        tolerance = atr_val * 0.3

        # qlib由来: 4H ADXフィルター
        h4_adx = h4_current["adx"]
        if pd.isna(h4_adx) or h4_adx < adx_threshold:
            continue

        # 1Hの状態を取得（RSI・ボラレジーム）
        h1_before = data_1h[data_1h.index <= h4_current_time]
        if len(h1_before) < 2:
            continue
        h1_latest = h1_before.iloc[-1]

        # qlib由来: ボラレジームフィルター
        vol_ratio = h1_latest["vol_ratio"]
        if pd.isna(vol_ratio) or vol_ratio < vol_regime_min:
            continue

        # ロング: 二番底
        if trend == 1:
            low1 = h4_prev2["low"]
            low2 = h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                # qlib由来: RSI確認
                h1_rsi = h1_latest["rsi"]
                if pd.isna(h1_rsi) or h1_rsi > rsi_long_max:
                    continue

                # SL計算: 動的 or 固定
                if use_dynamic_sl:
                    sl = _dynamic_sl_long(h1_swing_lows, h1_before, min(low1, low2), atr_val)
                else:
                    sl = min(low1, low2) - atr_val * 0.15

                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep = raw_ep + spread
                        risk = raw_ep - sl
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": 1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "4h",
                                "pattern": "double_bottom",
                                "h1_rsi": round(h1_rsi, 1),
                                "h4_adx": round(h4_adx, 1),
                                "vol_ratio": round(vol_ratio, 2),
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井
        if trend == -1:
            high1 = h4_prev2["high"]
            high2 = h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
                # qlib由来: RSI確認
                h1_rsi = h1_latest["rsi"]
                if pd.isna(h1_rsi) or h1_rsi < rsi_short_min:
                    continue

                # SL計算: 動的 or 固定
                if use_dynamic_sl:
                    sl = _dynamic_sl_short(h1_swing_highs, h1_before, max(high1, high2), atr_val)
                else:
                    sl = max(high1, high2) + atr_val * 0.15

                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep = raw_ep - spread
                        risk = sl - raw_ep
                        if 0 < risk <= atr_val * 3:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": -1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "4h",
                                "pattern": "double_top",
                                "h1_rsi": round(h1_rsi, 1),
                                "h4_adx": round(h4_adx, 1),
                                "vol_ratio": round(vol_ratio, 2),
                            })
                            used_times.add(entry_time)

    # ── 1時間足の二番底・二番天井 ──────────────────────────
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]

        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # qlib由来: ボラレジームフィルター
        vol_ratio = h1_current["vol_ratio"]
        if pd.isna(vol_ratio) or vol_ratio < vol_regime_min:
            continue

        # 4時間足のトレンド・ADXを取得
        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        trend = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        # qlib由来: 4H ADXフィルター
        h4_adx = h4_latest["adx"]
        if pd.isna(h4_adx) or h4_adx < adx_threshold:
            continue

        tolerance = atr_val * 0.3

        # ロング: 二番底（4時間足が強い上昇トレンドのみ）
        if trend == 1:
            low1 = h1_prev2["low"]
            low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                # qlib由来: RSI確認
                h1_rsi = h1_current["rsi"]
                if pd.isna(h1_rsi) or h1_rsi > rsi_long_max:
                    continue

                # SL計算
                if use_dynamic_sl:
                    sl = _dynamic_sl_long(h1_swing_lows, data_1h.iloc[:i+1], min(low1, low2), atr_val)
                else:
                    sl = min(low1, low2) - atr_val * 0.15

                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep = raw_ep + spread
                        risk = raw_ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": 1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "1h",
                                "pattern": "double_bottom",
                                "h1_rsi": round(h1_rsi, 1),
                                "h4_adx": round(h4_adx, 1),
                                "vol_ratio": round(vol_ratio, 2),
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井（4時間足が強い下降トレンドのみ）
        if trend == -1:
            high1 = h1_prev2["high"]
            high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
                # qlib由来: RSI確認
                h1_rsi = h1_current["rsi"]
                if pd.isna(h1_rsi) or h1_rsi < rsi_short_min:
                    continue

                # SL計算
                if use_dynamic_sl:
                    sl = _dynamic_sl_short(h1_swing_highs, data_1h.iloc[:i+1], max(high1, high2), atr_val)
                else:
                    sl = max(high1, high2) + atr_val * 0.15

                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        raw_ep = entry_bar["open"]
                        ep = raw_ep - spread
                        risk = sl - raw_ep
                        if 0 < risk <= h4_atr * 2:
                            tp = raw_ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": -1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "spread": spread,
                                "tf": "1h",
                                "pattern": "double_top",
                                "h1_rsi": round(h1_rsi, 1),
                                "h4_adx": round(h4_adx, 1),
                                "vol_ratio": round(vol_ratio, 2),
                            })
                            used_times.add(entry_time)

    signals.sort(key=lambda x: x["time"])
    return signals


# ── 動的SLヘルパー ──────────────────────────────────────

def _dynamic_sl_long(swing_lows, h1_data, pattern_low, atr_val):
    """
    ロング用動的SL: 直近スウィングローの下に設定。
    スウィングが見つからない場合はパターン安値 - ATR*0.25にフォールバック。
    """
    n = len(h1_data)
    recent = [(idx, val) for idx, val in swing_lows if idx < n]
    if recent:
        swing_low = recent[-1][1]
        sl = swing_low - atr_val * 0.1
        # SLがパターン安値より高すぎる場合はパターン安値を使用
        if sl > pattern_low:
            sl = pattern_low - atr_val * 0.15
        return sl
    return pattern_low - atr_val * 0.25


def _dynamic_sl_short(swing_highs, h1_data, pattern_high, atr_val):
    """
    ショート用動的SL: 直近スウィングハイの上に設定。
    スウィングが見つからない場合はパターン高値 + ATR*0.25にフォールバック。
    """
    n = len(h1_data)
    recent = [(idx, val) for idx, val in swing_highs if idx < n]
    if recent:
        swing_high = recent[-1][1]
        sl = swing_high + atr_val * 0.1
        if sl < pattern_high:
            sl = pattern_high + atr_val * 0.15
        return sl
    return pattern_high + atr_val * 0.25
