"""
yagami_mtf_v75.py
================
v74ベース + エントリーを1分足始値に変更

【変更点】
- v74: 4h/1h更新後の最初の15分足の始値でエントリー
- v75: 4h/1h更新後の最初の1分足の始値でエントリー（やがみnote準拠）

【エントリーウィンドウ】
- 4時間足更新: 更新から2分以内の最初の1分足（やがみ「更新から2分くらいのイメージ」）
- 1時間足更新: 更新から2分以内の最初の1分足
"""

import pandas as pd
import numpy as np


def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_signals(data_1m, data_15m, data_4h, spread_pips=0.2, rr_ratio=2.5):
    """
    4時間足 + 1時間足の二番底・二番天井でシグナルを生成する。
    エントリーは更新後の最初の1分足始値（成行）。

    Returns: list of signal dicts
    """
    spread = spread_pips * 0.01

    # 4時間足: ATR・EMA・トレンド
    data_4h = data_4h.copy()
    data_4h["atr"] = calculate_atr(data_4h, period=14)
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)

    # 1時間足データを15分足から集約
    data_1h = data_15m.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    data_1h["atr"] = calculate_atr(data_1h, period=14)

    signals = []
    used_times = set()  # 重複エントリー防止

    # ── 4時間足の二番底・二番天井 ──────────────────────────────
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

        # ロング: 二番底
        if trend == 1:
            low1 = h4_prev2["low"]
            low2 = h4_prev1["low"]
            if abs(low1 - low2) <= tolerance and h4_prev1["close"] > h4_prev1["open"]:
                sl = min(low1, low2) - atr_val * 0.15
                # 4時間足更新から2分以内の1分足
                entry_window_end = h4_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h4_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        ep = entry_bar["open"] + spread
                        risk = ep - sl
                        if 0 < risk <= atr_val * 3:
                            tp = ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": 1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "tf": "4h",
                                "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井
        if trend == -1:
            high1 = h4_prev2["high"]
            high2 = h4_prev1["high"]
            if abs(high1 - high2) <= tolerance and h4_prev1["close"] < h4_prev1["open"]:
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
                        ep = entry_bar["open"] - spread
                        risk = sl - ep
                        if 0 < risk <= atr_val * 3:
                            tp = ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": -1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "tf": "4h",
                                "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    # ── 1時間足の二番底・二番天井 ──────────────────────────────
    h1_times = data_1h.index.tolist()
    for i in range(2, len(h1_times)):
        h1_current_time = h1_times[i]
        h1_prev1 = data_1h.iloc[i - 1]
        h1_prev2 = data_1h.iloc[i - 2]
        h1_current = data_1h.iloc[i]

        atr_val = h1_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        # 4時間足のトレンドを取得
        h4_before = data_4h[data_4h.index <= h1_current_time]
        if len(h4_before) == 0:
            continue
        h4_latest = h4_before.iloc[-1]
        trend = h4_latest["trend"]
        h4_atr = h4_latest["atr"]

        tolerance = atr_val * 0.3

        # ロング: 二番底（4時間足が上昇トレンドのみ）
        if trend == 1:
            low1 = h1_prev2["low"]
            low2 = h1_prev1["low"]
            if abs(low1 - low2) <= tolerance and h1_prev1["close"] > h1_prev1["open"]:
                sl = min(low1, low2) - atr_val * 0.15
                # 1時間足更新から2分以内の1分足
                entry_window_end = h1_current_time + pd.Timedelta(minutes=2)
                m1_window = data_1m[
                    (data_1m.index >= h1_current_time) &
                    (data_1m.index < entry_window_end)
                ]
                if len(m1_window) > 0:
                    entry_bar = m1_window.iloc[0]
                    entry_time = entry_bar.name
                    if entry_time not in used_times:
                        ep = entry_bar["open"] + spread
                        risk = ep - sl
                        if 0 < risk <= h4_atr * 2:
                            tp = ep + risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": 1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "tf": "1h",
                                "pattern": "double_bottom"
                            })
                            used_times.add(entry_time)

        # ショート: 二番天井（4時間足が下降トレンドのみ）
        if trend == -1:
            high1 = h1_prev2["high"]
            high2 = h1_prev1["high"]
            if abs(high1 - high2) <= tolerance and h1_prev1["close"] < h1_prev1["open"]:
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
                        ep = entry_bar["open"] - spread
                        risk = sl - ep
                        if 0 < risk <= h4_atr * 2:
                            tp = ep - risk * rr_ratio
                            signals.append({
                                "time": entry_time,
                                "dir": -1,
                                "ep": ep,
                                "sl": sl,
                                "tp": tp,
                                "risk": risk,
                                "tf": "1h",
                                "pattern": "double_top"
                            })
                            used_times.add(entry_time)

    # 時刻順にソート
    signals.sort(key=lambda x: x["time"])
    return signals
