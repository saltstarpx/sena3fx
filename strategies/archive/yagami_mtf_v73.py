"""
yagami_mtf_v73.py
================
やがみnote準拠: 4時間足更新タイミング + 二番底・二番天井ロジック

【コンセプト】
- 4時間足が確定した瞬間の「更新初動（2分以内）」でエントリー
- 二番底・二番天井の形成を15分足で確認してからエントリー
- 損切りは「最後の押し安値・戻り高値」（チャート構造上の意味ある場所）
- RR 2.5 固定

【エントリー条件】
1. 4時間足EMA20でトレンド方向を確認
2. 直前の4時間足が「二番底（ロング）」または「二番天井（ショート）」を形成しているか確認
   - 二番底: 前々回の4h安値 ≈ 前回の4h安値（±ATR×0.3以内）かつ前回が陽線
   - 二番天井: 前々回の4h高値 ≈ 前回の4h高値（±ATR×0.3以内）かつ前回が陰線
3. 4時間足更新直後の15分足2本以内（＝更新から30分以内）で成行エントリー
4. 損切り: 二番底の安値 / 二番天井の高値の少し外側
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
    4時間足更新タイミング + 二番底・二番天井ロジックでシグナルを生成する。
    
    Returns: list of signal dicts
    """
    spread = spread_pips * 0.01
    
    # 4時間足のATRとEMA計算
    h4_atr = calculate_atr(data_4h, period=14)
    data_4h = data_4h.copy()
    data_4h["ema20"] = data_4h["close"].ewm(span=20, adjust=False).mean()
    data_4h["trend"] = np.where(data_4h["close"] > data_4h["ema20"], 1, -1)
    data_4h["atr"] = h4_atr
    
    signals = []
    h4_times = data_4h.index.tolist()
    
    # 4時間足を3本以上確認できる位置から開始
    for i in range(2, len(h4_times)):
        h4_current_time = h4_times[i]  # 現在確定した4時間足
        h4_prev1 = data_4h.iloc[i - 1]  # 1本前
        h4_prev2 = data_4h.iloc[i - 2]  # 2本前
        h4_current = data_4h.iloc[i]    # 現在足
        
        atr_val = h4_current["atr"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        
        trend = h4_current["trend"]
        tolerance = atr_val * 0.3  # 二番底・二番天井の許容誤差
        
        # ロング条件: 二番底の確認
        # - トレンドが上昇
        # - 前々回の安値 ≈ 前回の安値（±tolerance以内）
        # - 前回足が陽線（反転の証拠）
        # - 前回安値が4時間足EMA20付近または下方
        long_signal = False
        long_sl = None
        
        if trend == 1:
            low1 = h4_prev2["low"]
            low2 = h4_prev1["low"]
            prev1_bullish = h4_prev1["close"] > h4_prev1["open"]
            
            # 二番底の条件: 安値が近い（±tolerance）かつ前回が陽線
            if abs(low1 - low2) <= tolerance and prev1_bullish:
                # 損切りは二番底の安値の少し外側
                long_sl = min(low1, low2) - atr_val * 0.15
                long_signal = True
        
        # ショート条件: 二番天井の確認
        short_signal = False
        short_sl = None
        
        if trend == -1:
            high1 = h4_prev2["high"]
            high2 = h4_prev1["high"]
            prev1_bearish = h4_prev1["close"] < h4_prev1["open"]
            
            # 二番天井の条件: 高値が近い（±tolerance）かつ前回が陰線
            if abs(high1 - high2) <= tolerance and prev1_bearish:
                # 損切りは二番天井の高値の少し外側
                short_sl = max(high1, high2) + atr_val * 0.15
                short_signal = True
        
        if not long_signal and not short_signal:
            continue
        
        # 4時間足更新直後の15分足2本以内（30分以内）でエントリー
        # 更新時刻から30分後までの15分足を探す
        entry_window_end = h4_current_time + pd.Timedelta(minutes=30)
        
        # 該当する15分足を探す
        m15_in_window = data_15m[
            (data_15m.index >= h4_current_time) & 
            (data_15m.index < entry_window_end)
        ]
        
        if len(m15_in_window) == 0:
            continue
        
        # 最初の15分足でエントリー（4時間足更新の初動）
        entry_bar = m15_in_window.iloc[0]
        entry_time = entry_bar.name
        
        if long_signal:
            ep = entry_bar["open"] + spread
            risk = ep - long_sl
            if risk <= 0 or risk > atr_val * 3:  # リスクが大きすぎる場合はスキップ
                continue
            tp = ep + risk * rr_ratio
            signals.append({
                "time": entry_time,
                "dir": 1,
                "ep": ep,
                "sl": long_sl,
                "tp": tp,
                "risk": risk,
                "h4_time": h4_current_time,
                "pattern": "double_bottom"
            })
        
        elif short_signal:
            ep = entry_bar["open"] - spread
            risk = short_sl - ep
            if risk <= 0 or risk > atr_val * 3:
                continue
            tp = ep - risk * rr_ratio
            signals.append({
                "time": entry_time,
                "dir": -1,
                "ep": ep,
                "sl": short_sl,
                "tp": tp,
                "risk": risk,
                "h4_time": h4_current_time,
                "pattern": "double_top"
            })
    
    return signals
