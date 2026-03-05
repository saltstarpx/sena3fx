"""
指標前後 RR 推定モジュール。

経済指標発表前後の ATR ベース RR（リスクリワード比）を推定する。
発表後は通常ボラティリティが拡大するため、ATR 倍率を調整して
より現実的な SL/TP を計算する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EventRR:
    """指標発表前後の RR 推定結果。"""

    pre_event_sl: float
    """発表前推奨 SL 価格"""

    pre_event_tp: float
    """発表前推奨 TP 価格"""

    pre_event_rr: float
    """発表前 RR 比率"""

    post_event_sl: float
    """発表後推奨 SL 価格（ボラティリティ拡大考慮）"""

    post_event_tp: float
    """発表後推奨 TP 価格"""

    post_event_rr: float
    """発表後 RR 比率"""

    atr: float
    """計算に使用した ATR 値"""

    current_price: float
    """現在の参照価格"""


def calc_event_rr(
    ohlcv_df: pd.DataFrame,
    direction: str,
    pre_sl_atr_mult: float = 1.5,
    pre_tp_atr_mult: float = 3.0,
    post_sl_atr_mult: float = 3.0,
    post_tp_atr_mult: float = 6.0,
    atr_period: int = 14,
) -> Optional[EventRR]:
    """
    ATR ベースで発表前後の推奨 SL/TP を計算。

    Args:
        ohlcv_df:        OHLCV DataFrame
        direction:       'LONG' または 'SHORT'
        pre_sl_atr_mult: 発表前 SL 倍率（通常より小さめ）
        pre_tp_atr_mult: 発表前 TP 倍率
        post_sl_atr_mult: 発表後 SL 倍率（ボラ拡大想定で大きめ）
        post_tp_atr_mult: 発表後 TP 倍率
        atr_period:      ATR 計算期間

    Returns:
        EventRR または None (データ不足時)
    """
    if ohlcv_df.empty or len(ohlcv_df) < atr_period:
        return None

    # ATR 計算
    high = ohlcv_df["high"]
    low = ohlcv_df["low"]
    close = ohlcv_df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = tr.rolling(atr_period).mean()
    atr = float(atr_series.iloc[-1])

    if np.isnan(atr) or atr <= 0:
        return None

    price = float(ohlcv_df["close"].iloc[-1])

    if direction == "LONG":
        pre_sl = round(price - pre_sl_atr_mult * atr, 2)
        pre_tp = round(price + pre_tp_atr_mult * atr, 2)
        post_sl = round(price - post_sl_atr_mult * atr, 2)
        post_tp = round(price + post_tp_atr_mult * atr, 2)
    elif direction == "SHORT":
        pre_sl = round(price + pre_sl_atr_mult * atr, 2)
        pre_tp = round(price - pre_tp_atr_mult * atr, 2)
        post_sl = round(price + post_sl_atr_mult * atr, 2)
        post_tp = round(price - post_tp_atr_mult * atr, 2)
    else:
        return None

    pre_sl_dist = abs(price - pre_sl)
    pre_tp_dist = abs(pre_tp - price)
    post_sl_dist = abs(price - post_sl)
    post_tp_dist = abs(post_tp - price)

    pre_rr = round(pre_tp_dist / pre_sl_dist, 2) if pre_sl_dist > 0 else 0.0
    post_rr = round(post_tp_dist / post_sl_dist, 2) if post_sl_dist > 0 else 0.0

    return EventRR(
        pre_event_sl=pre_sl,
        pre_event_tp=pre_tp,
        pre_event_rr=pre_rr,
        post_event_sl=post_sl,
        post_event_tp=post_tp,
        post_event_rr=post_rr,
        atr=round(atr, 4),
        current_price=price,
    )
