"""
5条件スコアリング集約モジュール。

やがみメソッド グレード評価:
  A評価 (4-5条件充足): 高品質エントリー → STRONG
  B評価 (3条件充足):   慎重にエントリー → MODERATE
  C評価 (2条件以下):   エントリー禁止 → NO_SIGNAL

方向判定: 充足条件のスコア加重多数決 (BULL vs BEAR)
禁止ルール適用:
  - big_bull 検出時: LONG禁止（方向が逆のシグナルをキャンセル）
  - big_bear 検出時: SHORT禁止
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .conditions.base import ConditionResult


@dataclass
class ConditionScore:
    """5条件集約スコアと最終シグナル。"""

    total_score: int
    """充足した条件数 (0〜5)"""

    conditions: dict[str, ConditionResult]
    """条件ID → ConditionResult のマッピング"""

    signal: Optional[str]
    """'LONG' / 'SHORT' / None"""

    strength: str
    """'STRONG' / 'MODERATE' / 'NO_SIGNAL'"""

    grade: str
    """'A' (4-5条件) / 'B' (3条件) / 'C' (2条件以下)"""

    bull_score: float = 0.0
    """ロング方向の加重スコア合計"""

    bear_score: float = 0.0
    """ショート方向の加重スコア合計"""

    suggested_sl: Optional[float] = None
    """ATRベースの推奨SL価格"""

    suggested_tp: Optional[float] = None
    """ATRベースの推奨TP価格"""

    risk_reward: Optional[float] = None
    """推奨RR比率"""


class ConditionScorer:
    """5つの ConditionResult を集約して ConditionScore を生成。"""

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の 'scorer' セクション dict
        """
        self._entry_threshold = int(config.get("entry_threshold", 3))
        self._strong_threshold = int(config.get("strong_signal_threshold", 4))

    def score(
        self,
        results: list[ConditionResult],
        confirmed_df: pd.DataFrame,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0,
    ) -> ConditionScore:
        """
        5つの ConditionResult から ConditionScore を生成。

        Args:
            results:       C1〜C5 の ConditionResult リスト
            confirmed_df:  最後の確定バースライス (SL/TP計算用)
            sl_atr_mult:   SL距離の ATR 倍率
            tp_atr_mult:   TP距離の ATR 倍率

        Returns:
            ConditionScore
        """
        conditions = {r.condition_id: r for r in results}
        satisfied_count = sum(1 for r in results if r.satisfied)

        # グレード判定
        if satisfied_count >= 4:
            grade = "A"
        elif satisfied_count >= 3:
            grade = "B"
        else:
            grade = "C"

        # 閾値未満はシグナルなし
        if satisfied_count < self._entry_threshold:
            return ConditionScore(
                total_score=satisfied_count,
                conditions=conditions,
                signal=None,
                strength="NO_SIGNAL",
                grade=grade,
            )

        # 方向判定: 充足条件の direction × score で加重集計
        bull_score = 0.0
        bear_score = 0.0
        for r in results:
            if not r.satisfied:
                continue
            direction = r.details.get("direction", "NONE")
            if direction == "BULL":
                bull_score += r.score
            elif direction == "BEAR":
                bear_score += r.score

        # 大陽線/大陰線への逆張り禁止ルール (R2: やがみ「逆らうと死にます」)
        c2 = conditions.get("C2")
        if c2 and c2.satisfied:
            candle_type = c2.details.get("candle_type", "")
            if candle_type == "big_bull":
                bear_score = 0.0  # big_bull に対してショートは禁止
            elif candle_type == "big_bear":
                bull_score = 0.0  # big_bear に対してロングは禁止

        # シグナル方向決定
        if bull_score > bear_score:
            signal = "LONG"
        elif bear_score > bull_score:
            signal = "SHORT"
        else:
            signal = None  # 拮抗 → シグナルなし

        strength = "STRONG" if satisfied_count >= self._strong_threshold else "MODERATE"

        # SL/TP 計算
        sl, tp, rr = self._calc_sl_tp(confirmed_df, signal, sl_atr_mult, tp_atr_mult)

        return ConditionScore(
            total_score=satisfied_count,
            conditions=conditions,
            signal=signal,
            strength=strength,
            grade=grade,
            bull_score=round(bull_score, 3),
            bear_score=round(bear_score, 3),
            suggested_sl=sl,
            suggested_tp=tp,
            risk_reward=rr,
        )

    def _calc_sl_tp(
        self,
        confirmed_df: pd.DataFrame,
        signal: Optional[str],
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        ATRベースで推奨 SL/TP を計算。

        Returns:
            (suggested_sl, suggested_tp, risk_reward) または (None, None, None)
        """
        if confirmed_df.empty or signal is None:
            return None, None, None

        from .conditions.base import ConditionBase

        atr_series = ConditionBase._calc_atr(confirmed_df)
        atr = float(atr_series.iloc[-1])
        if np.isnan(atr) or atr <= 0:
            return None, None, None

        price = float(confirmed_df.iloc[-1]["close"])

        if signal == "LONG":
            sl = round(price - sl_atr_mult * atr, 2)
            tp = round(price + tp_atr_mult * atr, 2)
        else:  # SHORT
            sl = round(price + sl_atr_mult * atr, 2)
            tp = round(price - tp_atr_mult * atr, 2)

        sl_dist = abs(price - sl)
        tp_dist = abs(tp - price)
        rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else None

        return sl, tp, rr
