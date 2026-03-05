"""
C2: ローソク足の強弱 — Single Candle Strength

教材準拠:
  「実体の大きさで強弱を判断」
  「大陽線/大陰線への逆張り禁止（逆らうと死にます）」
  「包み足は転換の鉄板シグナル」

ラップ元: lib/candle.detect_single_candle()

注意:
  detect_single_candle() は i=1 からループを開始するため、
  先頭バーは必ず 'neutral' になる。確定バースライスを渡して iloc[-1] を読む。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.candle import detect_single_candle  # noqa: E402

from .base import ConditionBase, ConditionResult

# lib/yagami.py lines 126-141 と同じセット
# ロング根拠になるローソク足タイプ
BULL_TYPES = frozenset(
    {"big_bull", "engulf_bull", "hammer", "pinbar_bull", "pullback_bear"}
)
# ショート根拠になるローソク足タイプ
BEAR_TYPES = frozenset(
    {"big_bear", "engulf_bear", "inv_hammer", "pinbar_bear", "pullback_bull"}
)

# 逆張り禁止ローソク足（R2ルール）
# big_bull/big_bear に対してはシグナル方向と逆のエントリーをしてはいけない
COUNTER_TREND_FORBIDDEN = frozenset({"big_bull", "big_bear"})


class C2CandleStrength(ConditionBase):
    """C2: ローソク足単体の強弱でエントリー根拠を判定。"""

    CONDITION_ID = "C2"

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の 'c2' セクション dict

        注意: lib/candle.py の閾値はハードコードされているため、
             config の body_ratio_threshold 等はドキュメント目的のみ。
        """
        self._body_ratio_threshold = float(config.get("body_ratio_threshold", 0.5))

    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> ConditionResult:
        # ライブバーを除外
        confirmed = self._confirmed(ohlcv_df)
        if len(confirmed) < 2:  # engulf_bull/bear は前足が必要
            return self._not_enough_data("C2: 確認バー不足（最低2本必要）")

        # 確定バーのスライスに対して分析（確定バー全体を渡す）
        enriched = detect_single_candle(confirmed)
        last = enriched.iloc[-1]

        candle_type = str(last["candle_type"])
        candle_strength = float(last.get("candle_strength", 0.0))

        is_bull = candle_type in BULL_TYPES
        is_bear = candle_type in BEAR_TYPES
        satisfied = is_bull or is_bear
        direction = "BULL" if is_bull else ("BEAR" if is_bear else "NEUTRAL")

        # 実体比率の計算
        o = float(last["open"])
        c_price = float(last["close"])
        h = float(last["high"])
        lw = float(last["low"])
        rng = h - lw
        body_ratio = round(abs(c_price - o) / rng, 3) if rng > 0 else 0.0

        return ConditionResult(
            condition_id=self.CONDITION_ID,
            satisfied=satisfied,
            score=round(abs(candle_strength), 3),
            reason=(
                f"C2: {candle_type}, 実体比率={body_ratio:.2f}, "
                f"強度={candle_strength:+.2f} ({direction})"
            ),
            details={
                "candle_type": candle_type,
                "candle_strength": candle_strength,
                "body_ratio": body_ratio,
                "direction": direction,
                "is_counter_trend_forbidden": candle_type in COUNTER_TREND_FORBIDDEN,
                "bar_timestamp": str(last.name),
            },
        )
