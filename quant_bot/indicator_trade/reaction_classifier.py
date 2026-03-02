"""
発表後反応分類モジュール。

経済指標発表後の価格反応を4パターンに分類する。

4パターン:
  1. FOLLOW_THROUGH   順方向: 結果方向に素直に動く
  2. FADE             すぐ戻る: 一時動いてすぐ反転
  3. PENETRATE        貫通: 強烈に一方向に動き続ける
  4. REVERSAL         逆強い: 結果と逆方向に大きく動く
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd


class ReactionPattern(enum.Enum):
    """経済指標発表後の価格反応パターン。"""

    FOLLOW_THROUGH = "順方向"
    """結果方向に素直に動く（最も一般的）"""

    FADE = "すぐ戻る"
    """一時動いたが N バーで元の水準に戻る（Buy the rumor, sell the fact）"""

    PENETRATE = "貫通"
    """強烈に一方向に動き続ける（モメンタム継続）"""

    REVERSAL = "逆強い"
    """結果と逆方向に大きく動く（サプライズ逆張り）"""

    UNKNOWN = "判定不能"
    """データ不足等で判定できない"""


@dataclass
class ReactionResult:
    """発表後反応分類結果。"""

    pattern: ReactionPattern
    """分類されたパターン"""

    direction: str
    """実際の価格移動方向: 'UP' / 'DOWN' / 'FLAT'"""

    expected_direction: str
    """指標結果から期待される方向: 'UP' / 'DOWN'"""

    immediate_move_pct: float
    """発表直後 N バーの価格変動率 (%)"""

    sustained_move_pct: float
    """発表後 M バーの価格変動率 (%)"""

    confidence: float
    """分類の信頼度 0.0〜1.0"""


def classify_reaction(
    pre_event_df: pd.DataFrame,
    post_event_df: pd.DataFrame,
    expected_direction: str,
    immediate_bars: int = 3,
    fade_threshold_pct: float = 0.3,
    penetrate_threshold_pct: float = 0.8,
) -> ReactionResult:
    """
    指標発表前後の価格データから反応パターンを分類。

    Args:
        pre_event_df:           発表直前のOHLCVデータ (末尾バーが発表直前)
        post_event_df:          発表後のOHLCVデータ
        expected_direction:     指標結果から期待される方向 'UP' / 'DOWN'
        immediate_bars:         「即時反応」と判断するバー数
        fade_threshold_pct:     「すぐ戻る」判定の戻り閾値 (0.3 = 30%以上戻ったら Fade)
        penetrate_threshold_pct: 「貫通」判定の持続閾値 (0.8 = 80%以上維持したら Penetrate)

    Returns:
        ReactionResult
    """
    if pre_event_df.empty or post_event_df.empty or len(post_event_df) < immediate_bars:
        return ReactionResult(
            pattern=ReactionPattern.UNKNOWN,
            direction="FLAT",
            expected_direction=expected_direction,
            immediate_move_pct=0.0,
            sustained_move_pct=0.0,
            confidence=0.0,
        )

    base_price = float(pre_event_df["close"].iloc[-1])
    if base_price == 0:
        return ReactionResult(
            pattern=ReactionPattern.UNKNOWN,
            direction="FLAT",
            expected_direction=expected_direction,
            immediate_move_pct=0.0,
            sustained_move_pct=0.0,
            confidence=0.0,
        )

    # 即時価格変動（最初のNバー後の終値）
    immediate_price = float(post_event_df["close"].iloc[min(immediate_bars - 1, len(post_event_df) - 1)])
    immediate_move_pct = (immediate_price - base_price) / base_price * 100

    # 持続価格変動（全ポスト期間の終値）
    sustained_price = float(post_event_df["close"].iloc[-1])
    sustained_move_pct = (sustained_price - base_price) / base_price * 100

    # 実際の移動方向判定
    if immediate_move_pct > 0.05:
        actual_direction = "UP"
    elif immediate_move_pct < -0.05:
        actual_direction = "DOWN"
    else:
        actual_direction = "FLAT"

    # パターン分類
    is_expected_direction = (
        (expected_direction == "UP" and immediate_move_pct > 0) or
        (expected_direction == "DOWN" and immediate_move_pct < 0)
    )

    # 即時移動の絶対値
    imm_abs = abs(immediate_move_pct)
    sus_abs = abs(sustained_move_pct)

    if not is_expected_direction and imm_abs > 0.1:
        # 期待と逆方向に大きく動いた
        pattern = ReactionPattern.REVERSAL
        confidence = min(1.0, imm_abs / 0.5)

    elif is_expected_direction:
        if imm_abs > 0 and sus_abs / max(imm_abs, 0.001) < fade_threshold_pct:
            # 動いたが大半を戻した
            pattern = ReactionPattern.FADE
            confidence = 1.0 - sus_abs / max(imm_abs, 0.001)
        elif sus_abs / max(imm_abs, 0.001) >= penetrate_threshold_pct:
            # 動いた方向を大きく維持
            pattern = ReactionPattern.PENETRATE
            confidence = min(1.0, sus_abs / max(imm_abs, 0.001))
        else:
            # 標準的な順方向反応
            pattern = ReactionPattern.FOLLOW_THROUGH
            confidence = 0.6

    else:
        pattern = ReactionPattern.UNKNOWN
        confidence = 0.0

    return ReactionResult(
        pattern=pattern,
        direction=actual_direction,
        expected_direction=expected_direction,
        immediate_move_pct=round(immediate_move_pct, 4),
        sustained_move_pct=round(sustained_move_pct, 4),
        confidence=round(confidence, 3),
    )
