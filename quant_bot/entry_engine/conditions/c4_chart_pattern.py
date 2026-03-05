"""
C4: チャートパターン — Chart Pattern Detection

教材準拠:
  「三角持ち合い: 三角形の高さ分がTPの最低値幅」
  「フラッグの下落角度25-30°が理想」
  「ウェッジは先端方向と逆にブレイクする確率60%」
  「逆三尊: 上位足での環境認識必須」

ラップ元: lib/patterns.detect_chart_patterns()

注意:
  detect_chart_patterns() は内部で i=20 からループを開始し、
  _find_pivots() は window=5 両端を必要とする。
  最低 25 本の確定バーが必要 (config.min_bars のデフォルト値)。

  pattern_tp はATR倍率ではなく価格単位 (USD) の距離。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.patterns import detect_chart_patterns  # noqa: E402

from .base import ConditionBase, ConditionResult

# ロング根拠になるチャートパターン
BULL_PATTERNS = frozenset(
    {
        "inv_hs_long",          # 逆三尊
        "flag_bull",             # ブルフラッグ
        "wedge_bull",            # ウェッジ上抜け
        "triangle_break_bull",   # 三角持ち合い上抜け
        "ascending_tri",         # アセンディングトライアングル
    }
)
# ショート根拠になるチャートパターン
BEAR_PATTERNS = frozenset(
    {
        "hs_short",              # 三尊
        "flag_bear",             # ベアフラッグ
        "wedge_bear",            # ウェッジ下抜け
        "triangle_break_bear",   # 三角持ち合い下抜け
        "descending_tri",        # ディセンディングトライアングル
    }
)


class C4ChartPattern(ConditionBase):
    """C4: チャートパターンの成立・ブレイクアウトを判定。"""

    CONDITION_ID = "C4"

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の 'c4' セクション dict

        config キー:
            pivot_window: スイングハイ/ロー検出ウィンドウ (デフォルト 5)
            min_bars:     最低確定バー数 (デフォルト 25)
                          lib 内部ループが i=20 から開始し、
                          pivot_window=5 両端が必要なため 25 以上を推奨
        """
        self._pivot_window = int(config.get("pivot_window", 5))
        self._min_bars = int(config.get("min_bars", 25))

    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> ConditionResult:
        # ライブバーを除外
        confirmed = self._confirmed(ohlcv_df)
        if len(confirmed) < self._min_bars:
            return self._not_enough_data(
                f"C4: 確認バー不足（最低{self._min_bars}本, 現在{len(confirmed)}本）"
            )

        enriched = detect_chart_patterns(confirmed, pivot_window=self._pivot_window)
        last = enriched.iloc[-1]

        chart_pattern = last.get("chart_pattern")   # str or None
        pattern_tp = float(last.get("pattern_tp", 0.0))  # 価格単位 (USD)

        is_bull = chart_pattern in BULL_PATTERNS
        is_bear = chart_pattern in BEAR_PATTERNS
        satisfied = is_bull or is_bear
        direction = "BULL" if is_bull else ("BEAR" if is_bear else "NONE")

        # スコア: pattern_tp を ATR の 5 倍で正規化
        score = 0.0
        if satisfied and pattern_tp > 0:
            atr_val = float(self._calc_atr(confirmed).iloc[-1])
            if atr_val > 0:
                score = round(min(1.0, pattern_tp / (atr_val * 5.0)), 3)
            else:
                score = 0.5

        return ConditionResult(
            condition_id=self.CONDITION_ID,
            satisfied=satisfied,
            score=score,
            reason=(
                f"C4: {chart_pattern or 'パターン未成立'}, "
                f"推奨TP距離={pattern_tp:.2f}USD ({direction})"
            ),
            details={
                "chart_pattern": chart_pattern,
                "pattern_tp": pattern_tp,
                "direction": direction,
                "bar_timestamp": str(last.name),
            },
        )
