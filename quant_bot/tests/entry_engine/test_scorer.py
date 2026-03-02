"""
スコアラー (ConditionScorer) テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.entry_engine.conditions.base import ConditionResult
from quant_bot.entry_engine.scorer import ConditionScore, ConditionScorer


def make_result(
    cid: str,
    satisfied: bool,
    score: float = 0.5,
    direction: str = "NONE",
    candle_type: str = "",
) -> ConditionResult:
    """テスト用 ConditionResult ファクトリ。"""
    return ConditionResult(
        condition_id=cid,
        satisfied=satisfied,
        score=score,
        reason=f"テスト: {cid}",
        details={"direction": direction, "candle_type": candle_type},
    )


class TestConditionScorer:
    """ConditionScorer の基本動作テスト。"""

    @pytest.fixture
    def scorer(self):
        return ConditionScorer({"entry_threshold": 3, "strong_signal_threshold": 4})

    @pytest.fixture
    def confirmed_df(self):
        # テスト用合成データ — 実市場データではありません
        return make_synthetic_ohlcv(n=50)

    def test_no_signal_below_threshold(self, scorer, confirmed_df):
        """3条件未満の場合は NO_SIGNAL になることを確認。"""
        results = [
            make_result("C1", True, direction="BULL"),
            make_result("C2", True, direction="BULL"),
            make_result("C3", False),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.signal is None
        assert score.strength == "NO_SIGNAL"
        assert score.grade in ("B", "C")

    def test_grade_c_for_two_satisfied(self, scorer, confirmed_df):
        """2条件以下で C グレードになることを確認。"""
        results = [
            make_result("C1", True),
            make_result("C2", True),
            make_result("C3", False),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.grade == "C"
        assert score.total_score == 2

    def test_grade_b_for_three_satisfied(self, scorer, confirmed_df):
        """3条件で B グレードになることを確認。"""
        results = [
            make_result("C1", True, direction="BULL"),
            make_result("C2", True, direction="BULL"),
            make_result("C3", True, direction="BULL"),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.grade == "B"
        assert score.total_score == 3
        assert score.signal == "LONG"
        assert score.strength == "MODERATE"

    def test_grade_a_for_four_satisfied(self, scorer, confirmed_df):
        """4条件以上で A グレード、STRONG になることを確認。"""
        results = [
            make_result("C1", True, score=0.8, direction="BULL"),
            make_result("C2", True, score=0.9, direction="BULL"),
            make_result("C3", True, score=0.7, direction="BULL"),
            make_result("C4", True, score=0.6, direction="BULL"),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.grade == "A"
        assert score.total_score == 4
        assert score.signal == "LONG"
        assert score.strength == "STRONG"

    def test_long_signal_when_bull_dominates(self, scorer, confirmed_df):
        """BULL スコアが優勢なら LONG シグナルになることを確認。"""
        results = [
            make_result("C1", True, score=1.0, direction="BULL"),
            make_result("C2", True, score=1.0, direction="BULL"),
            make_result("C3", True, score=1.0, direction="BULL"),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.signal == "LONG"
        assert score.bull_score > score.bear_score

    def test_short_signal_when_bear_dominates(self, scorer, confirmed_df):
        """BEAR スコアが優勢なら SHORT シグナルになることを確認。"""
        results = [
            make_result("C1", True, score=1.0, direction="BEAR"),
            make_result("C2", True, score=1.0, direction="BEAR"),
            make_result("C3", True, score=1.0, direction="BEAR"),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.signal == "SHORT"
        assert score.bear_score > score.bull_score

    def test_r2_big_bull_cancels_bear(self, scorer, confirmed_df):
        """
        R2: big_bull 検出時に BEAR スコアがゼロになることを確認。
        （大陽線への逆張り SHORT 禁止）
        """
        results = [
            make_result("C1", True, score=0.5, direction="BEAR"),
            make_result("C2", True, score=1.0, direction="BULL", candle_type="big_bull"),
            make_result("C3", True, score=0.5, direction="BEAR"),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        # big_bull により bear_score はゼロになるべき
        assert score.bear_score == 0.0, (
            f"big_bull 時に bear_score がゼロになっていません: {score.bear_score}"
        )

    def test_r2_big_bear_cancels_bull(self, scorer, confirmed_df):
        """
        R2: big_bear 検出時に BULL スコアがゼロになることを確認。
        （大陰線への逆張り LONG 禁止）
        """
        results = [
            make_result("C1", True, score=0.5, direction="BULL"),
            make_result("C2", True, score=1.0, direction="BEAR", candle_type="big_bear"),
            make_result("C3", True, score=0.5, direction="BULL"),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.bull_score == 0.0, (
            f"big_bear 時に bull_score がゼロになっていません: {score.bull_score}"
        )

    def test_sl_tp_calculated_for_valid_signal(self, scorer, confirmed_df):
        """有効なシグナルがある場合 SL/TP が計算されることを確認。"""
        results = [
            make_result("C1", True, score=1.0, direction="BULL"),
            make_result("C2", True, score=1.0, direction="BULL"),
            make_result("C3", True, score=1.0, direction="BULL"),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        if score.signal is not None:
            # SL/TP は計算されるはず（ATR が 0 でなければ）
            # データが十分であれば計算できる
            assert score.suggested_sl is not None or len(confirmed_df) < 14

    def test_no_sl_tp_for_no_signal(self, scorer, confirmed_df):
        """NO_SIGNAL の場合は SL/TP が None であることを確認。"""
        results = [
            make_result("C1", False),
            make_result("C2", False),
            make_result("C3", False),
            make_result("C4", False),
            make_result("C5", False),
        ]
        score = scorer.score(results, confirmed_df)
        assert score.signal is None
        assert score.suggested_sl is None
        assert score.suggested_tp is None
        assert score.risk_reward is None
