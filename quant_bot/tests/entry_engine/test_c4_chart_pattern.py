"""
C4ChartPattern テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.entry_engine.conditions.c4_chart_pattern import C4ChartPattern

DEFAULT_CONFIG = {"pivot_window": 5, "min_bars": 25}


class TestC4ChartPattern:

    @pytest.fixture
    def c4(self):
        return C4ChartPattern(DEFAULT_CONFIG)

    def test_returns_condition_result(self, c4):
        """evaluate() が ConditionResult を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c4.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C4"
        assert isinstance(result.satisfied, bool)
        assert 0.0 <= result.score <= 1.0

    def test_insufficient_data_returns_not_satisfied(self, c4):
        """min_bars 未満のデータでは satisfied=False を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=20)  # min_bars=25 未満
        result = c4.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.satisfied is False

    def test_live_bar_excluded(self):
        """センチネル注入でライブバーが除外されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        df_sentinel = df.copy()
        df_sentinel.iloc[-1, df_sentinel.columns.get_loc("close")] = 999_999.99

        c4 = C4ChartPattern(DEFAULT_CONFIG)
        r_normal = c4.evaluate(df, "XAU_USD", "H4", df.index[-1])
        r_sentinel = c4.evaluate(df_sentinel, "XAU_USD", "H4", df.index[-1])

        assert r_normal.satisfied == r_sentinel.satisfied
        assert abs(r_normal.score - r_sentinel.score) < 1e-6

    def test_details_contain_chart_pattern(self, c4):
        """details に chart_pattern が含まれることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c4.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert "chart_pattern" in result.details

    def test_border_case_exactly_min_bars(self):
        """min_bars ちょうどのデータでエラーなく動くことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=26)  # confirmed は 25 本 = min_bars
        c4 = C4ChartPattern(DEFAULT_CONFIG)
        result = c4.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C4"
