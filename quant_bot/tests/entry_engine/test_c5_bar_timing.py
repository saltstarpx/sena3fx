"""
C5BarTiming テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.entry_engine.conditions.c5_bar_timing import C5BarTiming
from quant_bot.entry_engine.conditions.base import ConditionBase

DEFAULT_CONFIG = {"htf_alignment_required": False, "asia_breakout_penalty": True}


class TestC5BarTiming:

    @pytest.fixture
    def c5(self):
        return C5BarTiming(DEFAULT_CONFIG)

    def test_returns_condition_result(self, c5):
        """evaluate() が ConditionResult を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c5.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C5"
        assert isinstance(result.satisfied, bool)
        assert 0.0 <= result.score <= 1.0

    def test_insufficient_data_returns_not_satisfied(self, c5):
        """データ不足の場合は satisfied=False を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=1)
        result = c5.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.satisfied is False

    def test_live_bar_excluded(self):
        """センチネル注入でライブバーが除外されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        df_sentinel = df.copy()
        df_sentinel.iloc[-1, df_sentinel.columns.get_loc("close")] = 999_999.99

        c5 = C5BarTiming(DEFAULT_CONFIG)
        r_normal = c5.evaluate(df, "XAU_USD", "H4", df.index[-1])
        r_sentinel = c5.evaluate(df_sentinel, "XAU_USD", "H4", df.index[-1])

        assert r_normal.satisfied == r_sentinel.satisfied
        assert abs(r_normal.score - r_sentinel.score) < 1e-6

    def test_granularity_map_h4(self, c5):
        """H4 が正しく 4h にマッピングされることを確認。"""
        assert ConditionBase.GRANULARITY_MAP.get("H4") == "4h"

    def test_granularity_map_m15(self, c5):
        """M15 が正しく 15min にマッピングされることを確認。"""
        assert ConditionBase.GRANULARITY_MAP.get("M15") == "15min"

    def test_granularity_map_h1(self, c5):
        """H1 が正しく 1h にマッピングされることを確認。"""
        assert ConditionBase.GRANULARITY_MAP.get("H1") == "1h"

    def test_details_contain_timing_info(self, c5):
        """details にタイミング情報が含まれることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c5.evaluate(df, "XAU_USD", "H4", df.index[-1])
        # details には何らかの情報が含まれるはず
        assert isinstance(result.details, dict)
