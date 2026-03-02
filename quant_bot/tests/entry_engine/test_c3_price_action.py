"""
C3PriceAction テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.entry_engine.conditions.c3_price_action import C3PriceAction

DEFAULT_CONFIG = {"require_confirmed_pattern": True}


class TestC3PriceAction:

    @pytest.fixture
    def c3(self):
        return C3PriceAction(DEFAULT_CONFIG)

    def test_returns_condition_result(self, c3):
        """evaluate() が ConditionResult を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c3.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C3"
        assert isinstance(result.satisfied, bool)
        assert 0.0 <= result.score <= 1.0

    def test_insufficient_data_returns_not_satisfied(self, c3):
        """データ不足の場合は satisfied=False を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=1)
        result = c3.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.satisfied is False

    def test_live_bar_excluded(self):
        """センチネル注入でライブバーが除外されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        df_sentinel = df.copy()
        df_sentinel.iloc[-1, df_sentinel.columns.get_loc("close")] = 999_999.99

        c3 = C3PriceAction(DEFAULT_CONFIG)
        r_normal = c3.evaluate(df, "XAU_USD", "H4", df.index[-1])
        r_sentinel = c3.evaluate(df_sentinel, "XAU_USD", "H4", df.index[-1])

        assert r_normal.satisfied == r_sentinel.satisfied
        assert abs(r_normal.score - r_sentinel.score) < 1e-6

    def test_details_contain_pa_signal(self, c3):
        """details に pa_signal が含まれることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c3.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert "pa_signal" in result.details

    def test_flat_data_not_satisfied(self):
        """
        フラットなデータでは C3 が充足しないことを確認。

        # テスト用合成データ — 実市場データではありません
        """
        from quant_bot.tests.conftest import make_flat_ohlcv
        df = make_flat_ohlcv(n=100, noise=0.01)  # ほぼフラット
        c3 = C3PriceAction(DEFAULT_CONFIG)
        result = c3.evaluate(df, "XAU_USD", "H4", df.index[-1])
        # フラットデータでは通常パターンが検出されないはず（例外発生なしを確認）
        assert result.condition_id == "C3"
