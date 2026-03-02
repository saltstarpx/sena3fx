"""
C2CandleStrength テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.entry_engine.conditions.c2_candle import C2CandleStrength

DEFAULT_CONFIG = {"min_strength": 0.3}


class TestC2CandleStrength:

    @pytest.fixture
    def c2(self):
        return C2CandleStrength(DEFAULT_CONFIG)

    def test_returns_condition_result(self, c2):
        """evaluate() が ConditionResult を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c2.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C2"
        assert isinstance(result.satisfied, bool)
        assert 0.0 <= result.score <= 1.0

    def test_insufficient_data_returns_not_satisfied(self, c2):
        """データ不足の場合は satisfied=False を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=1)
        result = c2.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.satisfied is False

    def test_live_bar_excluded(self):
        """センチネル注入でライブバーが除外されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        df_sentinel = df.copy()
        df_sentinel.iloc[-1, df_sentinel.columns.get_loc("close")] = 999_999.99

        c2 = C2CandleStrength(DEFAULT_CONFIG)
        r_normal = c2.evaluate(df, "XAU_USD", "H4", df.index[-1])
        r_sentinel = c2.evaluate(df_sentinel, "XAU_USD", "H4", df.index[-1])

        assert r_normal.satisfied == r_sentinel.satisfied
        assert abs(r_normal.score - r_sentinel.score) < 1e-6

    def test_details_contain_candle_type(self, c2):
        """details に candle_type が含まれることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c2.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert "candle_type" in result.details

    def test_details_contain_direction(self, c2):
        """details に direction が含まれることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c2.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert "direction" in result.details
        assert result.details["direction"] in ("BULL", "BEAR", "NONE")

    def test_strong_bullish_candle_setup(self):
        """
        大陽線を模した合成データで C2 が充足するかテスト。

        # テスト用合成データ — 実市場データではありません
        注意: 実際の判定は lib/candle.detect_single_candle() に依存するため、
              結果は実装に依存する。ここでは例外が発生しないことを主に確認。
        """
        # テスト用合成データ — 実市場データではありません
        rng = np.random.default_rng(99)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")
        df = pd.DataFrame(
            {
                "open":  np.full(n, 2000.0),
                "high":  np.full(n, 2010.0),
                "low":   np.full(n, 1995.0),
                "close": np.full(n, 2008.0),
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        df.index.name = "datetime"

        c2 = C2CandleStrength(DEFAULT_CONFIG)
        result = c2.evaluate(df, "XAU_USD", "H4", df.index[-1])
        # 例外なく実行できることを確認
        assert result.condition_id == "C2"
