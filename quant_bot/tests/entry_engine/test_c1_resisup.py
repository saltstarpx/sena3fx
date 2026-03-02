"""
C1ResistanceSupport テスト。

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
from quant_bot.entry_engine.conditions.c1_resisup import C1ResistanceSupport


DEFAULT_CONFIG = {
    "atr_multiplier": 1.5,
    "min_touch_count": 2,
    "level_lookback": 100,
}


class TestC1ResistanceSupport:

    @pytest.fixture
    def c1(self):
        return C1ResistanceSupport(DEFAULT_CONFIG)

    def test_returns_condition_result(self, c1):
        """evaluate() が ConditionResult を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c1.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C1"
        assert isinstance(result.satisfied, bool)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.reason, str)

    def test_insufficient_data_returns_not_satisfied(self, c1):
        """データ不足の場合は satisfied=False を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=5)  # 非常に少ないデータ
        result = c1.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.satisfied is False

    def test_live_bar_excluded(self):
        """ライブバーのセンチネル価格がスコアに影響しないことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        df_sentinel = df.copy()
        df_sentinel.iloc[-1, df_sentinel.columns.get_loc("close")] = 999_999.99

        c1 = C1ResistanceSupport(DEFAULT_CONFIG)
        result_normal = c1.evaluate(df, "XAU_USD", "H4", df.index[-1])
        result_sentinel = c1.evaluate(df_sentinel, "XAU_USD", "H4", df.index[-1])

        # センチネルの有無でスコアが変わらないこと
        assert result_normal.satisfied == result_sentinel.satisfied
        assert abs(result_normal.score - result_sentinel.score) < 1e-6

    def test_condition_id_is_c1(self, c1):
        """condition_id が 'C1' であることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c1.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.condition_id == "C1"

    def test_non_textbook_false(self, c1):
        """non_textbook フラグが False であることを確認（C1 は教材準拠）。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        result = c1.evaluate(df, "XAU_USD", "H4", df.index[-1])
        assert result.non_textbook is False
