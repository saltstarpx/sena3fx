"""
最重要テスト: フューチャーデータリーケージ防止テスト。

センチネル価格 (999_999.99) をライブバー (iloc[-1]) に注入し、
全5条件クラスがそのバーを読まないことを検証する。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# quant_bot へのパス
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.entry_engine.conditions.base import ConditionBase
from quant_bot.entry_engine.conditions.c1_resisup import C1ResistanceSupport
from quant_bot.entry_engine.conditions.c2_candle import C2CandleStrength
from quant_bot.entry_engine.conditions.c3_price_action import C3PriceAction
from quant_bot.entry_engine.conditions.c4_chart_pattern import C4ChartPattern
from quant_bot.entry_engine.conditions.c5_bar_timing import C5BarTiming

SENTINEL_PRICE = 999_999.99


def make_sentinel_df(n: int = 100, seed: int = 42) -> tuple:
    """
    センチネル注入済みデータフレームを生成。

    Returns:
        (df_with_sentinel, df_clean):
        df_with_sentinel: iloc[-1] がセンチネル価格のデータ
        df_clean: センチネルなしの通常データ
    """
    # テスト用合成データ — 実市場データではありません
    df = make_synthetic_ohlcv(n=n, base_price=2000.0, seed=seed)
    df_sentinel = df.copy()

    # ライブバー (iloc[-1]) にセンチネルを注入
    df_sentinel.iloc[-1, df_sentinel.columns.get_loc("open")]  = SENTINEL_PRICE
    df_sentinel.iloc[-1, df_sentinel.columns.get_loc("high")]  = SENTINEL_PRICE + 10
    df_sentinel.iloc[-1, df_sentinel.columns.get_loc("low")]   = SENTINEL_PRICE - 10
    df_sentinel.iloc[-1, df_sentinel.columns.get_loc("close")] = SENTINEL_PRICE

    return df_sentinel, df


class TestConfirmedMethod:
    """ConditionBase._confirmed() の正確性テスト。"""

    def test_confirmed_excludes_last_bar(self, synthetic_ohlcv):
        """_confirmed() が iloc[-1] を除外することを確認。"""
        df = synthetic_ohlcv
        confirmed = ConditionBase._confirmed(df)

        assert len(confirmed) == len(df) - 1
        assert confirmed.index[-1] == df.index[-2]
        assert confirmed.index[0] == df.index[0]

    def test_confirmed_empty_for_single_bar(self):
        """1本のデータでは _confirmed() が空を返すことを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=1)
        confirmed = ConditionBase._confirmed(df)
        assert len(confirmed) == 0

    def test_confirmed_empty_for_empty_df(self):
        """空DataFrameでは _confirmed() が空を返すことを確認。"""
        import pandas as pd
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        confirmed = ConditionBase._confirmed(df)
        assert len(confirmed) == 0

    def test_confirmed_returns_n_minus_1_rows(self):
        """_confirmed() が正確に n-1 本を返すことを確認。"""
        for n in [2, 10, 50, 100, 200]:
            # テスト用合成データ — 実市場データではありません
            df = make_synthetic_ohlcv(n=n)
            confirmed = ConditionBase._confirmed(df)
            assert len(confirmed) == n - 1, f"n={n}: expected {n-1}, got {len(confirmed)}"


class TestSentinelNotLeaked:
    """センチネル注入テスト — 各条件クラスがライブバーを読まないことを確認。"""

    def _default_c1_config(self):
        return {"atr_multiplier": 1.5, "min_touch_count": 2, "level_lookback": 50}

    def _default_c2_config(self):
        return {"min_strength": 0.3}

    def _default_c3_config(self):
        return {"require_confirmed_pattern": True}

    def _default_c4_config(self):
        return {"pivot_window": 5, "min_bars": 25}

    def _default_c5_config(self):
        return {"htf_alignment_required": False}

    def _has_sentinel(self, result) -> bool:
        """ConditionResult にセンチネル値が含まれるかチェック。"""
        sentinel_str = str(SENTINEL_PRICE)
        # reason にセンチネルが含まれていないか
        if sentinel_str in result.reason:
            return True
        # details にセンチネルが含まれていないか
        for v in result.details.values():
            if sentinel_str in str(v):
                return True
        # score が 1.0 を超えていないか（通常は 0.0〜1.0）
        if result.score > 1.0:
            return True
        return False

    def test_c1_does_not_read_live_bar(self):
        """C1 がライブバーのセンチネル価格を読まないことを確認。"""
        df_sentinel, _ = make_sentinel_df(n=100)
        c1 = C1ResistanceSupport(self._default_c1_config())
        result = c1.evaluate(df_sentinel, "XAU_USD", "H4", df_sentinel.index[-1])

        assert not self._has_sentinel(result), (
            f"C1 がライブバーのセンチネル価格 {SENTINEL_PRICE} を読んでいます！\n"
            f"reason: {result.reason}\ndetails: {result.details}"
        )

    def test_c2_does_not_read_live_bar(self):
        """C2 がライブバーのセンチネル価格を読まないことを確認。"""
        df_sentinel, _ = make_sentinel_df(n=100)
        c2 = C2CandleStrength(self._default_c2_config())
        result = c2.evaluate(df_sentinel, "XAU_USD", "H4", df_sentinel.index[-1])

        assert not self._has_sentinel(result), (
            f"C2 がライブバーのセンチネル価格 {SENTINEL_PRICE} を読んでいます！\n"
            f"reason: {result.reason}\ndetails: {result.details}"
        )

    def test_c3_does_not_read_live_bar(self):
        """C3 がライブバーのセンチネル価格を読まないことを確認。"""
        df_sentinel, _ = make_sentinel_df(n=100)
        c3 = C3PriceAction(self._default_c3_config())
        result = c3.evaluate(df_sentinel, "XAU_USD", "H4", df_sentinel.index[-1])

        assert not self._has_sentinel(result), (
            f"C3 がライブバーのセンチネル価格 {SENTINEL_PRICE} を読んでいます！\n"
            f"reason: {result.reason}\ndetails: {result.details}"
        )

    def test_c4_does_not_read_live_bar(self):
        """C4 がライブバーのセンチネル価格を読まないことを確認。"""
        df_sentinel, _ = make_sentinel_df(n=100)
        c4 = C4ChartPattern(self._default_c4_config())
        result = c4.evaluate(df_sentinel, "XAU_USD", "H4", df_sentinel.index[-1])

        assert not self._has_sentinel(result), (
            f"C4 がライブバーのセンチネル価格 {SENTINEL_PRICE} を読んでいます！\n"
            f"reason: {result.reason}\ndetails: {result.details}"
        )

    def test_c5_does_not_read_live_bar(self):
        """C5 がライブバーのセンチネル価格を読まないことを確認。"""
        df_sentinel, _ = make_sentinel_df(n=100)
        c5 = C5BarTiming(self._default_c5_config())
        result = c5.evaluate(df_sentinel, "XAU_USD", "H4", df_sentinel.index[-1])

        assert not self._has_sentinel(result), (
            f"C5 がライブバーのセンチネル価格 {SENTINEL_PRICE} を読んでいます！\n"
            f"reason: {result.reason}\ndetails: {result.details}"
        )

    def test_all_conditions_same_result_with_and_without_sentinel(self):
        """
        センチネルありとなしで全条件の結果が同一であることを確認。
        （ライブバーを読んでいないなら結果は変わらないはず）
        """
        df_sentinel, df_clean = make_sentinel_df(n=100)

        configs = {
            "c1": self._default_c1_config(),
            "c2": self._default_c2_config(),
            "c3": self._default_c3_config(),
            "c4": self._default_c4_config(),
            "c5": self._default_c5_config(),
        }

        conditions = [
            ("C1", C1ResistanceSupport(configs["c1"])),
            ("C2", C2CandleStrength(configs["c2"])),
            ("C3", C3PriceAction(configs["c3"])),
            ("C4", C4ChartPattern(configs["c4"])),
            ("C5", C5BarTiming(configs["c5"])),
        ]

        for name, cond in conditions:
            ts = df_sentinel.index[-1]
            result_sentinel = cond.evaluate(df_sentinel, "XAU_USD", "H4", ts)
            result_clean = cond.evaluate(df_clean, "XAU_USD", "H4", ts)

            assert result_sentinel.satisfied == result_clean.satisfied, (
                f"{name}: センチネルありとなしで satisfied が異なります！"
                f"（sentinel={result_sentinel.satisfied}, clean={result_clean.satisfied}）"
                " → ライブバーを読んでいる可能性があります。"
            )
            assert abs(result_sentinel.score - result_clean.score) < 1e-6, (
                f"{name}: センチネルありとなしで score が異なります！"
                f"（sentinel={result_sentinel.score}, clean={result_clean.score}）"
                " → ライブバーを読んでいる可能性があります。"
            )
