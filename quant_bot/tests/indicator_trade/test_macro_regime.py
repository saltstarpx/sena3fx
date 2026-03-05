"""
indicator_trade/macro_regime.py テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.indicator_trade.macro_regime import (
    MacroRegime,
    classify_regime,
    get_xau_bias,
    get_recommendation,
)


class TestClassifyRegime:

    def test_stagflation(self):
        """高インフレ + 景気後退 = スタグフレーション。"""
        regime = classify_regime(cpi_trend="rising", gdp_growth="negative")
        assert regime == MacroRegime.STAGFLATION

    def test_goldilocks(self):
        """安定インフレ + 好景気 = ゴルディロックス。"""
        regime = classify_regime(cpi_trend="stable", gdp_growth="positive")
        assert regime == MacroRegime.GOLDILOCKS

    def test_high_inf_good(self):
        """上昇インフレ + 好景気 = 高インフレ景気良。"""
        regime = classify_regime(cpi_trend="rising", gdp_growth="positive")
        assert regime == MacroRegime.HIGH_INF_GOOD

    def test_low_inf_weak(self):
        """低下インフレ + 景気後退 = 低インフレ景気悪。"""
        regime = classify_regime(cpi_trend="falling", gdp_growth="negative")
        assert regime == MacroRegime.LOW_INF_WEAK

    def test_falling_inflation_positive_growth_goldilocks(self):
        """インフレ低下 + 好景気 = ゴルディロックス。"""
        regime = classify_regime(cpi_trend="falling", gdp_growth="positive")
        assert regime == MacroRegime.GOLDILOCKS

    def test_all_combinations_covered(self):
        """全9組み合わせが有効な MacroRegime を返すことを確認。"""
        for cpi in ["rising", "stable", "falling"]:
            for gdp in ["positive", "neutral", "negative"]:
                regime = classify_regime(cpi_trend=cpi, gdp_growth=gdp)
                assert isinstance(regime, MacroRegime)

    def test_invalid_cpi_raises_error(self):
        """無効な cpi_trend は ValueError を発生させることを確認。"""
        with pytest.raises(ValueError):
            classify_regime(cpi_trend="unknown", gdp_growth="positive")

    def test_invalid_gdp_raises_error(self):
        """無効な gdp_growth は ValueError を発生させることを確認。"""
        with pytest.raises(ValueError):
            classify_regime(cpi_trend="rising", gdp_growth="boom")


class TestRegimeMetadata:

    def test_stagflation_xau_bias_strong_bull(self):
        """スタグフレーションでの XAU バイアスが STRONG_BULL であることを確認。"""
        bias = get_xau_bias(MacroRegime.STAGFLATION)
        assert bias == "STRONG_BULL"

    def test_goldilocks_xau_bias_neutral(self):
        """ゴルディロックスでの XAU バイアスが NEUTRAL であることを確認。"""
        bias = get_xau_bias(MacroRegime.GOLDILOCKS)
        assert bias == "NEUTRAL"

    def test_all_regimes_have_recommendation(self):
        """全レジームに推奨テキストが設定されていることを確認。"""
        for regime in MacroRegime:
            rec = get_recommendation(regime)
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_recommendation_differs_by_regime(self):
        """異なるレジームで異なる推奨が返ることを確認。"""
        recs = {regime: get_recommendation(regime) for regime in MacroRegime}
        # 全て異なることを確認
        assert len(set(recs.values())) == len(MacroRegime)
