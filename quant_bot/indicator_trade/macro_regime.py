"""
マクロレジーム判定モジュール。

インフレトレンド × 景気成長の組み合わせで4つのレジームを判定する。

  GOLDILOCKS    低インフレ・好景気 → リスクオン全般
  LOW_INF_WEAK  低インフレ・景気悪 → グロース株・ナスダック
  HIGH_INF_GOOD 高インフレ・景気良 → バリュー株・ダウ
  STAGFLATION   高インフレ・景気悪 → コモディティ・金・全売り
"""
from __future__ import annotations

import enum
from typing import Literal


class MacroRegime(enum.Enum):
    """マクロ経済レジーム。"""

    GOLDILOCKS = "ゴルディロックス"
    """低インフレ・好景気: 株・リスク資産全般に有利"""

    LOW_INF_WEAK = "低インフレ景気悪"
    """低インフレ・景気後退: グロース株（ナスダック）有利"""

    HIGH_INF_GOOD = "高インフレ景気良"
    """高インフレ・好景気: バリュー株（ダウ）・コモディティ有利"""

    STAGFLATION = "スタグフレーション"
    """高インフレ・景気後退: 全資産売り・金（XAU）防衛買い"""


# インフレトレンドの有効な入力値
InflationTrend = Literal["rising", "stable", "falling"]

# 景気成長の有効な入力値
GrowthTrend = Literal["positive", "neutral", "negative"]

# レジームマトリクス
_REGIME_MATRIX: dict[tuple[str, str], MacroRegime] = {
    ("rising",  "positive"): MacroRegime.HIGH_INF_GOOD,
    ("rising",  "neutral"):  MacroRegime.HIGH_INF_GOOD,
    ("rising",  "negative"): MacroRegime.STAGFLATION,
    ("stable",  "positive"): MacroRegime.GOLDILOCKS,
    ("stable",  "neutral"):  MacroRegime.GOLDILOCKS,
    ("stable",  "negative"): MacroRegime.LOW_INF_WEAK,
    ("falling", "positive"): MacroRegime.GOLDILOCKS,
    ("falling", "neutral"):  MacroRegime.LOW_INF_WEAK,
    ("falling", "negative"): MacroRegime.LOW_INF_WEAK,
}

# 各レジームでの XAU/USD バイアス
REGIME_XAU_BIAS: dict[MacroRegime, str] = {
    MacroRegime.GOLDILOCKS:    "NEUTRAL",   # リスクオン→金需要低下気味
    MacroRegime.LOW_INF_WEAK:  "SLIGHT_BULL",  # 景気悪化→安全資産需要
    MacroRegime.HIGH_INF_GOOD: "BULL",      # インフレヘッジ需要
    MacroRegime.STAGFLATION:   "STRONG_BULL",  # 最強の金買い環境
}

# 各レジームでの推奨行動
REGIME_RECOMMENDATION: dict[MacroRegime, str] = {
    MacroRegime.GOLDILOCKS:    "株・リスク資産買い。金は中立。",
    MacroRegime.LOW_INF_WEAK:  "ナスダック・グロース株。金は防衛的に保有。",
    MacroRegime.HIGH_INF_GOOD: "ダウ・バリュー株・コモディティ・金。",
    MacroRegime.STAGFLATION:   "金・商品に集中。株は売り。現金保有。",
}


def classify_regime(
    cpi_trend: InflationTrend,
    gdp_growth: GrowthTrend,
) -> MacroRegime:
    """
    CPI トレンドと GDP 成長からマクロレジームを判定。

    Args:
        cpi_trend:  'rising' | 'stable' | 'falling'
        gdp_growth: 'positive' | 'neutral' | 'negative'

    Returns:
        MacroRegime

    Example:
        >>> classify_regime(cpi_trend='rising', gdp_growth='negative')
        <MacroRegime.STAGFLATION: 'スタグフレーション'>
    """
    key = (cpi_trend, gdp_growth)
    regime = _REGIME_MATRIX.get(key)
    if regime is None:
        raise ValueError(
            f"無効な組み合わせ: cpi_trend={cpi_trend!r}, gdp_growth={gdp_growth!r}. "
            f"cpi_trend は {list({'rising','stable','falling'})} のいずれか、"
            f"gdp_growth は {list({'positive','neutral','negative'})} のいずれか。"
        )
    return regime


def get_xau_bias(regime: MacroRegime) -> str:
    """レジームから XAU/USD のバイアスを取得。"""
    return REGIME_XAU_BIAS.get(regime, "NEUTRAL")


def get_recommendation(regime: MacroRegime) -> str:
    """レジームから推奨アクションを取得。"""
    return REGIME_RECOMMENDATION.get(regime, "判定不能")
