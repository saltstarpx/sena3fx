"""
最適銘柄選定モジュール。

マクロレジーム × 指標カテゴリから推奨トレード銘柄を選定する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .macro_regime import MacroRegime


@dataclass
class InstrumentRecommendation:
    """銘柄推奨結果。"""

    instrument: str
    """推奨銘柄 (例: 'XAU_USD', 'USD_JPY')"""

    direction: str
    """推奨方向: 'LONG' / 'SHORT' / 'NEUTRAL'"""

    confidence: str
    """信頼度: 'HIGH' / 'MEDIUM' / 'LOW'"""

    reason: str
    """推奨理由"""

    alternatives: list[str]
    """代替銘柄リスト"""


# レジーム × 指標カテゴリ → 推奨銘柄・方向
_SELECTION_MAP: dict[tuple[MacroRegime, str], InstrumentRecommendation] = {

    # === GOLDILOCKS (低インフレ・好景気) ===
    (MacroRegime.GOLDILOCKS, "employment"): InstrumentRecommendation(
        instrument="USD_JPY",
        direction="LONG",
        confidence="HIGH",
        reason="好景気 × 強い雇用 → ドル高・円安",
        alternatives=["US30_USD", "NAS100_USD"],
    ),
    (MacroRegime.GOLDILOCKS, "inflation"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="NEUTRAL",
        confidence="LOW",
        reason="ゴルディロックスでは金の必要性が低い",
        alternatives=["US30_USD"],
    ),
    (MacroRegime.GOLDILOCKS, "growth"): InstrumentRecommendation(
        instrument="NAS100_USD",
        direction="LONG",
        confidence="HIGH",
        reason="好景気 → テクノロジー・グロース株有利",
        alternatives=["US30_USD", "USD_JPY"],
    ),

    # === LOW_INF_WEAK (低インフレ・景気悪) ===
    (MacroRegime.LOW_INF_WEAK, "employment"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="MEDIUM",
        reason="景気悪化懸念 → 安全資産としての金需要",
        alternatives=["NAS100_USD"],
    ),
    (MacroRegime.LOW_INF_WEAK, "inflation"): InstrumentRecommendation(
        instrument="NAS100_USD",
        direction="LONG",
        confidence="MEDIUM",
        reason="低インフレ → 低金利継続 → グロース株有利",
        alternatives=["XAU_USD"],
    ),
    (MacroRegime.LOW_INF_WEAK, "growth"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="HIGH",
        reason="景気後退 → リスクオフ → 金買い",
        alternatives=["USD_JPY"],  # 円も安全資産
    ),

    # === HIGH_INF_GOOD (高インフレ・好景気) ===
    (MacroRegime.HIGH_INF_GOOD, "employment"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="MEDIUM",
        reason="高インフレ → インフレヘッジとしての金",
        alternatives=["US30_USD"],
    ),
    (MacroRegime.HIGH_INF_GOOD, "inflation"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="HIGH",
        reason="高CPI/PPI → インフレ懸念 → 金ヘッジ需要増",
        alternatives=["XAGUSD"],
    ),
    (MacroRegime.HIGH_INF_GOOD, "growth"): InstrumentRecommendation(
        instrument="US30_USD",
        direction="LONG",
        confidence="MEDIUM",
        reason="好景気 → バリュー株（ダウ）有利",
        alternatives=["XAU_USD"],
    ),

    # === STAGFLATION (高インフレ・景気悪) ===
    (MacroRegime.STAGFLATION, "employment"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="HIGH",
        reason="スタグフレーション → 金への逃避需要が最大",
        alternatives=["XAG_USD"],
    ),
    (MacroRegime.STAGFLATION, "inflation"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="HIGH",
        reason="高インフレ + 景気悪化 → 金が最良のヘッジ",
        alternatives=["XAG_USD"],
    ),
    (MacroRegime.STAGFLATION, "growth"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="HIGH",
        reason="GDP 悪化 + スタグフレーション → 全資産売り・金保有",
        alternatives=["USD_JPY"],  # 円安が進む可能性
    ),
    (MacroRegime.STAGFLATION, "monetary_policy"): InstrumentRecommendation(
        instrument="XAU_USD",
        direction="LONG",
        confidence="HIGH",
        reason="FOMC が利上げ困難なスタグフレーション環境 → 金最強",
        alternatives=["XAG_USD"],
    ),
}

# デフォルト推奨（マッピングにない場合）
_DEFAULT_RECOMMENDATION = InstrumentRecommendation(
    instrument="XAU_USD",
    direction="NEUTRAL",
    confidence="LOW",
    reason="レジーム/指標の組み合わせに対するマッピングなし。中立で観察。",
    alternatives=[],
)


def select_instrument(
    regime: MacroRegime,
    indicator_category: str,
) -> InstrumentRecommendation:
    """
    マクロレジームと指標カテゴリから推奨銘柄を選定。

    Args:
        regime:             classify_regime() で取得した MacroRegime
        indicator_category: 'employment' / 'inflation' / 'growth' /
                            'housing' / 'consumer' / 'manufacturing' /
                            'monetary_policy' 等

    Returns:
        InstrumentRecommendation
    """
    key = (regime, indicator_category.lower())
    return _SELECTION_MAP.get(key, _DEFAULT_RECOMMENDATION)
