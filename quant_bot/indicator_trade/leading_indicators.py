"""
先行指標マッピング。

各主要経済指標に対して、事前に注目すべき先行指標（leading indicators）を
ハードコード辞書として管理する。

先行指標は本指標の結果方向を事前に示唆することが多く、
戦略的なポジション構築に活用できる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LeadingIndicator:
    """先行指標の定義。"""

    name: str
    """先行指標の名称"""

    release_timing: str
    """本指標の何週間/何日前に発表されるか"""

    direction_correlation: str
    """'positive' (同方向) / 'negative' (逆方向) / 'mixed' (まちまち)"""

    description: str
    """先行指標が本指標をどう示唆するか"""

    reliability: str = "medium"
    """'high' / 'medium' / 'low'"""


@dataclass
class IndicatorLeadingMap:
    """本指標とその先行指標群の定義。"""

    indicator_name: str
    """本指標の名称"""

    category: str
    """'employment' / 'inflation' / 'growth' / 'housing' / 'consumer' / 'manufacturing'"""

    release_frequency: str
    """'monthly' / 'quarterly' / 'weekly'"""

    xau_impact: str
    """XAU/USD への通常の影響: 'bullish' / 'bearish' / 'mixed'"""

    leading_indicators: list[LeadingIndicator] = field(default_factory=list)

    notes: str = ""


# ----------------------------------------
# 先行指標マッピング辞書
# ----------------------------------------

LEADING_INDICATOR_MAP: dict[str, IndicatorLeadingMap] = {

    "NFP": IndicatorLeadingMap(
        indicator_name="Non-Farm Payrolls (雇用統計)",
        category="employment",
        release_frequency="monthly",
        xau_impact="bearish",  # 強いNFP→ドル高→金安
        leading_indicators=[
            LeadingIndicator(
                name="ADP Employment Report",
                release_timing="本指標の2日前（第1金曜日の水曜）",
                direction_correlation="positive",
                description="民間部門雇用の先行指標。ADP が強いと NFP も強い傾向。",
                reliability="medium",
            ),
            LeadingIndicator(
                name="Jobless Claims (初回失業保険申請件数)",
                release_timing="本指標の週と前週の木曜日",
                direction_correlation="negative",
                description="申請件数が少ないほど雇用市場が強く、NFP 強化を示唆。",
                reliability="medium",
            ),
            LeadingIndicator(
                name="ISM Manufacturing Employment",
                release_timing="本指標の1週間前",
                direction_correlation="positive",
                description="製造業雇用サブ指数。50超なら製造業雇用改善を示唆。",
                reliability="low",
            ),
            LeadingIndicator(
                name="JOLTS (求人労働異動調査)",
                release_timing="本指標の約2週間前",
                direction_correlation="positive",
                description="求人数の増加は将来の雇用増加を示唆。",
                reliability="medium",
            ),
        ],
        notes="最も注目度の高い米雇用統計。金市場への影響大。",
    ),

    "CPI": IndicatorLeadingMap(
        indicator_name="Consumer Price Index (消費者物価指数)",
        category="inflation",
        release_frequency="monthly",
        xau_impact="bullish",  # 高CPI→インフレ懸念→金ヘッジ需要
        leading_indicators=[
            LeadingIndicator(
                name="PPI (生産者物価指数)",
                release_timing="CPI の前日または同週",
                direction_correlation="positive",
                description="企業のコスト上昇が消費者価格に転嫁されるため先行的。",
                reliability="high",
            ),
            LeadingIndicator(
                name="Import Price Index",
                release_timing="CPI の数日前",
                direction_correlation="positive",
                description="輸入物価の上昇は国内インフレの先行指標となる。",
                reliability="medium",
            ),
            LeadingIndicator(
                name="University of Michigan Inflation Expectations",
                release_timing="CPI の約2週間前",
                direction_correlation="positive",
                description="消費者のインフレ期待。実際のCPIと連動することが多い。",
                reliability="low",
            ),
        ],
        notes="FRB の金融政策に直結。予想を大きく上回る場合は特に影響大。",
    ),

    "PCE": IndicatorLeadingMap(
        indicator_name="Personal Consumption Expenditures (PCE デフレーター)",
        category="inflation",
        release_frequency="monthly",
        xau_impact="bullish",
        leading_indicators=[
            LeadingIndicator(
                name="CPI",
                release_timing="PCE の約2週間前",
                direction_correlation="positive",
                description="CPI と PCE は高い相関を持つ。CPI で方向を先読みできる。",
                reliability="high",
            ),
        ],
        notes="FRB が最重視するインフレ指標。コア PCE が特に重要。",
    ),

    "FOMC": IndicatorLeadingMap(
        indicator_name="FOMC Meeting (連邦公開市場委員会)",
        category="monetary_policy",
        release_frequency="6-8 weeks",
        xau_impact="mixed",  # タカ派→金安、ハト派→金高
        leading_indicators=[
            LeadingIndicator(
                name="Fed Fund Futures (FF 金利先物)",
                release_timing="常時",
                direction_correlation="positive",
                description="市場の利上げ/利下げ織り込み確率。確率変動が先行指標。",
                reliability="high",
            ),
            LeadingIndicator(
                name="Fed Speakers (連銀総裁発言)",
                release_timing="FOMC の数週間前",
                direction_correlation="positive",
                description="Fed 当局者の発言で次回会合の方向性を把握できる。",
                reliability="high",
            ),
            LeadingIndicator(
                name="CPI / PCE",
                release_timing="FOMC の数週間前",
                direction_correlation="positive",
                description="インフレ指標が予想より高ければタカ派的姿勢を示唆。",
                reliability="medium",
            ),
        ],
        notes="政策金利変更と声明の両方に注意。ドットチャートも確認。",
    ),

    "GDP": IndicatorLeadingMap(
        indicator_name="Gross Domestic Product (GDP 成長率)",
        category="growth",
        release_frequency="quarterly",
        xau_impact="bearish",  # 強いGDP→リスクオン→金安
        leading_indicators=[
            LeadingIndicator(
                name="ISM Manufacturing PMI",
                release_timing="四半期内で毎月",
                direction_correlation="positive",
                description="製造業景況感。50超が拡大を示し、GDP 成長と連動。",
                reliability="medium",
            ),
            LeadingIndicator(
                name="ISM Services PMI",
                release_timing="四半期内で毎月",
                direction_correlation="positive",
                description="サービス業景況感。米国経済の約70%がサービス業。",
                reliability="medium",
            ),
            LeadingIndicator(
                name="Atlanta Fed GDPNow",
                release_timing="四半期を通じてリアルタイム更新",
                direction_correlation="positive",
                description="Fed アトランタ支部のリアルタイム GDP 成長率推定。",
                reliability="high",
            ),
        ],
        notes="速報値→改定値→確定値の順で発表。速報値が最も市場インパクト大。",
    ),
}


def get_leading_indicators(indicator_name: str) -> Optional[IndicatorLeadingMap]:
    """
    指定した経済指標の先行指標マッピングを返す。

    Args:
        indicator_name: 'NFP' / 'CPI' / 'FOMC' / 'GDP' / 'PCE' 等

    Returns:
        IndicatorLeadingMap または None（未登録の場合）
    """
    return LEADING_INDICATOR_MAP.get(indicator_name.upper())


def list_indicators() -> list[str]:
    """登録済みの経済指標名一覧を返す。"""
    return list(LEADING_INDICATOR_MAP.keys())
