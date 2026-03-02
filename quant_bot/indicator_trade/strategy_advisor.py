"""
指標トレード戦略アドバイザー。

経済指標発表に対して3つのトレード手法のどれを採用すべきか提案する。

3手法:
  1. PRE_BET     事前BET: 発表前にポジション構築（先行指標が明確な場合）
  2. POST_SHORT  結果見て短期: 発表後の即時反応にエントリー
  3. LEADING_SWING 先行スイング: 先行指標でポジション→本指標で利確
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from .macro_regime import MacroRegime
from .reaction_classifier import ReactionPattern


class TradingMethod(enum.Enum):
    """指標トレードの3手法。"""

    PRE_BET = "事前BET"
    """先行指標が明確な場合に発表前にポジション構築"""

    POST_SHORT = "結果見て短期"
    """発表後の即時反応（最初の数バー）にエントリー"""

    LEADING_SWING = "先行スイング"
    """先行指標でポジション→本指標発表時に利確"""

    NO_TRADE = "見送り"
    """エントリーなし（条件が整わない場合）"""


@dataclass
class StrategyAdvice:
    """戦略提案結果。"""

    method: TradingMethod
    """推奨手法"""

    confidence: str
    """信頼度: 'HIGH' / 'MEDIUM' / 'LOW'"""

    entry_timing: str
    """エントリータイミングの説明"""

    exit_timing: str
    """決済タイミングの説明"""

    key_risks: list[str]
    """主なリスク"""

    reason: str
    """提案理由"""


class StrategyAdvisor:
    """指標トレード戦略提案クラス。"""

    def advise(
        self,
        indicator_name: str,
        regime: MacroRegime,
        leading_signal_strength: str = "weak",
        historical_reaction: Optional[ReactionPattern] = None,
        hours_to_event: float = 24.0,
        non_textbook_enabled: bool = True,
    ) -> StrategyAdvice:
        """
        指標トレード戦略を提案。

        Args:
            indicator_name:          経済指標名 ('NFP', 'CPI', 'FOMC' 等)
            regime:                  現在のマクロレジーム
            leading_signal_strength: 先行指標の強さ 'strong' / 'moderate' / 'weak'
            historical_reaction:     過去の典型的な反応パターン (ReactionPattern)
            hours_to_event:          指標発表まで何時間か
            non_textbook_enabled:    教材外ルール (NT3) を適用するか

        Returns:
            StrategyAdvice
        """
        # NT3: 超重要指標の 48 時間前フィルター
        if non_textbook_enabled and hours_to_event <= 48 and hours_to_event > 4:
            major_events = {"NFP", "FOMC", "CPI", "PCE", "GDP"}
            if indicator_name.upper() in major_events:
                return StrategyAdvice(
                    method=TradingMethod.NO_TRADE,
                    confidence="HIGH",
                    entry_timing="発表後4時間経過後に検討",
                    exit_timing="—",
                    key_risks=["指標直前のスプレッド拡大", "方向性の不確実性"],
                    reason=(
                        f"NT3 適用: {indicator_name} 発表まで {hours_to_event:.0f}時間。"
                        "超重要指標の48時間前〜発表後4時間はエントリー禁止。"
                    ),
                )

        # 先行指標が強く、かつ発表まで時間がある場合 → 事前BET
        if leading_signal_strength == "strong" and hours_to_event >= 48:
            return StrategyAdvice(
                method=TradingMethod.PRE_BET,
                confidence="MEDIUM",
                entry_timing="先行指標発表直後にポジション構築",
                exit_timing="本指標発表の2時間前に利確（スプレッド拡大回避）",
                key_risks=[
                    "先行指標が本指標と乖離するリスク",
                    "マーケットコンセンサスが既に織り込み済みの可能性",
                ],
                reason=(
                    f"先行指標シグナルが強い（{leading_signal_strength}）。"
                    f"発表まで {hours_to_event:.0f}時間の余裕あり。"
                    "先行スイングより短期のBETを推奨。"
                ),
            )

        # 過去に FOLLOW_THROUGH パターンが多い指標 → 結果見て短期
        if historical_reaction in (ReactionPattern.FOLLOW_THROUGH, ReactionPattern.PENETRATE):
            return StrategyAdvice(
                method=TradingMethod.POST_SHORT,
                confidence="MEDIUM",
                entry_timing="発表後の最初のローソク足が確定したらエントリー",
                exit_timing="ATR × 2 の TP または発表後4時間以内に撤退",
                key_risks=[
                    "初動が速すぎてエントリーできないリスク",
                    "スプレッドが拡大している可能性",
                ],
                reason=(
                    f"過去の {indicator_name} は {historical_reaction.value} パターンが多い。"
                    "発表後の即時反応を狙う短期手法を推奨。"
                ),
            )

        # 先行指標シグナルが強く、発表まで1週間以上ある場合 → 先行スイング
        if leading_signal_strength in ("strong", "moderate") and hours_to_event >= 168:
            return StrategyAdvice(
                method=TradingMethod.LEADING_SWING,
                confidence="LOW",
                entry_timing="先行指標（ADP, JOLTS 等）発表後にポジション構築",
                exit_timing="本指標発表の前後でポジションを閉じる",
                key_risks=[
                    "先行指標との乖離リスク（ADP-NFP の相関は 0.5〜0.7 程度）",
                    "保有期間が長いため日足レベルのリスク管理が必要",
                    "スワップコストの発生",
                ],
                reason=(
                    f"先行指標シグナル（{leading_signal_strength}）あり、"
                    f"発表まで {hours_to_event:.0f}時間。"
                    "先行指標でポジション→本指標で利確の先行スイングを推奨。"
                ),
            )

        # デフォルト: 結果見て短期
        return StrategyAdvice(
            method=TradingMethod.POST_SHORT,
            confidence="LOW",
            entry_timing="発表後の値動きが一方向に定まったらエントリー",
            exit_timing="ATR × 1.5 の TP または当日中に撤退",
            key_risks=[
                "指標後のボラティリティ拡大",
                "方向感が定まらない可能性",
            ],
            reason=(
                "明確な先行指標シグナルなし。"
                "発表後の反応を確認してからエントリーする安全手法を採用。"
            ),
        )
