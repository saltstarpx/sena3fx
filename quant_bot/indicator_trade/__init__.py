"""
quant_bot.indicator_trade — 指標トレード支援モジュール (Module 2)

経済指標の発表を活用したトレード戦略支援ツール群。
  - 経済カレンダー取得 (Investing.com スクレイピング)
  - 先行指標マッピング
  - マクロレジーム判定
  - 最適銘柄選定
  - RR 推定
  - 発表後反応分類
  - 戦略提案
"""
from .macro_regime import MacroRegime, classify_regime
from .leading_indicators import get_leading_indicators
from .instrument_selector import select_instrument
from .reaction_classifier import ReactionPattern, classify_reaction
from .strategy_advisor import StrategyAdvisor, TradingMethod

__all__ = [
    "MacroRegime",
    "classify_regime",
    "get_leading_indicators",
    "select_instrument",
    "ReactionPattern",
    "classify_reaction",
    "StrategyAdvisor",
    "TradingMethod",
]
