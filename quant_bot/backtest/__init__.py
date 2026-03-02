"""
quant_bot.backtest — バックテストインフラ (Module 3)

lib/backtest.BacktestEngine をラップし、JSONL 形式の
トレードイベントロギングを追加する。
"""
from .engine import QuantBacktestEngine
from .logger import TradeEventLogger, TradeEvent
from .risk_manager import (
    VolatilityAdjustedSizer,
    KellyCriterionSizer,
    HybridKellySizer,
    LivermorePyramidingSizer,
)

__all__ = [
    "QuantBacktestEngine",
    "TradeEventLogger",
    "TradeEvent",
    "VolatilityAdjustedSizer",
    "KellyCriterionSizer",
    "HybridKellySizer",
    "LivermorePyramidingSizer",
]
