"""
broker_base.py - ブローカー抽象基底クラス
=============================================
OANDA / Exness(MetaApi) を統一インターフェースで扱う。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class BrokerBase(ABC):
    """ブローカー共通インターフェース"""

    @abstractmethod
    def get_candles(self, symbol: str, granularity: str,
                    count: int = 200) -> pd.DataFrame:
        """OHLCVデータ取得。空DataFrameで失敗を示す。"""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """現在のmid価格。0.0で失敗。"""

    @abstractmethod
    def place_order(self, symbol: str, units: int,
                    sl: float, tp: float) -> dict:
        """成行注文。{"trade_id": str, "fill_price": float} or {}"""

    @abstractmethod
    def close_trade(self, trade_id: str) -> dict:
        """ポジション決済。{"exit_price": float} or {}"""

    @abstractmethod
    def get_account_equity(self) -> float:
        """口座残高（JPY）"""

    @abstractmethod
    def get_open_positions(self) -> dict:
        """ブローカー上のオープンポジション一覧。
        {position_id: {"symbol": str, "type": "buy"|"sell", "profit": float}} or {}"""

    def partial_close(self, trade_id: str, volume: float) -> dict:
        """ポジションの一部（volume lot分）を決済する。半利確用。
        デフォルトは全決済にフォールバック。"""
        return self.close_trade(trade_id)

    def modify_position(self, trade_id: str, sl: float, tp: float = None) -> bool:
        """ポジションのSL/TPを変更する。デフォルトは未対応（False）。"""
        return False
