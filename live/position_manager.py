"""
XAU/XAG 排他的ポジション管理
==============================
ルール:
  - XAU_USD と XAG_USD の同時保有は禁止
  - どちらかにポジションがある場合、もう一方のエントリーをスキップ
  - 新規シグナルが出ても既存ポジションがある場合はスキップ

設計:
  OANDA API をリアルタイムで問い合わせてポジション状況を確認する。
  ローカルキャッシュは持たず、常にブローカーの状態を正として判断する。
"""

import logging
from typing import Optional

log = logging.getLogger('sena3fx')


class PositionManager:
    """
    XAU_USD / XAG_USD の排他的ポジション管理クラス。

    Usage:
        >>> pm = PositionManager(broker)
        >>> if pm.can_enter('XAU_USD'):
        ...     broker.place_market_order(...)
    """

    MANAGED_INSTRUMENTS = {'XAU_USD', 'XAG_USD'}
    COUNTERPART = {
        'XAU_USD': 'XAG_USD',
        'XAG_USD': 'XAU_USD',
    }

    def __init__(self, broker):
        """
        Args:
            broker: OandaBroker インスタンス
        """
        self.broker = broker

    def can_enter(self, instrument: str) -> bool:
        """
        指定通貨ペアへの新規エントリー可否を判定する。

        チェック内容:
          1. 自分自身のポジションが既にあるか
          2. 相手側 (XAU ↔ XAG) にポジションがあるか

        Args:
            instrument: 'XAU_USD' | 'XAG_USD'

        Returns:
            True  = エントリー可 (どちらにもポジションなし)
            False = エントリー不可
        """
        if instrument not in self.MANAGED_INSTRUMENTS:
            # XAU/XAG 以外は排他制御対象外 → 常に許可
            return True

        # 自分自身のポジション確認
        long_u, short_u = self.broker.get_position(instrument)
        if long_u != 0 or short_u != 0:
            log.info(
                f"[{instrument}] 既存ポジションあり "
                f"(long={long_u}, short={short_u}) → スキップ"
            )
            return False

        # 相手側のポジション確認 (排他制御)
        counterpart = self.COUNTERPART.get(instrument)
        if counterpart:
            long_u_cp, short_u_cp = self.broker.get_position(counterpart)
            if long_u_cp != 0 or short_u_cp != 0:
                log.info(
                    f"[{instrument}] {counterpart}にポジションあり "
                    f"(排他制御) → スキップ"
                )
                return False

        return True

    def has_any_position(self) -> bool:
        """XAU_USD または XAG_USD のいずれかにポジションがあるか確認"""
        for inst in self.MANAGED_INSTRUMENTS:
            long_u, short_u = self.broker.get_position(inst)
            if long_u != 0 or short_u != 0:
                return True
        return False

    def get_current_positions(self) -> dict:
        """
        現在の全ポジション状況を取得する。

        Returns:
            dict: {
                'XAU_USD': {'long': int, 'short': int},
                'XAG_USD': {'long': int, 'short': int},
            }
        """
        result = {}
        for inst in self.MANAGED_INSTRUMENTS:
            long_u, short_u = self.broker.get_position(inst)
            result[inst] = {'long': long_u, 'short': short_u}
        return result

    def log_status(self):
        """現在のポジション状況をログに出力する"""
        positions = self.get_current_positions()
        for inst, pos in positions.items():
            if pos['long'] != 0 or pos['short'] != 0:
                log.info(f"  {inst}: long={pos['long']}, short={pos['short']}")
            else:
                log.debug(f"  {inst}: ポジションなし")
