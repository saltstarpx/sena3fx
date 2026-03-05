"""
Exit Manager — トレード状態管理
================================
TradePhase 列挙型と TradeState データクラスを定義。
メモリ内レジストリ（TradeRegistry）で全アクティブトレードを管理する。

設計方針:
  OANDA は決済の「正」のソース。
  TradeRegistry は Exit Manager の「出口管理メタデータ」のみを追跡する:
    - TP1が済んでいるか
    - 1R距離（R倍率計算の基準）
    - フェーズ（OPEN/TP1_HIT/TRAILING/CLOSED）
    - ピーク含み益R（Giveback Stop用）

再起動時:
  レジストリは空になる。main.py が OANDA のオープントレードと照合し、
  未知の trade_id を自動登録する（フェーズ=OPEN として保守的に扱う）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TradePhase(Enum):
    """
    トレードの状態遷移フェーズ。

    遷移図:
      REGISTERED → OPEN → TP1_HIT → TRAILING → CLOSED
                    ↓       ↓            ↓
                  SL_HIT  SL_HIT   REVERSAL_EXIT
                    ↓       ↓       PEAK_DD_EXIT
                  CLOSED  CLOSED      CLOSED
                              ↓（どの状態からも）
                        MAX_LOSS_GUARD → CLOSED
    """
    REGISTERED = "REGISTERED"  # CLIで登録済み、未約定確認
    OPEN       = "OPEN"        # 約定済み、TP1前
    TP1_HIT    = "TP1_HIT"     # TP1到達、半利済み
    TRAILING   = "TRAILING"    # トレール中（残ポジ）
    CLOSED     = "CLOSED"      # 完全決済（終端状態）


@dataclass
class TradeState:
    """
    1トレードの Exit Manager メタデータ。

    フィールド:
        trade_id:          OANDA トレードID（str）
        instrument:        "XAU_USD" | "XAG_USD"
        side:              "long" | "short"
        entry_price:       約定価格（USD）
        entry_time:        約定時刻（UTC、TZなし ← 既存コードと統一）
        sl_price:          現在のSL価格（USD）。登録時に必須
        tp_price:          TP価格（USD）。None可
        original_units:    エントリー時のユニット数
        current_units:     現在のオープンユニット数
        one_r_jpy:         このトレードの1R（JPY）
        sl_distance_usd:   abs(entry_price - initial_sl) = 1Rの距離（USD）
        phase:             現在のフェーズ
        tp1_executed:      True = TP1部分利確済み（繰り返し防止）
        peak_unrealized_r: 含み益の最大R倍率（Giveback Stop用）
        non_textbook:      True = 非教材ルールが適用済み
        notes:             自由記述（ログ用）
    """
    trade_id:          str
    instrument:        str
    side:              str           # "long" | "short"
    entry_price:       float
    entry_time:        datetime       # UTC、TZなし
    sl_price:          float
    tp_price:          Optional[float] = None
    original_units:    int = 0
    current_units:     int = 0
    one_r_jpy:         float = 150_000.0
    sl_distance_usd:   float = 0.0
    phase:             TradePhase = TradePhase.REGISTERED
    tp1_executed:      bool = False
    peak_unrealized_r: float = 0.0
    non_textbook:      bool = False
    notes:             str = ""

    def unrealized_r(self, current_price: float) -> float:
        """
        現在の含み益をR倍率で返す。

        LONG:  (current_price - entry_price) / sl_distance_usd
        SHORT: (entry_price - current_price) / sl_distance_usd

        Returns:
            float: 正=利益, 負=損失
        """
        if self.sl_distance_usd <= 0:
            return 0.0
        if self.side == 'long':
            return (current_price - self.entry_price) / self.sl_distance_usd
        else:
            return (self.entry_price - current_price) / self.sl_distance_usd

    def unrealized_pnl_jpy(self, current_price: float, jpy_per_usd: float) -> float:
        """
        現在ユニット数での含み損益（JPY）を計算する。

        Args:
            current_price: 現在の価格（USD）
            jpy_per_usd:   jpy_per_dollar_per_unit（instrument設定値）

        Returns:
            float: 損益（JPY）。正=利益, 負=損失
        """
        if self.side == 'long':
            pnl_usd = (current_price - self.entry_price) * self.current_units
        else:
            pnl_usd = (self.entry_price - current_price) * self.current_units
        return pnl_usd * jpy_per_usd

    def sl_distance_r(self, price: float) -> float:
        """
        指定価格とSLの距離をR倍率で返す。
        主に「建値からSLまでのバッファ確認」に使用。
        """
        if self.sl_distance_usd <= 0:
            return 0.0
        return abs(price - self.sl_price) / self.sl_distance_usd


class TradeRegistry:
    """
    アクティブトレードのメモリ内レジストリ。

    OANDA が決済の正のソース。このレジストリは Exit Manager の
    出口管理メタデータのみを追跡する。

    Usage:
        registry = TradeRegistry()
        registry.register(trade)
        active = registry.get_active_trades()
        registry.update('12345', sl_price=1995.0, phase=TradePhase.TRAILING)
        registry.mark_closed('12345')
    """

    def __init__(self):
        self._trades: dict[str, TradeState] = {}

    def register(self, trade: TradeState) -> None:
        """トレードを登録（または上書き）する。"""
        self._trades[trade.trade_id] = trade

    def get(self, trade_id: str) -> Optional[TradeState]:
        """指定IDのトレードを返す。存在しない場合は None。"""
        return self._trades.get(trade_id)

    def update(self, trade_id: str, **kwargs) -> Optional[TradeState]:
        """
        指定フィールドを更新する。更新後の TradeState を返す。
        trade_id が存在しない場合は None。
        """
        t = self._trades.get(trade_id)
        if t is None:
            return None
        for k, v in kwargs.items():
            setattr(t, k, v)
        return t

    def get_active_trades(self) -> list[TradeState]:
        """CLOSED 以外の全トレードを返す。"""
        return [t for t in self._trades.values() if t.phase != TradePhase.CLOSED]

    def get_all_trades(self) -> list[TradeState]:
        """全トレード（CLOSED含む）を返す。"""
        return list(self._trades.values())

    def mark_closed(self, trade_id: str) -> None:
        """指定トレードを CLOSED フェーズにする。"""
        t = self._trades.get(trade_id)
        if t is not None:
            t.phase = TradePhase.CLOSED

    def is_kill_switch_active(self) -> bool:
        """
        Kill Switch フラグ。main.py が設定・参照する。
        ここでは状態変数としてのみ保持。
        """
        return getattr(self, '_kill_switch_active', False)

    def set_kill_switch(self, active: bool) -> None:
        """Kill Switch 状態を設定する。"""
        self._kill_switch_active = active
