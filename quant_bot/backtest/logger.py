"""
トレードイベント JSONL ロガー。

トレードの ENTRY / EXIT / SIGNAL を JSONL 形式で記録する。
各イベントは non_textbook フラグ（教材外ルール）を含む。

JSONL 形式:
  {"timestamp":..., "event":"ENTRY", "instrument":..., "direction":...,
   "price":..., "lots":..., "rule_ids":[...], "non_textbook":false,
   "sl":..., "tp":..., "rr":..., "pnl":null, "account_risk_pct":...}
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("quant_bot.backtest")


@dataclass
class TradeEvent:
    """単一トレードイベント。"""

    timestamp: str
    """ISO 8601 タイムスタンプ"""

    event: str
    """'ENTRY' | 'EXIT' | 'SIGNAL'"""

    instrument: str
    """'XAU_USD' 等"""

    direction: str
    """'LONG' | 'SHORT'"""

    price: float
    """イベント発生時の価格"""

    lots: float
    """取引ロット数"""

    rule_ids: List[str] = field(default_factory=list)
    """適用ルールID リスト (例: ['C1', 'C2', 'C3']"""

    non_textbook: bool = False
    """教材外ルールが適用されたか"""

    sl: Optional[float] = None
    """ストップロス価格"""

    tp: Optional[float] = None
    """テイクプロフィット価格"""

    rr: Optional[float] = None
    """リスクリワード比"""

    pnl: Optional[float] = None
    """損益 (USD) — EXIT 時のみ"""

    account_risk_pct: Optional[float] = None
    """口座残高に対するリスク% — ENTRY 時のみ"""

    exit_reason: Optional[str] = None
    """決済理由 — EXIT 時のみ ('sl_hit', 'tp_hit', 'manual')"""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TradeEventLogger:
    """JSONL 形式でトレードイベントを記録するロガー。"""

    def __init__(self, output_path: str | Path):
        """
        Args:
            output_path: JSONL ファイルのパス
                         親ディレクトリが存在しない場合は自動作成
        """
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[TradeEvent] = []

    # ------------------------------------------------------------------ #
    #  イベント記録                                                        #
    # ------------------------------------------------------------------ #

    def log_entry(
        self,
        timestamp,
        instrument: str,
        direction: str,
        price: float,
        lots: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        rr: Optional[float] = None,
        rule_ids: Optional[List[str]] = None,
        non_textbook: bool = False,
        account_risk_pct: Optional[float] = None,
    ) -> TradeEvent:
        """ENTRY イベントを記録。"""
        event = TradeEvent(
            timestamp=self._ts(timestamp),
            event="ENTRY",
            instrument=instrument,
            direction=direction,
            price=price,
            lots=lots,
            rule_ids=rule_ids or [],
            non_textbook=non_textbook,
            sl=sl,
            tp=tp,
            rr=rr,
            account_risk_pct=account_risk_pct,
        )
        self._append(event)
        return event

    def log_exit(
        self,
        timestamp,
        instrument: str,
        direction: str,
        price: float,
        lots: float,
        pnl: float,
        exit_reason: str = "manual",
        rule_ids: Optional[List[str]] = None,
        non_textbook: bool = False,
    ) -> TradeEvent:
        """EXIT イベントを記録。"""
        event = TradeEvent(
            timestamp=self._ts(timestamp),
            event="EXIT",
            instrument=instrument,
            direction=direction,
            price=price,
            lots=lots,
            rule_ids=rule_ids or [],
            non_textbook=non_textbook,
            pnl=pnl,
            exit_reason=exit_reason,
        )
        self._append(event)
        return event

    def log_signal(
        self,
        timestamp,
        instrument: str,
        direction: str,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        rr: Optional[float] = None,
        rule_ids: Optional[List[str]] = None,
        non_textbook: bool = False,
    ) -> TradeEvent:
        """SIGNAL イベント（未執行シグナル）を記録。"""
        event = TradeEvent(
            timestamp=self._ts(timestamp),
            event="SIGNAL",
            instrument=instrument,
            direction=direction,
            price=price,
            lots=0.0,
            rule_ids=rule_ids or [],
            non_textbook=non_textbook,
            sl=sl,
            tp=tp,
            rr=rr,
        )
        self._append(event)
        return event

    # ------------------------------------------------------------------ #
    #  バックテスト結果一括取込み                                          #
    # ------------------------------------------------------------------ #

    def ingest_backtest_trades(
        self,
        trades: list[dict],
        instrument: str,
        rule_ids: Optional[List[str]] = None,
        non_textbook: bool = False,
        initial_balance: float = 10_000.0,
        risk_pct: float = 2.0,
    ) -> int:
        """
        lib/backtest.BacktestEngine の trades リストを一括取込み。

        Args:
            trades:          BacktestEngine.run() が返す trade dict リスト
            instrument:      'XAU_USD' 等
            rule_ids:        適用ルールID ('C1'〜'C5' 等)
            non_textbook:    教材外ルール使用フラグ
            initial_balance: 初期資金 (risk_pct 計算用)
            risk_pct:        口座リスク%

        Returns:
            取込んだトレード数
        """
        rule_ids = rule_ids or []
        count = 0

        for t in trades:
            direction = t.get("direction", "LONG")
            entry_time = t.get("entry_time")
            exit_time = t.get("exit_time")
            entry_price = float(t.get("entry_price", 0))
            exit_price = float(t.get("exit_price", 0))
            sl = t.get("sl")
            tp = t.get("tp")
            size = float(t.get("size", 1.0))
            pnl = float(t.get("pnl", 0))
            exit_reason = t.get("exit_reason", "manual")

            # RR 計算
            rr = None
            if sl is not None and tp is not None and sl != entry_price:
                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                if sl_dist > 0:
                    rr = round(tp_dist / sl_dist, 2)

            self.log_entry(
                timestamp=entry_time,
                instrument=instrument,
                direction=direction,
                price=entry_price,
                lots=size,
                sl=float(sl) if sl is not None else None,
                tp=float(tp) if tp is not None else None,
                rr=rr,
                rule_ids=rule_ids,
                non_textbook=non_textbook,
                account_risk_pct=risk_pct,
            )

            self.log_exit(
                timestamp=exit_time,
                instrument=instrument,
                direction=direction,
                price=exit_price,
                lots=size,
                pnl=pnl,
                exit_reason=str(exit_reason),
                rule_ids=rule_ids,
                non_textbook=non_textbook,
            )

            count += 1

        log.info(f"バックテスト結果取込み完了: {count} トレード → {self._path}")
        return count

    def flush(self) -> int:
        """バッファをファイルに書き出す（追記モード）。"""
        with self._path.open("a", encoding="utf-8") as f:
            for ev in self._events:
                f.write(ev.to_json() + "\n")
        count = len(self._events)
        self._events.clear()
        return count

    def get_events(self) -> list[TradeEvent]:
        """バッファ内のイベントリストを返す（ファイルへの書き出しなし）。"""
        return list(self._events)

    # ------------------------------------------------------------------ #
    #  内部                                                                #
    # ------------------------------------------------------------------ #

    def _append(self, event: TradeEvent) -> None:
        self._events.append(event)
        # 即時書き出し（バッファリングなし）
        with self._path.open("a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

    @staticmethod
    def _ts(timestamp) -> str:
        if isinstance(timestamp, (pd.Timestamp if False else object,)):
            pass
        try:
            import pandas as pd
            if isinstance(timestamp, pd.Timestamp):
                return timestamp.isoformat()
        except ImportError:
            pass
        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return str(timestamp)
