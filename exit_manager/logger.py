"""
Exit Manager — JSONL 構造化ログ
================================
1行 = 1イベント。UTF-8 JSONL形式で ./logs/exit_manager.jsonl に追記。
"""

import json
import logging
from datetime import datetime
from pathlib import Path

_std_log = logging.getLogger('sena3fx.exit_manager')


class ExitManagerLogger:
    """
    JSONL構造化ログ出力クラス。

    Usage:
        logger = ExitManagerLogger(log_dir='./logs')
        logger.log('TRADE_REGISTERED', trade_id='12345', symbol='XAU_USD', ...)
    """

    VALID_EVENTS = {
        'TRADE_REGISTERED',
        'SL_PLACED',
        'SL_MODIFIED',
        'TP1_HIT',
        'PARTIAL_CLOSE',
        'TRAILING_UPDATE',
        'REVERSAL_EXIT',
        'PEAK_DD_EXIT',
        'MAX_LOSS_GUARD',
        'LOCKOUT_BLOCKED',
        'MANUAL_OVERRIDE',
        'TRADE_CLOSED',
        'VALIDATION_WARNING',
        'KILL_SWITCH',
        'GIVEBACK_STOP',
        'SILVER_TIME_STOP',
    }

    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'exit_manager.jsonl'

    def log(self, event: str, **kwargs):
        """
        イベントをJSONL形式でログに記録する。

        Args:
            event: イベント種別 (VALID_EVENTS のいずれか)
            **kwargs: イベント固有のフィールド
        """
        record = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'event': event,
        }
        record.update(kwargs)

        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

        # 標準ログにも出力（デバッグ用）
        _std_log.info(f"[{event}] {kwargs.get('trade_id', '')} {kwargs.get('reason', '')}")

    def log_trade_registered(self, trade_id: str, symbol: str, direction: str,
                              entry_price: float, initial_sl: float,
                              sl_distance_usd: float, units: int, risk_jpy: float,
                              tp1_price: float, lockout_until, reason: str = '',
                              **kwargs):
        self.log(
            'TRADE_REGISTERED',
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            initial_sl=initial_sl,
            sl_distance_usd=sl_distance_usd,
            units=units,
            risk_jpy=risk_jpy,
            tp1_price=tp1_price,
            lockout_until=str(lockout_until) if lockout_until else None,
            reason=reason,
            **kwargs,
        )

    def log_tp1_hit(self, trade_id: str, price: float, partial_close_units: int,
                    remaining_units: int, realized_pnl_jpy: float,
                    new_sl: float, new_sl_reason: str, **kwargs):
        self.log(
            'TP1_HIT',
            trade_id=trade_id,
            price=price,
            partial_close_units=partial_close_units,
            remaining_units=remaining_units,
            realized_pnl_jpy=realized_pnl_jpy,
            new_sl=new_sl,
            new_sl_reason=new_sl_reason,
            **kwargs,
        )

    def log_trailing_update(self, trade_id: str, old_sl: float, new_sl: float,
                             reason: str, remaining_units: int, **kwargs):
        self.log(
            'TRAILING_UPDATE',
            trade_id=trade_id,
            old_sl=old_sl,
            new_sl=new_sl,
            reason=reason,
            remaining_units=remaining_units,
            **kwargs,
        )

    def log_trade_closed(self, trade_id: str, price: float, reason: str,
                          pnl_jpy: float, hold_hours: float, **kwargs):
        self.log(
            'TRADE_CLOSED',
            trade_id=trade_id,
            price=price,
            reason=reason,
            pnl_jpy=pnl_jpy,
            hold_hours=round(hold_hours, 2),
            **kwargs,
        )

    def log_lockout_blocked(self, trade_id: str, attempted_action: str,
                             lockout_remaining_min: float, message: str,
                             non_textbook: bool = False, **kwargs):
        self.log(
            'LOCKOUT_BLOCKED',
            trade_id=trade_id,
            attempted_action=attempted_action,
            lockout_remaining_min=round(lockout_remaining_min, 1),
            message=message,
            non_textbook=non_textbook,
            **kwargs,
        )

    def log_validation_warning(self, trade_id: str, warning: str,
                                blocked_action: str = '', **kwargs):
        self.log(
            'VALIDATION_WARNING',
            trade_id=trade_id,
            warning=warning,
            blocked_action=blocked_action,
            **kwargs,
        )

    def log_kill_switch(self, scope: str, total_pnl_jpy: float,
                         threshold_jpy: float, block_new_entries: bool, **kwargs):
        self.log(
            'KILL_SWITCH',
            scope=scope,
            total_pnl_jpy=total_pnl_jpy,
            threshold_jpy=threshold_jpy,
            block_new_entries=block_new_entries,
            **kwargs,
        )

    def read_all(self) -> list:
        """
        JSONL ファイルから全レコードを読み込む。
        evaluator.py が使用する。
        """
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
