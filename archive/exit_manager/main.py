"""
Exit Manager — メインループ
============================
ポーリング方式（デフォルト10秒ごと）でOANDAのオープントレードを監視し、
出口ルールを適用する。

処理順序（毎サイクル）:
  1. OANDA からオープントレード一覧を取得
  2. TradeRegistry と照合（未登録の trade_id を自動登録）
  3. Kill Switch チェック（日次/週次 realized+unrealized）
  4. 各トレードについて:
     a. max_loss_guard  (Priority 3, ティックベース)
     b. lockout_short_term (60分ロックアウト)
     c. lockout_time_filter (銘柄別時間フィルター、non_textbook)
     d. tp1             (Priority 4)
     e. giveback_stop   (Priority 5)
     f. trailing_stop   (Priority 6, 4H足確定時のみ)
     g. reversal_exit   (Priority 7, 4H足確定時のみ)
     h. silver_time_stop (Priority 8, XAG_USD only)
     i. anti_patterns   (Priority 9, veto層)
  5. アクションを実行（OANDA API）
  6. ログ出力 + Discord通知
  7. sleep(poll_interval_sec)

4H足確定タイミング: UTC 0:00, 4:00, 8:00, 12:00, 16:00, 20:00

Usage:
    python -m exit_manager.main
    # または
    python exit_manager/main.py
"""

import logging
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from exit_manager.exit_rules import (
    Action,
    calc_breakeven_sl,
    check_anti_patterns,
    check_giveback_stop,
    check_kill_switch,
    check_lockout_short_term,
    check_lockout_time_filter,
    check_max_loss_guard,
    check_reversal_exit,
    check_silver_time_stop,
    check_tp1,
    check_trailing_stop,
    check_volatility_regime,
)
from exit_manager.logger import ExitManagerLogger
from exit_manager.notifier import DiscordNotifier
from exit_manager.oanda_client import ExitManagerClient
from exit_manager.position_manager import TradePhase, TradeRegistry, TradeState

log = logging.getLogger('sena3fx.exit_manager')

# UTC の 4H 足確定時刻（時間）
H4_CLOSE_HOURS = {0, 4, 8, 12, 16, 20}


class ExitManager:
    """
    Exit Manager のメインクラス。

    Usage:
        cfg = load_config()
        mgr = ExitManager(cfg)
        mgr.run()
    """

    def __init__(self, config: dict):
        self.config = config
        self.client = ExitManagerClient.from_env()
        self.registry = TradeRegistry()
        self.logger = ExitManagerLogger(
            log_dir=config['logging']['output_dir']
        )
        self.notifier = DiscordNotifier(config)
        self.running = True

        # 4H足確定追跡（同一時間で2回発火しないよう管理）
        self._last_4h_close_hour: int = -1

        # 日次損益追跡（実現損益のみ、JPY）
        self._daily_realized_jpy: float = 0.0
        self._daily_reset_date: str = ''

        # Kill Switch 状態
        self._kill_switch_active: bool = False

        # ボラティリティ・レジーム状態（non_textbook）
        self._vol_regime: dict = {
            'is_high_vol': False,
            'effective_risk_jpy': float(config['account']['max_loss_jpy']),
            'effective_tp1_r': float(config['exit_rules']['tp1']['r_multiple']),
            'max_concurrent': int(config['account']['max_concurrent_trades']),
            'lockout_minutes': 60,
        }
        self._last_daily_candle_date: str = ''

        # instrument → jpy_per_dollar_per_unit のキャッシュ
        self._jpy_rates: dict = {
            inst: float(cfg.get('jpy_per_dollar_per_unit', 151.8))
            for inst, cfg in config.get('instruments', {}).items()
        }

        signal.signal(signal.SIGINT, self._on_shutdown)
        signal.signal(signal.SIGTERM, self._on_shutdown)

    def _on_shutdown(self, signum, frame):
        log.info('シャットダウン信号受信。ループを終了します。')
        self.running = False

    # ------------------------------------------------------------------ #
    #  メインループ                                                        #
    # ------------------------------------------------------------------ #

    def run(self):
        poll_interval = int(self.config['monitoring']['poll_interval_sec'])
        log.info(f'Exit Manager 起動。poll_interval={poll_interval}秒')

        while self.running:
            try:
                self._cycle()
            except Exception as e:
                log.error(f'サイクルエラー: {e}')
                log.debug(traceback.format_exc())
            time.sleep(poll_interval)

        log.info('Exit Manager 停止。')

    def _cycle(self):
        now = datetime.utcnow()

        # 日次リセット（UTC 00:00）
        today_str = now.strftime('%Y-%m-%d')
        if today_str != self._daily_reset_date:
            self._daily_realized_jpy = 0.0
            self._daily_reset_date = today_str
            log.info(f'日次P&Lリセット: {today_str}')

            # 2.5 ボラティリティ・レジーム判定（日次1回）
            if today_str != self._last_daily_candle_date:
                self._update_volatility_regime()
                self._last_daily_candle_date = today_str

        # 1. オープントレード取得
        try:
            open_trades = self.client.get_open_trades()
        except Exception as e:
            log.warning(f'オープントレード取得失敗: {e}')
            return

        # 2. レジストリ照合（再起動後の自動登録）
        self._reconcile(open_trades, now)

        # 3. Kill Switch チェック
        unrealized_jpy = self._compute_total_unrealized_jpy(open_trades)
        kill_action = check_kill_switch(
            realized_pnl_jpy=self._daily_realized_jpy,
            unrealized_pnl_jpy=unrealized_jpy,
            config=self.config,
            period='daily',
        )
        if kill_action and not self._kill_switch_active:
            log.warning(f'Kill Switch 発動: {kill_action.reason}')
            self._kill_switch_active = True
            self.logger.log_kill_switch(
                scope='daily',
                total_pnl_jpy=self._daily_realized_jpy + unrealized_jpy,
                threshold_jpy=self.config['exit_rules']['kill_switch']['daily']['max_loss_jpy'],
                block_new_entries=True,
            )
        elif not kill_action:
            self._kill_switch_active = False

        # 4H 足確定チェック
        is_4h_close = self._detect_4h_close(now)

        # 4H キャンドルキャッシュ（足確定時のみ取得）
        candles_cache: dict = {}

        # 4. 各アクティブトレードを処理
        for trade in self.registry.get_active_trades():
            try:
                self._process_trade(trade, now, is_4h_close, candles_cache)
            except Exception as e:
                log.error(f'[{trade.trade_id}] トレード処理エラー: {e}')
                log.debug(traceback.format_exc())

    # ------------------------------------------------------------------ #
    #  4H 足確定検知                                                      #
    # ------------------------------------------------------------------ #

    def _detect_4h_close(self, now: datetime) -> bool:
        """
        現在時刻が 4H 足確定時刻（UTC 0/4/8/12/16/20時）かつ
        同一時間での発火が初回かどうかを判定する。
        """
        h = now.hour
        if h in H4_CLOSE_HOURS and h != self._last_4h_close_hour:
            self._last_4h_close_hour = h
            return True
        if h not in H4_CLOSE_HOURS:
            # 次の確定時間に向けてリセット
            self._last_4h_close_hour = -1
        return False

    # ------------------------------------------------------------------ #
    #  個別トレード処理                                                    #
    # ------------------------------------------------------------------ #

    def _process_trade(
        self,
        trade: TradeState,
        now: datetime,
        is_4h_close: bool,
        candles_cache: dict,
    ):
        jpy_rate = self._jpy_rates.get(trade.instrument, 151.8)

        # 現在価格取得
        try:
            bid, ask = self.client.get_current_price(trade.instrument)
        except Exception as e:
            log.warning(f'[{trade.trade_id}] 価格取得失敗: {e}')
            return

        current_price = ask if trade.side == 'long' else bid
        current_r = trade.unrealized_r(current_price)

        # ピーク含み益 R を更新（Giveback Stop 用）
        if current_r > trade.peak_unrealized_r:
            self.registry.update(trade.trade_id, peak_unrealized_r=current_r)

        # --- Priority 3: max_loss_guard（ティックベース、最高優先度） ---
        action = check_max_loss_guard(trade, current_price, self.config, jpy_rate)
        if action:
            self._execute_action(action, trade, current_price, jpy_rate)
            return

        # --- 60分ロックアウト（SL以外の決済をブロック） ---
        if check_lockout_short_term(trade, now):
            elapsed_min = (now - trade.entry_time).total_seconds() / 60.0
            remaining_min = 60.0 - elapsed_min
            self.logger.log_lockout_blocked(
                trade_id=trade.trade_id,
                attempted_action='exit_rules',
                lockout_remaining_min=remaining_min,
                message=f'ロックアウト中: あと{remaining_min:.0f}分は手動決済できません',
                non_textbook=True,
            )
            return

        # --- 銘柄別時間フィルター（non_textbook） ---
        if check_lockout_time_filter(trade, now, self.config):
            elapsed_h = (now - trade.entry_time).total_seconds() / 3600.0
            self.logger.log_lockout_blocked(
                trade_id=trade.trade_id,
                attempted_action='exit_rules',
                lockout_remaining_min=max(0, (8.0 - elapsed_h) * 60.0),
                message=f'時間フィルター中: 金8h未満は利確禁止',
                non_textbook=True,
            )
            return

        # --- Priority 4: TP1（50%利確 + SL建値移動） ---
        action = check_tp1(trade, current_price, self.config)
        if action:
            blocked, reason = check_anti_patterns(action, trade, self.config)
            if blocked:
                self.logger.log_validation_warning(
                    trade_id=trade.trade_id,
                    warning=reason,
                    blocked_action='tp1_partial_close',
                )
            else:
                self._execute_action(action, trade, current_price, jpy_rate)
                # TP1後にSLを建値+バッファへ移動
                if self.config['exit_rules']['tp1'].get('move_sl_to_breakeven', True):
                    self._move_sl_to_breakeven(trade)
                return

        # --- Priority 5: Giveback Stop ---
        action = check_giveback_stop(trade, current_r, self.config)
        if action:
            blocked, reason = check_anti_patterns(action, trade, self.config)
            if not blocked:
                self._execute_action(action, trade, current_price, jpy_rate)
                return

        # --- Priority 6,7: 4H 足確定時のみ ---
        if is_4h_close:
            if trade.instrument not in candles_cache:
                try:
                    candles_cache[trade.instrument] = self.client.get_candles(
                        trade.instrument, 'H4', count=20
                    )
                except Exception as e:
                    log.warning(f'[{trade.trade_id}] 4Hキャンドル取得失敗: {e}')
                    candles_cache[trade.instrument] = None

            candles_4h = candles_cache.get(trade.instrument)

            # Priority 6: Trailing Stop
            action = check_trailing_stop(trade, candles_4h, self.config, is_4h_close=True)
            if action:
                blocked, reason = check_anti_patterns(action, trade, self.config)
                if not blocked:
                    self._execute_action(action, trade, current_price, jpy_rate)
                    return
                else:
                    self.logger.log_validation_warning(
                        trade_id=trade.trade_id, warning=reason,
                        blocked_action='trailing_stop',
                    )

            # Priority 7: Reversal Exit
            action = check_reversal_exit(trade, candles_4h, self.config, is_4h_close=True)
            if action:
                blocked, reason = check_anti_patterns(action, trade, self.config)
                if not blocked:
                    self._execute_action(action, trade, current_price, jpy_rate)
                    return

        # --- Priority 8: Silver Time Stop（XAG_USD専用） ---
        action = check_silver_time_stop(trade, now, current_r, self.config)
        if action:
            blocked, reason = check_anti_patterns(action, trade, self.config)
            if not blocked:
                self._execute_action(action, trade, current_price, jpy_rate)

    # ------------------------------------------------------------------ #
    #  アクション実行                                                      #
    # ------------------------------------------------------------------ #

    def _execute_action(
        self,
        action: Action,
        trade: TradeState,
        current_price: float,
        jpy_rate: float,
    ):
        """アクションを OANDA API で実行し、ログ・通知する。"""
        try:
            if action.action_type == 'CLOSE_ALL':
                self._do_close_all(action, trade, current_price, jpy_rate)

            elif action.action_type == 'PARTIAL_CLOSE':
                self._do_partial_close(action, trade, current_price, jpy_rate)

            elif action.action_type == 'MODIFY_SL':
                self._do_modify_sl(action, trade)

        except Exception as e:
            log.error(
                f'[{trade.trade_id}] アクション実行失敗 '
                f'({action.action_type}): {e}'
            )
            log.debug(traceback.format_exc())

    def _do_close_all(
        self,
        action: Action,
        trade: TradeState,
        current_price: float,
        jpy_rate: float,
    ):
        pnl_jpy = trade.unrealized_pnl_jpy(current_price, jpy_rate)
        self.client.close_position(trade.instrument, trade.side)
        self._daily_realized_jpy += pnl_jpy
        hold_h = (datetime.utcnow() - trade.entry_time).total_seconds() / 3600.0
        self.registry.mark_closed(trade.trade_id)

        self.logger.log_trade_closed(
            trade_id=trade.trade_id,
            price=current_price,
            reason=action.reason,
            pnl_jpy=pnl_jpy,
            hold_hours=hold_h,
            instrument=trade.instrument,
            non_textbook=action.non_textbook,
        )
        self.notifier.notify(
            'TRADE_CLOSED',
            trade_id=trade.trade_id,
            instrument=trade.instrument,
            pnl_jpy=pnl_jpy,
            reason=action.reason,
        )
        log.info(
            f'[{trade.trade_id}] 全決済: {trade.instrument} {trade.side} '
            f'pnl=¥{pnl_jpy:,.0f} reason={action.reason}'
        )

    def _do_partial_close(
        self,
        action: Action,
        trade: TradeState,
        current_price: float,
        jpy_rate: float,
    ):
        units_to_close = action.units or (trade.current_units // 2)
        self.client.close_trade_partial(trade.trade_id, units_to_close)

        # 部分利確のP&L計算（近似）
        if trade.side == 'long':
            pnl_jpy = (current_price - trade.entry_price) * units_to_close * jpy_rate
        else:
            pnl_jpy = (trade.entry_price - current_price) * units_to_close * jpy_rate

        self._daily_realized_jpy += pnl_jpy
        new_units = trade.current_units - units_to_close

        self.registry.update(
            trade.trade_id,
            current_units=new_units,
            tp1_executed=True,
            phase=TradePhase.TP1_HIT,
        )

        self.logger.log_tp1_hit(
            trade_id=trade.trade_id,
            price=current_price,
            partial_close_units=units_to_close,
            remaining_units=new_units,
            realized_pnl_jpy=pnl_jpy,
            new_sl=trade.sl_price,
            new_sl_reason='TP1後に建値移動予定',
            instrument=trade.instrument,
        )
        self.notifier.notify(
            'TP1_HIT',
            trade_id=trade.trade_id,
            instrument=trade.instrument,
            price=current_price,
            partial_close_units=units_to_close,
            pnl_jpy=pnl_jpy,
        )
        log.info(
            f'[{trade.trade_id}] TP1: {units_to_close}units利確 '
            f'残{new_units}units pnl=¥{pnl_jpy:,.0f}'
        )

    def _do_modify_sl(self, action: Action, trade: TradeState):
        self.client.modify_trade(trade.trade_id, sl_price=action.new_sl)
        old_sl = trade.sl_price
        self.registry.update(
            trade.trade_id,
            sl_price=action.new_sl,
            phase=TradePhase.TRAILING,
        )
        self.logger.log_trailing_update(
            trade_id=trade.trade_id,
            old_sl=old_sl,
            new_sl=action.new_sl,
            reason=action.reason,
            remaining_units=trade.current_units,
            instrument=trade.instrument,
        )
        log.info(
            f'[{trade.trade_id}] SL更新: {old_sl:.3f} → {action.new_sl:.3f} '
            f'({action.reason})'
        )

    def _move_sl_to_breakeven(self, trade: TradeState):
        """TP1後にSLを建値+バッファへ移動する。"""
        new_sl = calc_breakeven_sl(trade, self.config)
        try:
            self.client.modify_trade(trade.trade_id, sl_price=new_sl)
            old_sl = trade.sl_price
            self.registry.update(trade.trade_id, sl_price=new_sl)
            self.logger.log(
                'SL_MODIFIED',
                trade_id=trade.trade_id,
                instrument=trade.instrument,
                old_sl=old_sl,
                new_sl=new_sl,
                reason='breakeven_after_tp1',
            )
            log.info(f'[{trade.trade_id}] 建値SL移動: {old_sl:.3f} → {new_sl:.3f}')
        except Exception as e:
            log.error(f'[{trade.trade_id}] 建値SL移動失敗: {e}')

    # ------------------------------------------------------------------ #
    #  ユーティリティ                                                      #
    # ------------------------------------------------------------------ #

    def _reconcile(self, open_trades: list, now: datetime):
        """
        OANDAのオープントレードとレジストリを照合。
        レジストリにない trade_id を OPEN フェーズで自動登録する。
        （再起動後に既存ポジションを継続管理するため）
        """
        known_ids = {t.trade_id for t in self.registry.get_active_trades()}

        for ot in open_trades:
            tid = str(ot.get('id', ''))
            if not tid or tid in known_ids:
                continue

            instrument = ot.get('instrument', '')
            raw_units = int(ot.get('currentUnits', 0))
            side = 'long' if raw_units > 0 else 'short'
            entry_price = float(ot.get('price', 0))
            sl_order = ot.get('stopLossOrder', {})
            sl_price = float(sl_order.get('price', 0)) if sl_order else 0.0
            sl_dist = abs(entry_price - sl_price) if sl_price else 1.0

            trade = TradeState(
                trade_id=tid,
                instrument=instrument,
                side=side,
                entry_price=entry_price,
                entry_time=now,
                sl_price=sl_price,
                original_units=abs(raw_units),
                current_units=abs(raw_units),
                sl_distance_usd=sl_dist,
                one_r_jpy=float(self.config['account']['max_loss_jpy']),
                phase=TradePhase.OPEN,
                notes='auto_reconciled_on_restart',
            )
            self.registry.register(trade)
            log.info(
                f'再起動後の自動登録: {tid} ({instrument} {side}) '
                f'entry={entry_price:.3f} sl={sl_price:.3f}'
            )

    def _compute_total_unrealized_jpy(self, open_trades: list) -> float:
        """全オープントレードの含み損益合計（JPY）を計算する。"""
        total = 0.0
        for ot in open_trades:
            inst = ot.get('instrument', '')
            jpy_rate = self._jpy_rates.get(inst, 151.8)
            unrealized_usd = float(ot.get('unrealizedPL', 0.0))
            total += unrealized_usd * jpy_rate
        return total

    def _update_volatility_regime(self):
        """
        日足ATR(14)を取得してボラティリティ・レジームを判定する。
        高ボラ時はリスク・TP1・同時建玉・ロックアウトを調整する（non_textbook）。
        """
        try:
            candles_daily = self.client.get_candles('XAU_USD', 'D', count=35)
        except Exception as e:
            log.warning(f'ボラティリティ判定用日足取得失敗: {e}')
            return

        prev_regime = self._vol_regime.get('is_high_vol', False)
        self._vol_regime = check_volatility_regime(candles_daily, self.config)

        if self._vol_regime['is_high_vol'] != prev_regime:
            new_label = '高ボラ' if self._vol_regime['is_high_vol'] else '通常'
            log.warning(
                f'ボラティリティ・レジーム変化: {new_label} '
                f"(ATR ratio={self._vol_regime['ratio']:.2f})"
            )
            self.logger.log(
                'VOLATILITY_REGIME_CHANGE',
                is_high_vol=self._vol_regime['is_high_vol'],
                current_atr=self._vol_regime['current_atr'],
                avg_atr=self._vol_regime['avg_atr'],
                ratio=self._vol_regime['ratio'],
                effective_risk_jpy=self._vol_regime['effective_risk_jpy'],
                effective_tp1_r=self._vol_regime['effective_tp1_r'],
                max_concurrent=self._vol_regime['max_concurrent'],
                lockout_minutes=self._vol_regime['lockout_minutes'],
                non_textbook=True,
            )


# ------------------------------------------------------------------ #
#  エントリーポイント                                                  #
# ------------------------------------------------------------------ #

def load_config() -> dict:
    cfg_path = Path(__file__).parent / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-5s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    config = load_config()
    mgr = ExitManager(config)
    mgr.run()


if __name__ == '__main__':
    main()
