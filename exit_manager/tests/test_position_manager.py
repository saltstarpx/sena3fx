"""
exit_manager/position_manager.py のテスト

TradeState のデータクラスと TradeRegistry の動作をテストする。
"""

from datetime import datetime

import pytest

from exit_manager.position_manager import TradePhase, TradeRegistry, TradeState


class TestTradeState:

    def test_unrealized_r_long_at_1r(self, make_trade):
        """LONG: SL距離分上昇 → +1R"""
        trade = make_trade(entry_price=2000.0, sl_price=1990.0)
        assert trade.unrealized_r(2010.0) == pytest.approx(1.0)

    def test_unrealized_r_long_at_2r(self, make_trade):
        """LONG: 2倍SL距離 → +2R"""
        trade = make_trade(entry_price=2000.0, sl_price=1990.0)
        assert trade.unrealized_r(2020.0) == pytest.approx(2.0)

    def test_unrealized_r_long_at_minus_half(self, make_trade):
        """LONG: 半分下落 → -0.5R"""
        trade = make_trade(entry_price=2000.0, sl_price=1990.0)
        assert trade.unrealized_r(1995.0) == pytest.approx(-0.5)

    def test_unrealized_r_short_at_1r(self, make_trade):
        """SHORT: SL距離分下落 → +1R"""
        trade = make_trade(side='short', entry_price=2000.0, sl_price=2010.0)
        assert trade.unrealized_r(1990.0) == pytest.approx(1.0)

    def test_unrealized_r_zero_sl_distance(self):
        """SL距離=0 → 0.0（ZeroDivisionError なし）"""
        trade = TradeState(
            trade_id='T1', instrument='XAU_USD', side='long',
            entry_price=2000.0, entry_time=datetime.utcnow(),
            sl_price=2000.0, sl_distance_usd=0.0,
        )
        assert trade.unrealized_r(2010.0) == 0.0

    def test_unrealized_pnl_jpy_long_win(self, make_trade):
        """LONG勝ち: P&L = units × price_move × jpy_rate"""
        trade = make_trade(entry_price=2000.0, current_units=100)
        # 10 USD × 100 units × 151.8 = 151,800 JPY
        pnl = trade.unrealized_pnl_jpy(2010.0, 151.8)
        assert pnl == pytest.approx(151_800.0, abs=1.0)

    def test_unrealized_pnl_jpy_short_win(self, make_trade):
        """SHORT勝ち: 価格下落で正のP&L"""
        trade = make_trade(side='short', entry_price=2000.0, current_units=100)
        pnl = trade.unrealized_pnl_jpy(1990.0, 151.8)
        assert pnl == pytest.approx(151_800.0, abs=1.0)

    def test_unrealized_pnl_jpy_loss(self, make_trade):
        """LONG負け: 負のP&L"""
        trade = make_trade(entry_price=2000.0, current_units=100)
        pnl = trade.unrealized_pnl_jpy(1995.0, 151.8)
        assert pnl < 0

    def test_default_phase_is_registered(self):
        """デフォルトフェーズは REGISTERED"""
        trade = TradeState(
            trade_id='T1', instrument='XAU_USD', side='long',
            entry_price=2000.0, entry_time=datetime.utcnow(),
            sl_price=1990.0,
        )
        assert trade.phase == TradePhase.REGISTERED

    def test_sl_distance_r(self, make_trade):
        """sl_distance_r: 価格とSLの距離をR倍率で表現"""
        trade = make_trade(entry_price=2000.0, sl_price=1990.0)
        # 現在価格2000.0 と SL 1990.0 の距離 = 10.0 → 10/10 = 1.0R
        assert trade.sl_distance_r(2000.0) == pytest.approx(1.0)


class TestTradeRegistry:

    def test_register_and_get(self, make_trade):
        """登録したトレードを取得できる"""
        reg = TradeRegistry()
        trade = make_trade(trade_id='T001')
        reg.register(trade)
        assert reg.get('T001') is trade

    def test_get_returns_none_for_unknown(self):
        """存在しないIDは None"""
        reg = TradeRegistry()
        assert reg.get('UNKNOWN') is None

    def test_get_active_excludes_closed(self, make_trade):
        """CLOSED は active に含まれない"""
        reg = TradeRegistry()
        for i, phase in enumerate([
            TradePhase.OPEN, TradePhase.CLOSED, TradePhase.TP1_HIT, TradePhase.TRAILING
        ]):
            t = make_trade(trade_id=f'T{i}', phase=phase)
            reg.register(t)

        active = reg.get_active_trades()
        assert len(active) == 3  # OPEN, TP1_HIT, TRAILING（CLOSEDを除く）
        assert all(t.phase != TradePhase.CLOSED for t in active)

    def test_get_all_trades_includes_closed(self, make_trade):
        """get_all_trades は CLOSED も含む"""
        reg = TradeRegistry()
        for i, phase in enumerate([TradePhase.OPEN, TradePhase.CLOSED]):
            t = make_trade(trade_id=f'T{i}', phase=phase)
            reg.register(t)
        assert len(reg.get_all_trades()) == 2

    def test_update_modifies_fields(self, make_trade):
        """update() でフィールドを更新できる"""
        reg = TradeRegistry()
        trade = make_trade(trade_id='T001', sl_price=1990.0)
        reg.register(trade)

        updated = reg.update('T001', sl_price=1995.0, phase=TradePhase.TRAILING)
        assert updated.sl_price == 1995.0
        assert updated.phase == TradePhase.TRAILING
        assert reg.get('T001').sl_price == 1995.0

    def test_update_returns_none_for_unknown(self):
        """存在しないIDのupdate → None"""
        reg = TradeRegistry()
        result = reg.update('UNKNOWN', sl_price=1990.0)
        assert result is None

    def test_mark_closed_sets_closed_phase(self, make_trade):
        """mark_closed() で CLOSED フェーズになる"""
        reg = TradeRegistry()
        trade = make_trade(trade_id='T001', phase=TradePhase.OPEN)
        reg.register(trade)
        reg.mark_closed('T001')
        assert reg.get('T001').phase == TradePhase.CLOSED

    def test_mark_closed_unknown_id_noop(self):
        """存在しないIDの mark_closed → エラーなし"""
        reg = TradeRegistry()
        reg.mark_closed('UNKNOWN')  # 例外が出ないこと

    def test_register_overwrites_existing(self, make_trade):
        """同一IDで再登録すると上書きされる"""
        reg = TradeRegistry()
        trade1 = make_trade(trade_id='T001', sl_price=1990.0)
        trade2 = make_trade(trade_id='T001', sl_price=1995.0)
        reg.register(trade1)
        reg.register(trade2)
        assert reg.get('T001').sl_price == 1995.0

    def test_kill_switch_state(self):
        """Kill Switch 状態の get/set"""
        reg = TradeRegistry()
        assert reg.is_kill_switch_active() is False
        reg.set_kill_switch(True)
        assert reg.is_kill_switch_active() is True
        reg.set_kill_switch(False)
        assert reg.is_kill_switch_active() is False

    def test_multiple_trades(self, make_trade):
        """複数トレードを管理できる"""
        reg = TradeRegistry()
        for i in range(5):
            reg.register(make_trade(trade_id=f'T{i:03d}'))
        assert len(reg.get_active_trades()) == 5
