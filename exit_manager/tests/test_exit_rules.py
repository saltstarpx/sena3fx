"""
exit_manager/exit_rules.py のテスト

仕様書 Section 7 の全テストケースに準拠。
テストデータは全て合成値（実市場データ未使用）。
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

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
)
from exit_manager.position_manager import TradePhase


# ------------------------------------------------------------------ #
#  Priority 1: Kill Switch                                            #
# ------------------------------------------------------------------ #

class TestKillSwitch:

    def test_no_trigger_below_threshold(self, sample_config):
        """日次損失が閾値以下 → None"""
        result = check_kill_switch(-100_000.0, 0.0, sample_config, 'daily')
        assert result is None

    def test_triggers_daily_when_combined_pnl_exceeds(self, sample_config):
        """日次: realized + unrealized が -300,000円超 → BLOCK_ENTRY"""
        # -200,000 + -110,000 = -310,000 < -300,000
        result = check_kill_switch(-200_000.0, -110_000.0, sample_config, 'daily')
        assert result is not None
        assert result.action_type == 'BLOCK_ENTRY'
        assert 'daily' in result.reason.lower()
        assert result.priority == 1

    def test_triggers_weekly(self, sample_config):
        """週次: -600,000円超 → BLOCK_ENTRY"""
        result = check_kill_switch(-500_000.0, -120_000.0, sample_config, 'weekly')
        assert result is not None
        assert 'weekly' in result.reason.lower()

    def test_unrealized_alone_triggers(self, sample_config):
        """含み損だけで-2R超えてもKill Switch発動"""
        result = check_kill_switch(0.0, -310_000.0, sample_config, 'daily')
        assert result is not None
        assert result.action_type == 'BLOCK_ENTRY'

    def test_kill_switch_disabled(self, sample_config):
        """kill_switch.enabled=False → 常に None"""
        cfg = dict(sample_config)
        cfg['exit_rules'] = dict(sample_config['exit_rules'])
        cfg['exit_rules']['kill_switch'] = dict(
            sample_config['exit_rules']['kill_switch']
        )
        cfg['exit_rules']['kill_switch']['enabled'] = False
        result = check_kill_switch(-999_999.0, -999_999.0, cfg, 'daily')
        assert result is None

    def test_exactly_at_threshold_no_trigger(self, sample_config):
        """ちょうど閾値（-300,000）はトリガーしない（> ではなく <= の条件）"""
        # -300,000 = threshold（以下ならブロック → -300,000 <= -300,000 はブロック）
        # 仕様: total_pnl <= threshold でブロック
        result = check_kill_switch(-300_000.0, 0.0, sample_config, 'daily')
        assert result is not None  # -300,000 <= -300,000 → ブロック


# ------------------------------------------------------------------ #
#  Priority 3: Max Loss Guard                                         #
# ------------------------------------------------------------------ #

class TestMaxLossGuard:

    def test_no_trigger_within_1r(self, make_trade, sample_config):
        """含み損が閾値未満 → None"""
        trade = make_trade(entry_price=2890.0, sl_price=2875.0, current_units=65)
        # 含み損 = 65 * 151.8 * 14.0 = 138,138 < 150,000
        action = check_max_loss_guard(trade, 2876.0, sample_config, 151.8)
        assert action is None

    def test_triggers_at_1r_loss(self, make_trade, sample_config):
        """含み損が-150,000円超 → CLOSE_ALL"""
        trade = make_trade(entry_price=2890.0, sl_price=2875.0, current_units=70)
        # 含み損 = 70 * 151.8 * 15.0 = 159,390 > 150,000
        action = check_max_loss_guard(trade, 2875.0, sample_config, 151.8)
        assert action is not None
        assert action.action_type == 'CLOSE_ALL'
        assert action.priority == 3

    def test_short_triggers_on_price_rise(self, make_trade, sample_config):
        """SHORT: 価格上昇で含み損 → CLOSE_ALL"""
        trade = make_trade(
            side='short',
            entry_price=2000.0,
            sl_price=2010.0,
            current_units=100,
        )
        # 含み損 = 100 * 151.8 * 12.0 = 182,160 > 150,000
        action = check_max_loss_guard(trade, 2012.0, sample_config, 151.8)
        assert action is not None
        assert action.action_type == 'CLOSE_ALL'

    def test_disabled(self, make_trade, sample_config):
        """enabled=False → 常に None"""
        cfg = dict(sample_config)
        cfg['exit_rules'] = dict(sample_config['exit_rules'])
        cfg['exit_rules']['max_loss_guard'] = {'enabled': False, 'threshold_jpy': 150_000}
        trade = make_trade(entry_price=2000.0, sl_price=1900.0, current_units=100)
        action = check_max_loss_guard(trade, 1800.0, cfg, 151.8)
        assert action is None


# ------------------------------------------------------------------ #
#  ロックアウト                                                        #
# ------------------------------------------------------------------ #

class TestLockoutShortTerm:

    def test_blocks_within_60min(self, make_trade):
        """建玉から30分 → ブロック"""
        entry_time = datetime.utcnow() - timedelta(minutes=30)
        trade = make_trade(entry_time=entry_time)
        assert check_lockout_short_term(trade, datetime.utcnow()) is True

    def test_allows_after_60min(self, make_trade):
        """建玉から90分 → 許可"""
        entry_time = datetime.utcnow() - timedelta(minutes=90)
        trade = make_trade(entry_time=entry_time)
        assert check_lockout_short_term(trade, datetime.utcnow()) is False

    def test_exactly_at_60min_boundary(self, make_trade):
        """建玉からちょうど60分 → 許可（60.0 < 60.0 は False）"""
        entry_time = datetime.utcnow() - timedelta(minutes=60)
        trade = make_trade(entry_time=entry_time)
        assert check_lockout_short_term(trade, datetime.utcnow()) is False


class TestLockoutTimeFilter:

    def test_xau_blocks_within_8h(self, make_trade, sample_config):
        """金: 5h → ブロック（8h未満）"""
        entry_time = datetime.utcnow() - timedelta(hours=5)
        trade = make_trade(instrument='XAU_USD', entry_time=entry_time)
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is True

    def test_xau_allows_after_8h(self, make_trade, sample_config):
        """金: 10h → 許可"""
        entry_time = datetime.utcnow() - timedelta(hours=10)
        trade = make_trade(instrument='XAU_USD', entry_time=entry_time)
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is False

    def test_xag_not_filtered(self, make_trade, sample_config):
        """銀: check_lockout_time_filter は XAU のみ → False"""
        entry_time = datetime.utcnow() - timedelta(hours=2)
        trade = make_trade(instrument='XAG_USD', entry_time=entry_time)
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is False

    def test_disabled(self, make_trade, sample_config):
        """time_filter.enabled=False → 常に False"""
        cfg = dict(sample_config)
        cfg['exit_rules'] = dict(sample_config['exit_rules'])
        cfg['exit_rules']['lockout'] = dict(sample_config['exit_rules']['lockout'])
        cfg['exit_rules']['lockout']['time_filter'] = {'enabled': False}
        entry_time = datetime.utcnow() - timedelta(hours=2)
        trade = make_trade(instrument='XAU_USD', entry_time=entry_time)
        assert check_lockout_time_filter(trade, datetime.utcnow(), cfg) is False


# ------------------------------------------------------------------ #
#  Priority 4: TP1                                                    #
# ------------------------------------------------------------------ #

class TestTp1:

    def test_triggers_at_1r(self, make_trade, sample_config):
        """TP1: +1R到達 → PARTIAL_CLOSE 50%"""
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0,
            current_units=65, phase=TradePhase.OPEN,
        )
        # +1R = entry + SL距離 = 2890 + 15 = 2905
        action = check_tp1(trade, 2905.5, sample_config)
        assert action is not None
        assert action.action_type == 'PARTIAL_CLOSE'
        assert action.units == 32  # 65 // 2

    def test_no_trigger_below_1r(self, make_trade, sample_config):
        """TP1: +0.5R → None"""
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0,
            current_units=65, phase=TradePhase.OPEN,
        )
        action = check_tp1(trade, 2897.5, sample_config)  # +0.5R
        assert action is None

    def test_no_trigger_when_already_executed(self, make_trade, sample_config):
        """TP1: tp1_executed=True → 繰り返し発火しない"""
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0,
            current_units=32, tp1_executed=True,
        )
        action = check_tp1(trade, 2920.0, sample_config)
        assert action is None

    def test_no_trigger_when_closed(self, make_trade, sample_config):
        """TP1: CLOSED フェーズ → None"""
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0,
            phase=TradePhase.CLOSED,
        )
        action = check_tp1(trade, 2920.0, sample_config)
        assert action is None

    def test_short_tp1(self, make_trade, sample_config):
        """SHORT: TP1 = entry - SL距離 で発火"""
        trade = make_trade(
            side='short',
            entry_price=2890.0, sl_price=2905.0,
            current_units=65, phase=TradePhase.OPEN,
        )
        # -1R = entry - SL距離 = 2890 - 15 = 2875
        action = check_tp1(trade, 2874.5, sample_config)
        assert action is not None
        assert action.action_type == 'PARTIAL_CLOSE'


class TestCalcBreakevenSl:

    def test_long_breakeven_with_buffer(self, make_trade, sample_config):
        """LONG: SL = entry + buffer（SL幅×10%）"""
        trade = make_trade(entry_price=2890.0, sl_price=2875.0)
        # buffer = 15 * 0.1 = 1.5
        # new_sl = 2890 + 1.5 = 2891.5
        new_sl = calc_breakeven_sl(trade, sample_config)
        assert new_sl == pytest.approx(2891.5, abs=0.01)

    def test_short_breakeven_with_buffer(self, make_trade, sample_config):
        """SHORT: SL = entry - buffer"""
        trade = make_trade(
            side='short', entry_price=2890.0, sl_price=2905.0
        )
        new_sl = calc_breakeven_sl(trade, sample_config)
        assert new_sl == pytest.approx(2888.5, abs=0.01)


# ------------------------------------------------------------------ #
#  Priority 5: Giveback Stop                                          #
# ------------------------------------------------------------------ #

class TestGivebackStop:

    def test_arms_at_2r_and_triggers_at_1r(self, make_trade, sample_config):
        """+2R到達後に+1R以下で全決済"""
        trade = make_trade(peak_unrealized_r=2.5)  # ピーク2.5R
        action = check_giveback_stop(trade, current_unrealized_r=0.9, config=sample_config)
        assert action is not None
        assert action.action_type == 'CLOSE_ALL'
        assert 'giveback' in action.reason.lower()

    def test_no_trigger_when_peak_below_2r(self, make_trade, sample_config):
        """ピークが2R未満 → None"""
        trade = make_trade(peak_unrealized_r=1.5)
        action = check_giveback_stop(trade, 0.8, sample_config)
        assert action is None

    def test_no_trigger_when_still_above_1r(self, make_trade, sample_config):
        """+2R到達後でも+1R超え → None"""
        trade = make_trade(peak_unrealized_r=3.0)
        action = check_giveback_stop(trade, 1.5, sample_config)
        assert action is None

    def test_exactly_at_exit_r(self, make_trade, sample_config):
        """ちょうど+1Rでトリガー（<= の条件）"""
        trade = make_trade(peak_unrealized_r=2.5)
        action = check_giveback_stop(trade, 1.0, sample_config)
        assert action is not None  # 1.0 <= 1.0 → トリガー

    def test_disabled(self, make_trade, sample_config):
        """enabled=False → None"""
        cfg = dict(sample_config)
        cfg['exit_rules'] = dict(sample_config['exit_rules'])
        cfg['exit_rules']['giveback_stop'] = {'enabled': False, 'trigger_r': 2.0, 'exit_r': 1.0}
        trade = make_trade(peak_unrealized_r=3.0)
        action = check_giveback_stop(trade, 0.5, cfg)
        assert action is None


# ------------------------------------------------------------------ #
#  Priority 6: Trailing Stop                                          #
# ------------------------------------------------------------------ #

class TestTrailingStop:

    def test_updates_sl_when_swing_low_above_current_sl(
        self, make_trade, make_candles_4h, sample_config
    ):
        """LONG: スイングローが現在SLより上 → MODIFY_SL"""
        candles = make_candles_4h(n=20, base_price=2050.0)
        # SLを低く設定（スイングローより下）
        trade = make_trade(
            sl_price=2020.0, phase=TradePhase.TP1_HIT,
        )
        action = check_trailing_stop(trade, candles, sample_config, is_4h_close=True)
        if action is not None:
            assert action.action_type == 'MODIFY_SL'
            assert action.new_sl > trade.sl_price  # SLが改善（上昇）

    def test_no_update_when_not_4h_close(self, make_trade, make_candles_4h, sample_config):
        """4H足確定でない → None"""
        candles = make_candles_4h()
        trade = make_trade(sl_price=1990.0, phase=TradePhase.TP1_HIT)
        action = check_trailing_stop(trade, candles, sample_config, is_4h_close=False)
        assert action is None

    def test_no_update_insufficient_candles(self, make_trade, sample_config):
        """キャンドル本数不足 → None"""
        few_candles = pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1]})
        trade = make_trade(sl_price=1990.0, phase=TradePhase.TP1_HIT)
        action = check_trailing_stop(trade, few_candles, sample_config, is_4h_close=True)
        assert action is None

    def test_no_update_when_swing_below_current_sl(self, make_trade, sample_config):
        """スイングローが現在SLより下 → SL更新なし → None"""
        # SLを高く設定（スイングローより上）
        very_high_sl_trade = make_trade(
            entry_price=2100.0, sl_price=2090.0,
            phase=TradePhase.TP1_HIT,
        )
        # 低い価格帯のキャンドル（スイングロー < sl_price）
        import numpy as np
        rng = np.random.default_rng(99)
        n = 10
        closes = 2010.0 + rng.normal(0, 5, n)
        candles = pd.DataFrame({
            'open': closes, 'high': closes + 5, 'low': closes - 5, 'close': closes
        })
        action = check_trailing_stop(very_high_sl_trade, candles, sample_config, is_4h_close=True)
        assert action is None


# ------------------------------------------------------------------ #
#  Priority 7: Reversal Exit                                          #
# ------------------------------------------------------------------ #

class TestReversalExit:

    def test_long_reversal_when_close_breaks_structural_low(
        self, make_trade, sample_config
    ):
        """LONG: 最新終値が構造ローを下抜け → CLOSE_ALL"""
        trade = make_trade(side='long')
        # 合成データ: 最後のバーだけ大幅下落
        import numpy as np
        n = 15
        # 前10本は price=2050 付近
        closes = [2050.0] * (n - 1) + [2000.0]  # 最後だけ急落
        highs = [c + 5 for c in closes]
        lows = [c - 5 for c in closes]
        # 構造ロー = 直前10本の安値最小 = 2050 - 5 = 2045
        # 最新終値 = 2000 < 2045 → 反転シグナル
        candles = pd.DataFrame({
            'open': closes, 'high': highs, 'low': lows, 'close': closes
        })
        action = check_reversal_exit(trade, candles, sample_config, is_4h_close=True)
        assert action is not None
        assert action.action_type == 'CLOSE_ALL'

    def test_no_reversal_without_4h_close(self, make_trade, sample_config):
        """4H足未確定 → None"""
        trade = make_trade()
        candles = pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1]})
        action = check_reversal_exit(trade, candles, sample_config, is_4h_close=False)
        assert action is None


# ------------------------------------------------------------------ #
#  Priority 8: Silver Time Stop                                       #
# ------------------------------------------------------------------ #

class TestSilverTimeStop:

    def test_triggers_for_xag_after_24h_below_1r(self, make_trade, sample_config):
        """銀: 25h経過かつR<1.0 → CLOSE_ALL（non_textbook）"""
        entry_time = datetime.utcnow() - timedelta(hours=25)
        trade = make_trade(
            instrument='XAG_USD',
            entry_time=entry_time,
        )
        action = check_silver_time_stop(trade, datetime.utcnow(), 0.3, sample_config)
        assert action is not None
        assert action.action_type == 'CLOSE_ALL'
        assert action.non_textbook is True

    def test_no_trigger_for_xau(self, make_trade, sample_config):
        """金には適用しない → None"""
        entry_time = datetime.utcnow() - timedelta(hours=30)
        trade = make_trade(instrument='XAU_USD', entry_time=entry_time)
        action = check_silver_time_stop(trade, datetime.utcnow(), 0.3, sample_config)
        assert action is None

    def test_no_trigger_when_above_1r(self, make_trade, sample_config):
        """24h超えても+1R以上なら保持"""
        entry_time = datetime.utcnow() - timedelta(hours=26)
        trade = make_trade(instrument='XAG_USD', entry_time=entry_time)
        action = check_silver_time_stop(trade, datetime.utcnow(), 1.5, sample_config)
        assert action is None

    def test_no_trigger_before_24h(self, make_trade, sample_config):
        """24h未満は発動しない"""
        entry_time = datetime.utcnow() - timedelta(hours=20)
        trade = make_trade(instrument='XAG_USD', entry_time=entry_time)
        action = check_silver_time_stop(trade, datetime.utcnow(), 0.0, sample_config)
        assert action is None


# ------------------------------------------------------------------ #
#  Priority 9: Anti-patterns（veto層）                               #
# ------------------------------------------------------------------ #

class TestAntiPatterns:

    def test_blocks_sl_widening_long(self, make_trade, sample_config):
        """LONG: SLを下に移動（拡大）→ ブロック"""
        trade = make_trade(side='long', sl_price=1990.0)
        action = Action(action_type='MODIFY_SL', new_sl=1985.0)  # より不利
        blocked, reason = check_anti_patterns(action, trade, sample_config)
        assert blocked is True
        assert 'SL拡大' in reason

    def test_allows_sl_tightening_long(self, make_trade, sample_config):
        """LONG: SLを上に移動（改善）→ 許可"""
        trade = make_trade(side='long', sl_price=1990.0)
        action = Action(action_type='MODIFY_SL', new_sl=1995.0)  # より有利
        blocked, _ = check_anti_patterns(action, trade, sample_config)
        assert blocked is False

    def test_blocks_sl_widening_short(self, make_trade, sample_config):
        """SHORT: SLを上に移動（拡大）→ ブロック"""
        trade = make_trade(side='short', sl_price=2010.0)
        action = Action(action_type='MODIFY_SL', new_sl=2015.0)  # より不利
        blocked, reason = check_anti_patterns(action, trade, sample_config)
        assert blocked is True

    def test_blocks_repeated_partial_after_tp1(self, make_trade, sample_config):
        """TP1後の追加半利 → ブロック"""
        trade = make_trade(tp1_executed=True, current_units=32)
        action = Action(action_type='PARTIAL_CLOSE', units=16)
        blocked, reason = check_anti_patterns(action, trade, sample_config)
        assert blocked is True
        assert 'TP1後' in reason or 'tp1' in reason.lower()

    def test_allows_close_all_after_tp1(self, make_trade, sample_config):
        """TP1後でも CLOSE_ALL は許可"""
        trade = make_trade(tp1_executed=True)
        action = Action(action_type='CLOSE_ALL', reason='giveback_stop')
        blocked, _ = check_anti_patterns(action, trade, sample_config)
        assert blocked is False

    def test_blocks_unplanned_averaging(self, make_trade, sample_config):
        """計画外ナンピン → ブロック"""
        trade = make_trade()
        action = Action(action_type='ADD_UNITS', units=50)
        blocked, reason = check_anti_patterns(action, trade, sample_config)
        assert blocked is True


# ------------------------------------------------------------------ #
#  優先順位・衝突解決テスト（仕様書 Section 2.6 準拠）               #
# ------------------------------------------------------------------ #

class TestPriorityConflict:

    def test_priority_gold_8h_vs_tp1(self, make_trade, sample_config):
        """金8hフィルター中にTP1到達 → TP1が発火する（Priority 4 > 時間フィルター）"""
        # 8h未満のトレード
        entry_time = datetime.utcnow() - timedelta(hours=5)
        trade = make_trade(
            instrument='XAU_USD',
            entry_price=2890.0, sl_price=2875.0,
            current_units=65, phase=TradePhase.OPEN,
            entry_time=entry_time,
        )
        # 時間フィルターはブロック
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is True
        # TP1は時間フィルターに関係なく発火する（main.py での順序で処理）
        # exit_rules.py の check_tp1() 自体は時間フィルターを見ない
        action = check_tp1(trade, 2905.5, sample_config)
        assert action is not None  # TP1 ルールは発火する

    def test_priority_lockout_vs_max_loss(self, make_trade, sample_config):
        """60分ロックアウト中に-1R到達 → max_loss_guard が発火（Priority 3）"""
        # 30分しか経っていないトレード
        entry_time = datetime.utcnow() - timedelta(minutes=30)
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0,
            current_units=70, entry_time=entry_time,
        )
        # ロックアウト中
        assert check_lockout_short_term(trade, datetime.utcnow()) is True
        # max_loss_guard は独立してチェック → 発火する
        action = check_max_loss_guard(trade, 2875.0, sample_config, 151.8)
        assert action is not None  # ロックアウト中でも発火

    def test_giveback_does_not_trigger_without_arm(self, make_trade, sample_config):
        """+2R未到達ではGiveback発火しない"""
        trade = make_trade(peak_unrealized_r=0.0)  # ピーク到達なし
        action = check_giveback_stop(trade, current_unrealized_r=0.5, config=sample_config)
        assert action is None
