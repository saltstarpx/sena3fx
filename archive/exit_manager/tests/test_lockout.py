"""
exit_manager のロックアウト機能テスト

60分ロックアウトと銘柄別時間フィルターの動作を詳細にテスト。
仕様書 Section 7 の lockout 関連ケースに準拠。
"""

from datetime import datetime, timedelta

import pytest

from exit_manager.exit_rules import (
    check_lockout_short_term,
    check_lockout_time_filter,
    check_silver_time_stop,
    check_max_loss_guard,
    check_tp1,
)
from exit_manager.position_manager import TradePhase


class TestShortTermLockout60min:
    """60分ロックアウトの境界条件テスト"""

    def test_blocked_at_0min(self, make_trade):
        """建玉直後（0分）→ ブロック"""
        trade = make_trade(entry_time=datetime.utcnow())
        assert check_lockout_short_term(trade, datetime.utcnow()) is True

    def test_blocked_at_30min(self, make_trade):
        """30分後 → ブロック"""
        trade = make_trade(entry_time=datetime.utcnow() - timedelta(minutes=30))
        assert check_lockout_short_term(trade, datetime.utcnow()) is True

    def test_blocked_at_59min(self, make_trade):
        """59分後 → ブロック"""
        trade = make_trade(entry_time=datetime.utcnow() - timedelta(minutes=59))
        assert check_lockout_short_term(trade, datetime.utcnow()) is True

    def test_allowed_at_60min(self, make_trade):
        """ちょうど60分 → 許可（60.0 < 60.0 は False）"""
        trade = make_trade(entry_time=datetime.utcnow() - timedelta(minutes=60))
        assert check_lockout_short_term(trade, datetime.utcnow()) is False

    def test_allowed_at_61min(self, make_trade):
        """61分後 → 許可"""
        trade = make_trade(entry_time=datetime.utcnow() - timedelta(minutes=61))
        assert check_lockout_short_term(trade, datetime.utcnow()) is False

    def test_allowed_at_2h(self, make_trade):
        """2時間後 → 許可"""
        trade = make_trade(entry_time=datetime.utcnow() - timedelta(hours=2))
        assert check_lockout_short_term(trade, datetime.utcnow()) is False


class TestGoldTimeFilter8h:
    """金の8時間時間フィルター（non_textbook）"""

    def test_blocked_at_1h(self, make_trade, sample_config):
        """1h → ブロック"""
        trade = make_trade(
            instrument='XAU_USD',
            entry_time=datetime.utcnow() - timedelta(hours=1),
        )
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is True

    def test_blocked_at_7h59m(self, make_trade, sample_config):
        """7h59m → ブロック"""
        trade = make_trade(
            instrument='XAU_USD',
            entry_time=datetime.utcnow() - timedelta(hours=7, minutes=59),
        )
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is True

    def test_allowed_at_8h(self, make_trade, sample_config):
        """8h → 許可"""
        trade = make_trade(
            instrument='XAU_USD',
            entry_time=datetime.utcnow() - timedelta(hours=8),
        )
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is False

    def test_allowed_at_24h(self, make_trade, sample_config):
        """24h → 許可"""
        trade = make_trade(
            instrument='XAU_USD',
            entry_time=datetime.utcnow() - timedelta(hours=24),
        )
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is False

    def test_silver_not_filtered_by_time_filter(self, make_trade, sample_config):
        """銀には XAU_USD の時間フィルターを適用しない"""
        trade = make_trade(
            instrument='XAG_USD',
            entry_time=datetime.utcnow() - timedelta(hours=3),
        )
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is False


class TestSilverTimeFilter24h:
    """銀の24時間フィルター（non_textbook）"""

    def test_blocked_at_25h_below_1r(self, make_trade, sample_config):
        """25h経過 + R<1.0 → CLOSE_ALL"""
        trade = make_trade(
            instrument='XAG_USD',
            entry_time=datetime.utcnow() - timedelta(hours=25),
        )
        action = check_silver_time_stop(trade, datetime.utcnow(), 0.5, sample_config)
        assert action is not None
        assert action.action_type == 'CLOSE_ALL'
        assert action.non_textbook is True

    def test_not_blocked_at_23h(self, make_trade, sample_config):
        """23h → ブロックしない"""
        trade = make_trade(
            instrument='XAG_USD',
            entry_time=datetime.utcnow() - timedelta(hours=23),
        )
        action = check_silver_time_stop(trade, datetime.utcnow(), 0.5, sample_config)
        assert action is None

    def test_not_blocked_when_above_1r(self, make_trade, sample_config):
        """25h経過でも+1.5R → ブロックしない（伸びているので保持）"""
        trade = make_trade(
            instrument='XAG_USD',
            entry_time=datetime.utcnow() - timedelta(hours=25),
        )
        action = check_silver_time_stop(trade, datetime.utcnow(), 1.5, sample_config)
        assert action is None

    def test_not_applied_to_gold(self, make_trade, sample_config):
        """金には銀の時間フィルターを適用しない"""
        trade = make_trade(
            instrument='XAU_USD',
            entry_time=datetime.utcnow() - timedelta(hours=30),
        )
        action = check_silver_time_stop(trade, datetime.utcnow(), 0.0, sample_config)
        assert action is None


class TestLockoutPriorityConflicts:
    """仕様書 Section 2.6 の例外マトリクス準拠"""

    def test_max_loss_guard_overrides_lockout(self, make_trade, sample_config):
        """
        60分ロックアウト中に-1R到達 → Max Loss Guard が発火（Priority 3）
        ロックアウトは Max Loss Guard をブロックしない
        （main.py の処理順: max_loss_guard を先にチェック）
        """
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0, current_units=70,
            entry_time=datetime.utcnow() - timedelta(minutes=30),
        )
        # ロックアウト中
        assert check_lockout_short_term(trade, datetime.utcnow()) is True

        # max_loss_guard は exit_rules.py レベルでは独立 → 発火する
        action = check_max_loss_guard(trade, 2875.0, sample_config, 151.8)
        assert action is not None  # Priority 3 が発火

    def test_tp1_fires_even_within_lockout_rule_level(self, make_trade, sample_config):
        """
        TP1 ルール自体はロックアウトを考慮しない
        （main.py がロックアウトを先にチェックして return する）
        check_tp1() 単体テストとして、TP1 条件が満たされれば発火する
        """
        trade = make_trade(
            entry_price=2890.0, sl_price=2875.0,
            current_units=65, phase=TradePhase.OPEN,
            entry_time=datetime.utcnow() - timedelta(minutes=30),
        )
        # TP1条件を満たす価格
        action = check_tp1(trade, 2905.5, sample_config)
        assert action is not None  # ルール自体は発火する（main.py の順序が優先度を制御）

    def test_gold_8h_filter_allows_max_loss_guard(self, make_trade, sample_config):
        """
        金8hフィルター中でも Max Loss Guard は発火する
        check_lockout_time_filter は exit rules レベルでは Max Loss Guard に影響しない
        """
        trade = make_trade(
            instrument='XAU_USD',
            entry_price=2890.0, sl_price=2875.0, current_units=70,
            entry_time=datetime.utcnow() - timedelta(hours=5),
        )
        # 時間フィルターはブロック
        assert check_lockout_time_filter(trade, datetime.utcnow(), sample_config) is True

        # Max Loss Guard は独立 → 発火する
        action = check_max_loss_guard(trade, 2875.0, sample_config, 151.8)
        assert action is not None
