"""
exit_manager/lot_calculator.py のテスト

テストデータは全て合成値（実市場データ未使用）。
仕様書 Section 4.1 のテストケースに準拠。
"""

import pytest
from exit_manager.lot_calculator import (
    calculate_position_size,
    calc_tp1_price,
    calc_unrealized_r,
    calc_pnl_jpy,
    validate_minimum_size,
)


class TestCalculatePositionSize:
    """calculate_position_size() のテスト"""

    def test_gold_normal_sl15(self, sample_config):
        """金: SL幅$15 → 65units, リスク≤¥150,000"""
        result = calculate_position_size(
            symbol='XAU_USD',
            entry_price=2890.50,
            invalidation_price=2875.50,
            config=sample_config,
        )
        assert not result['rejected']
        assert result['units'] == 65
        assert result['risk_jpy'] <= 150_000
        assert result['sl_distance_usd'] == pytest.approx(15.0, abs=0.001)
        assert result['rr_at_tp1'] == 1.0

    def test_gold_wide_sl30(self, sample_config):
        """金: SL幅$30 → ロット半減（約32units）"""
        result = calculate_position_size(
            symbol='XAU_USD',
            entry_price=2890.50,
            invalidation_price=2860.50,
            config=sample_config,
        )
        assert not result['rejected']
        assert result['units'] == 32
        assert result['risk_jpy'] <= 150_000

    def test_gold_too_wide_sl_rejected(self, sample_config):
        """金: SL幅$100超 → min_units未満で拒否"""
        result = calculate_position_size(
            symbol='XAU_USD',
            entry_price=2890.50,
            invalidation_price=2790.50,
            config=sample_config,
        )
        assert result['rejected']
        assert 'reason' in result

    def test_silver_normal_sl080(self, sample_config):
        """銀: SL幅$0.80 → 約120units"""
        result = calculate_position_size(
            symbol='XAG_USD',
            entry_price=32.50,
            invalidation_price=31.70,
            config=sample_config,
        )
        assert not result['rejected']
        assert result['units'] == 120
        assert result['risk_jpy'] <= 150_000

    def test_zero_sl_distance_raises(self, sample_config):
        """SL距離=0 → ValueError"""
        with pytest.raises(ValueError, match='SL距離が0以下'):
            calculate_position_size(
                symbol='XAU_USD',
                entry_price=2000.0,
                invalidation_price=2000.0,
                config=sample_config,
            )

    def test_negative_sl_distance_raises(self, sample_config):
        """entry = invalidation (distance=0) → ValueError"""
        with pytest.raises(ValueError):
            calculate_position_size(
                symbol='XAU_USD',
                entry_price=2000.0,
                invalidation_price=2000.0,
                config=sample_config,
            )

    def test_risk_jpy_formula(self, sample_config):
        """リスク計算式: units × jpy_per_unit × sl_distance"""
        result = calculate_position_size(
            symbol='XAU_USD',
            entry_price=2890.50,
            invalidation_price=2875.50,
            config=sample_config,
        )
        factor = sample_config['instruments']['XAU_USD']['jpy_per_dollar_per_unit']
        expected_risk = result['units'] * factor * result['sl_distance_usd']
        assert result['risk_jpy'] == pytest.approx(expected_risk, abs=1.0)

    def test_max_units_cap(self, sample_config):
        """max_units を超える場合はキャップされる"""
        # 非常に狭いSL幅 → 大量のユニット → max_units でキャップ
        small_sl_config = dict(sample_config)
        small_sl_config['instruments'] = dict(sample_config['instruments'])
        small_sl_config['instruments']['XAU_USD'] = dict(
            sample_config['instruments']['XAU_USD']
        )
        small_sl_config['instruments']['XAU_USD']['max_units'] = 10  # 極端に小さいキャップ

        result = calculate_position_size(
            symbol='XAU_USD',
            entry_price=2890.50,
            invalidation_price=2889.50,  # $1幅
            config=small_sl_config,
        )
        assert not result['rejected']
        assert result['units'] <= 10  # max_unitsでキャップ

    def test_short_position(self, sample_config):
        """ショート: SL価格がエントリーより高い場合"""
        result = calculate_position_size(
            symbol='XAU_USD',
            entry_price=2890.50,
            invalidation_price=2905.50,  # ショートの場合はSLが上
            config=sample_config,
        )
        assert not result['rejected']
        assert result['sl_distance_usd'] == pytest.approx(15.0, abs=0.001)


class TestCalcTp1Price:
    """calc_tp1_price() のテスト"""

    def test_long_tp1_at_1r(self, sample_config):
        """LONG: TP1 = entry + SL距離"""
        tp1 = calc_tp1_price(2890.50, 2875.50, 'LONG', r_multiple=1.0)
        assert tp1 == pytest.approx(2905.50, abs=0.001)

    def test_short_tp1_at_1r(self, sample_config):
        """SHORT: TP1 = entry - SL距離"""
        tp1 = calc_tp1_price(2890.50, 2905.50, 'SHORT', r_multiple=1.0)
        assert tp1 == pytest.approx(2875.50, abs=0.001)

    def test_tp1_at_2r(self, sample_config):
        """2R倍率: TP1 = entry + 2×SL距離"""
        tp1 = calc_tp1_price(2890.50, 2875.50, 'LONG', r_multiple=2.0)
        assert tp1 == pytest.approx(2920.50, abs=0.001)


class TestCalcUnrealizedR:
    """calc_unrealized_r() のテスト"""

    def test_long_at_1r(self):
        """LONG: 価格がSL距離分上昇 → +1R"""
        r = calc_unrealized_r(
            trade_entry=2000.0, trade_sl=1990.0,
            current_price=2010.0, direction='LONG'
        )
        assert r == pytest.approx(1.0)

    def test_long_at_minus_half_r(self):
        """LONG: 価格が半分下落 → -0.5R"""
        r = calc_unrealized_r(
            trade_entry=2000.0, trade_sl=1990.0,
            current_price=1995.0, direction='LONG'
        )
        assert r == pytest.approx(-0.5)

    def test_short_at_1r(self):
        """SHORT: 価格がSL距離分下落 → +1R"""
        r = calc_unrealized_r(
            trade_entry=2000.0, trade_sl=2010.0,
            current_price=1990.0, direction='SHORT'
        )
        assert r == pytest.approx(1.0)

    def test_zero_sl_distance(self):
        """SL距離=0 → 0.0を返す（ZeroDivisionError なし）"""
        r = calc_unrealized_r(2000.0, 2000.0, 2010.0, 'LONG')
        assert r == 0.0


class TestCalcPnlJpy:
    """calc_pnl_jpy() のテスト"""

    def test_long_win(self, sample_config):
        """LONG勝ち: exit > entry → 正のP&L"""
        pnl = calc_pnl_jpy(
            entry_price=2000.0, exit_price=2010.0,
            units=100, direction='LONG',
            symbol='XAU_USD', config=sample_config,
        )
        # 10 USD * 100 units * 151.8 = 151,800 JPY
        assert pnl == pytest.approx(151_800.0, abs=1.0)

    def test_short_win(self, sample_config):
        """SHORT勝ち: exit < entry → 正のP&L"""
        pnl = calc_pnl_jpy(
            entry_price=2000.0, exit_price=1990.0,
            units=100, direction='SHORT',
            symbol='XAU_USD', config=sample_config,
        )
        assert pnl == pytest.approx(151_800.0, abs=1.0)

    def test_long_loss(self, sample_config):
        """LONG負け: exit < entry → 負のP&L"""
        pnl = calc_pnl_jpy(
            entry_price=2000.0, exit_price=1990.0,
            units=100, direction='LONG',
            symbol='XAU_USD', config=sample_config,
        )
        assert pnl == pytest.approx(-151_800.0, abs=1.0)
