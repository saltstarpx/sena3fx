"""
Exit Manager テスト — 共通フィクスチャ
========================================
全テストファイルで共有される pytest フィクスチャを定義する。

設計方針:
  - 全テストデータは合成データ（実市場データは使用しない）
  - OANDA API 呼び出しはモックで代替
  - フィクスチャはシンプルで再現性のある値を使用
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exit_manager.position_manager import TradePhase, TradeState


# ------------------------------------------------------------------ #
#  config フィクスチャ                                                #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_config() -> dict:
    """テスト用の設定辞書。"""
    return {
        'account': {
            'balance_jpy': 5_000_000,
            'risk_per_trade_pct': 3.0,
            'max_loss_jpy': 150_000,
            'max_concurrent_trades': 3,
        },
        'instruments': {
            'XAU_USD': {
                'jpy_per_dollar_per_unit': 151.8,
                'min_units': 1,
                'max_units': 500,
                'pip_size': 0.01,
                'spread_typical_usd': 0.30,
                'min_sl_distance': 1.0,
                'max_sl_distance': 50.0,
                'min_hold_hours': 8,
            },
            'XAG_USD': {
                'jpy_per_dollar_per_unit': 1554.9,
                'min_units': 1,
                'max_units': 5000,
                'pip_size': 0.001,
                'spread_typical_usd': 0.02,
                'min_sl_distance': 0.05,
                'max_sl_distance': 5.0,
            },
        },
        'exit_rules': {
            'initial_sl': {
                'required': True,
                'placement': 'structural',
                'min_sl_atr_ratio': 0.5,
                'max_sl_atr_ratio': 3.0,
            },
            'max_loss_guard': {
                'enabled': True,
                'threshold_jpy': 150_000,
                'check_interval_sec': 5,
            },
            'tp1': {
                'r_multiple': 1.0,
                'partial_close_pct': 50,
                'move_sl_to_breakeven': True,
                'breakeven_buffer_pct': 0.1,
            },
            'trailing': {
                'method': 'swing_4h',
                'atr_multiplier': 2.0,
                'update_on': 'candle_close',
                'min_remaining_pct': 25,
            },
            'reversal_exit': {
                'enabled': True,
                'timeframe': 'H4',
            },
            'giveback_stop': {
                'enabled': True,
                'trigger_r': 2.0,
                'exit_r': 1.0,
            },
            'lockout': {
                'short_term': {
                    'enabled': True,
                    'duration_minutes': 60,
                },
                'time_filter': {
                    'enabled': True,
                    'non_textbook': True,
                    'XAU_USD': {
                        'min_hold_hours': 8,
                        'time_stop_enabled': False,
                        'time_stop_threshold_r': -0.5,
                    },
                    'XAG_USD': {
                        'max_hold_hours_if_flat': 24,
                        'flat_threshold_r': 1.0,
                    },
                },
                'allow_sl_hit': True,
                'allow_manual_override': False,
                'emergency_password': None,
            },
            'kill_switch': {
                'enabled': True,
                'pnl_calculation': 'realized_plus_unrealized',
                'daily': {
                    'max_loss_r': 2.0,
                    'max_loss_jpy': 300_000,
                    'reset_time_utc': '00:00',
                },
                'weekly': {
                    'max_loss_r': 4.0,
                    'max_loss_jpy': 600_000,
                    'reset_day': 'monday',
                },
            },
            'anti_patterns': {
                'no_manual_sl_widen': True,
                'no_unplanned_averaging': True,
                'no_repeated_partial': True,
            },
        },
        'logging': {
            'format': 'jsonl',
            'output_dir': './logs',
        },
        'notification': {
            'enabled': False,
            'discord_webhook_url': None,
            'notify_on': ['TP1_HIT', 'MAX_LOSS_GUARD', 'TRADE_CLOSED', 'LOCKOUT_BLOCKED'],
        },
        'monitoring': {
            'poll_interval_sec': 10,
            'candle_check_interval_sec': 60,
            'health_check_interval_sec': 300,
        },
    }


# ------------------------------------------------------------------ #
#  TradeState ファクトリー                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def make_trade():
    """TradeState を作成するファクトリー関数を返すフィクスチャ。"""

    def _factory(
        trade_id: str = 'T001',
        instrument: str = 'XAU_USD',
        side: str = 'long',
        entry_price: float = 2000.0,
        sl_price: float = 1990.0,
        tp_price: float = None,
        current_units: int = 100,
        entry_time: datetime = None,
        phase: TradePhase = TradePhase.OPEN,
        tp1_executed: bool = False,
        peak_unrealized_r: float = 0.0,
        non_textbook: bool = False,
        **kwargs,
    ) -> TradeState:
        if entry_time is None:
            entry_time = datetime(2026, 3, 1, 10, 0, 0)

        sl_distance = abs(entry_price - sl_price)

        return TradeState(
            trade_id=trade_id,
            instrument=instrument,
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            sl_price=sl_price,
            tp_price=tp_price,
            original_units=current_units,
            current_units=current_units,
            sl_distance_usd=sl_distance,
            one_r_jpy=150_000.0,
            phase=phase,
            tp1_executed=tp1_executed,
            peak_unrealized_r=peak_unrealized_r,
            non_textbook=non_textbook,
            **kwargs,
        )

    return _factory


# ------------------------------------------------------------------ #
#  合成 OHLCV データ                                                  #
# ------------------------------------------------------------------ #

@pytest.fixture
def make_candles_4h():
    """合成 4H OHLCV データを作成するファクトリー関数を返すフィクスチャ。"""

    def _factory(
        n: int = 20,
        base_price: float = 2000.0,
        trend: float = 0.5,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        合成 4H キャンドルデータを生成する。
        実市場データではない。

        Args:
            n:          バー本数
            base_price: 基準価格
            trend:      トレンド方向（正=上昇, 負=下降）
            seed:       乱数シード（再現性のため）
        """
        rng = np.random.default_rng(seed)
        closes = base_price + np.cumsum(rng.normal(trend, 2.0, n))
        highs = closes + rng.uniform(1.0, 5.0, n)
        lows = closes - rng.uniform(1.0, 5.0, n)
        opens = closes - rng.normal(0, 1.0, n)

        idx = [
            datetime(2026, 1, 1) + timedelta(hours=4 * i)
            for i in range(n)
        ]
        return pd.DataFrame(
            {
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': rng.integers(50, 200, n),
            },
            index=idx,
        )

    return _factory


# ------------------------------------------------------------------ #
#  時刻フィクスチャ                                                   #
# ------------------------------------------------------------------ #

@pytest.fixture
def now_utc() -> datetime:
    """固定UTC時刻（テストの再現性のため）。"""
    return datetime(2026, 3, 4, 12, 0, 0)  # UTC 12:00（4H足確定時刻）
