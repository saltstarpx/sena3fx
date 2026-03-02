"""
バックテストエンジン — lib/backtest.BacktestEngine ラッパー。

lib/backtest.BacktestEngine を直接再利用し、
JSONL トレードイベントロギングを追加する。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# lib/ への参照
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.backtest import BacktestEngine  # noqa: E402

from .logger import TradeEventLogger  # noqa: E402

log = logging.getLogger("quant_bot.backtest")


class QuantBacktestEngine:
    """
    lib/backtest.BacktestEngine ラッパー。

    既存エンジンをそのまま使用し、実行後に JSONL ロギングを追加。
    """

    def __init__(self, config: dict):
        """
        Args:
            config: backtest/config.yaml の全 dict
        """
        bt_cfg = config.get("backtest", {})
        log_cfg = config.get("logging", {})

        self._initial_balance = float(bt_cfg.get("initial_balance", 10_000.0))
        self._risk_pct = float(bt_cfg.get("risk_pct", 2.0))
        self._sl_atr_mult = float(bt_cfg.get("sl_atr_mult", 2.0))
        self._tp_atr_mult = float(bt_cfg.get("tp_atr_mult", 4.0))

        self._jsonl_dir = Path(log_cfg.get("jsonl_dir", "trade_logs"))

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        strategy_fn,
        instrument: str,
        timeframe: str = "H4",
        output_path: Optional[str | Path] = None,
        rule_ids: Optional[list[str]] = None,
        non_textbook: bool = False,
    ) -> dict:
        """
        バックテストを実行し JSONL ログを出力。

        Args:
            ohlcv_df:     OHLCV DataFrame (index=datetime)
            strategy_fn:  シグナル生成関数。シグネチャ:
                          fn(df: pd.DataFrame) -> pd.Series (values: 'LONG'/'SHORT'/'FLAT')
            instrument:   'XAU_USD' 等
            timeframe:    'H4' / 'M15' 等
            output_path:  JSONL 出力ファイルパス (省略時は自動生成)
            rule_ids:     適用ルールIDリスト
            non_textbook: 教材外ルール使用フラグ

        Returns:
            {
                'trades': list[dict],
                'summary': dict,
                'jsonl_path': Path,
                'trade_count': int,
            }
        """
        # lib/backtest.BacktestEngine でシグナル生成・取引シミュレーション
        engine = BacktestEngine(
            initial_balance=self._initial_balance,
            risk_pct=self._risk_pct,
            sl_atr_mult=self._sl_atr_mult,
            tp_atr_mult=self._tp_atr_mult,
        )

        signal_series = strategy_fn(ohlcv_df)
        result = engine.run(ohlcv_df, signal_series)
        trades = result.get("trades", [])

        # JSONL ログ出力パスの決定
        if output_path is None:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{instrument}_{timeframe}_{ts}.jsonl"
            output_path = self._jsonl_dir / fname

        output_path = Path(output_path)

        # JSONL 取込み
        trade_logger = TradeEventLogger(output_path)
        count = trade_logger.ingest_backtest_trades(
            trades=trades,
            instrument=instrument,
            rule_ids=rule_ids or [],
            non_textbook=non_textbook,
            initial_balance=self._initial_balance,
            risk_pct=self._risk_pct,
        )

        log.info(
            f"バックテスト完了: {instrument} {timeframe}, "
            f"trades={count}, jsonl={output_path}"
        )

        return {
            "trades": trades,
            "summary": result.get("summary", {}),
            "jsonl_path": output_path,
            "trade_count": count,
        }

    def run_with_signals(
        self,
        ohlcv_df: pd.DataFrame,
        signal_series: pd.Series,
        instrument: str,
        timeframe: str = "H4",
        output_path: Optional[str | Path] = None,
        rule_ids: Optional[list[str]] = None,
        non_textbook: bool = False,
    ) -> dict:
        """
        シグナルシリーズを直接渡してバックテストを実行。

        signal_series: pd.Series (index=datetime, values='LONG'/'SHORT'/'FLAT')
        """
        return self.run(
            ohlcv_df=ohlcv_df,
            strategy_fn=lambda _df: signal_series,
            instrument=instrument,
            timeframe=timeframe,
            output_path=output_path,
            rule_ids=rule_ids,
            non_textbook=non_textbook,
        )
