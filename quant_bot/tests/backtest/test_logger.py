"""
backtest/logger.py テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from quant_bot.backtest.logger import TradeEventLogger, TradeEvent


class TestTradeEventLogger:

    @pytest.fixture
    def logger(self, tmp_path):
        """一時ファイルに書き込む TradeEventLogger。"""
        return TradeEventLogger(tmp_path / "test_trades.jsonl")

    def test_log_entry_creates_event(self, logger):
        """log_entry() が TradeEvent を返すことを確認。"""
        event = logger.log_entry(
            timestamp="2024-01-01T08:00:00",
            instrument="XAU_USD",
            direction="LONG",
            price=2000.0,
            lots=0.5,
            sl=1990.0,
            tp=2020.0,
            rr=2.0,
            rule_ids=["C1", "C2", "C3"],
        )
        assert isinstance(event, TradeEvent)
        assert event.event == "ENTRY"
        assert event.direction == "LONG"
        assert event.price == 2000.0

    def test_log_exit_creates_event(self, logger):
        """log_exit() が TradeEvent を返すことを確認。"""
        event = logger.log_exit(
            timestamp="2024-01-02T08:00:00",
            instrument="XAU_USD",
            direction="LONG",
            price=2020.0,
            lots=0.5,
            pnl=500.0,
            exit_reason="tp_hit",
        )
        assert event.event == "EXIT"
        assert event.pnl == 500.0
        assert event.exit_reason == "tp_hit"

    def test_jsonl_file_written(self, tmp_path):
        """イベントが JSONL ファイルに書き出されることを確認。"""
        path = tmp_path / "trades.jsonl"
        logger = TradeEventLogger(path)

        logger.log_entry(
            timestamp="2024-01-01T08:00:00",
            instrument="XAU_USD",
            direction="LONG",
            price=2000.0,
            lots=0.5,
        )

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["event"] == "ENTRY"
        assert record["instrument"] == "XAU_USD"
        assert record["price"] == 2000.0

    def test_jsonl_format_is_valid_json(self, tmp_path):
        """JSONL の各行が有効な JSON であることを確認。"""
        path = tmp_path / "trades.jsonl"
        logger = TradeEventLogger(path)

        for i in range(5):
            logger.log_entry(
                timestamp=f"2024-01-0{i+1}T08:00:00",
                instrument="XAU_USD",
                direction="LONG",
                price=2000.0 + i,
                lots=0.1,
            )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            record = json.loads(line)  # 例外なくパースできることを確認
            assert "event" in record
            assert "timestamp" in record

    def test_ingest_backtest_trades(self, tmp_path):
        """ingest_backtest_trades() が正しく取込むことを確認。"""
        path = tmp_path / "bt_trades.jsonl"
        logger = TradeEventLogger(path)

        # バックテストエンジンが返す形式の trades リスト
        trades = [
            {
                "direction": "LONG",
                "entry_time": "2024-01-01",
                "exit_time": "2024-01-03",
                "entry_price": 2000.0,
                "exit_price": 2030.0,
                "sl": 1990.0,
                "tp": 2030.0,
                "size": 0.5,
                "pnl": 750.0,
                "exit_reason": "tp_hit",
            },
        ]

        count = logger.ingest_backtest_trades(trades, "XAU_USD")
        assert count == 1

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # ENTRY + EXIT

    def test_non_textbook_flag(self, logger):
        """non_textbook フラグが正しく記録されることを確認。"""
        event = logger.log_entry(
            timestamp="2024-01-01T08:00:00",
            instrument="XAU_USD",
            direction="LONG",
            price=2000.0,
            lots=0.5,
            non_textbook=True,
        )
        assert event.non_textbook is True
