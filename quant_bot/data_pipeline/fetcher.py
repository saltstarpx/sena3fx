"""
OANDA REST API データ取得ラッパー。

scripts/fetch_data.py の関数をクラスベースでラップ。
APIキーは環境変数 OANDA_API_KEY から取得（コード内直書き禁止）。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# プロジェクトルートを sys.path に追加して scripts/ を import 可能にする
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_data import fetch_oanda_candles, fetch_oanda_long  # noqa: E402

log = logging.getLogger(__name__)


class OandaFetcher:
    """
    OANDA v20 REST API からローソク足データを取得するラッパー。

    全メソッドの戻り値:
      pd.DataFrame (DatetimeIndex UTC-naive, columns: open/high/low/close/volume/spread)
      または None（APIエラー時）
    """

    def __init__(self, config: dict):
        """
        Args:
            config: data_pipeline/config.yaml の 'oanda' セクション dict

        Raises:
            EnvironmentError: OANDA_API_KEY が設定されていない場合
        """
        self._api_key = os.environ.get("OANDA_API_KEY") or config.get("api_key", "")
        if not self._api_key:
            raise EnvironmentError(
                "OANDA_API_KEY 環境変数が設定されていません。\n"
                "実行前に以下のコマンドで設定してください:\n"
                "  export OANDA_API_KEY='your-api-key'"
            )

        env_override = os.environ.get("OANDA_ENVIRONMENT", "")
        self._environment = env_override or config.get("environment", "practice")
        log.info(f"OandaFetcher 初期化 (environment={self._environment})")

    def fetch_recent(
        self,
        instrument: str,
        granularity: str,
        count: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        直近 count 本の確定バーを取得。

        Args:
            instrument:  OANDA形式 'XAU_USD'
            granularity: OANDA形式 'H4', 'M15', 'D' 等
            count:       取得本数（最大 5000）

        Returns:
            pd.DataFrame または None
        """
        log.info(f"fetch_recent: {instrument} {granularity} {count}本")
        return fetch_oanda_candles(
            instrument=instrument,
            granularity=granularity,
            count=count,
            api_key=self._api_key,
            account_type=self._environment,
        )

    def fetch_history(
        self,
        instrument: str,
        granularity: str,
        days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """
        長期履歴データを取得（内部でチャンク分割）。

        Args:
            instrument:  OANDA形式 'XAU_USD'
            granularity: OANDA形式 'H4' 等
            days:        取得日数

        Returns:
            pd.DataFrame または None
        """
        log.info(f"fetch_history: {instrument} {granularity} {days}日分")
        return fetch_oanda_long(
            instrument=instrument,
            granularity=granularity,
            days=days,
            api_key=self._api_key,
            account_type=self._environment,
        )
