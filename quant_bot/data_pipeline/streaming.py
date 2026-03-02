"""
OANDA ストリーミング価格フィードスケルトン。

Phase 3-4（全自動）実装時に完成予定。
現在は connect() が NotImplementedError を返す。
バーベースのポーリングには OandaFetcher.fetch_recent() を使用すること。
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

log = logging.getLogger(__name__)


class StreamingPriceFeed:
    """
    OANDA v20 ストリーミング API スケルトン。

    将来の実装インターフェース:
      connect()      → ストリーミング接続を開始
      get_tick()     → 次の (instrument, bid, ask, timestamp) を返す
      stop()         → 接続を閉じる

    現在はバックテストモードで未使用。
    リアルタイムデータには OandaFetcher.fetch_recent() でポーリングすること。
    """

    def __init__(
        self,
        instruments: list[str],
        api_key: str,
        account_id: str,
        environment: str = "practice",
    ):
        self._instruments = instruments
        self._api_key = api_key
        self._account_id = account_id
        self._environment = environment
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> None:
        """ストリーミング接続を開始する（未実装）。"""
        raise NotImplementedError(
            "StreamingPriceFeed.connect() は未実装です。\n"
            "バーベースのポーリングには OandaFetcher.fetch_recent() を使用してください。"
        )

    def get_tick(self, timeout: float = 1.0) -> Optional[tuple]:
        """
        次のティックを取得。

        Returns:
            (instrument, bid, ask, timestamp) または timeout 時は None
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """ストリーミング接続を閉じる。"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        log.info("StreamingPriceFeed 停止")
