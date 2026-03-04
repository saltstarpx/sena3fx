"""
Exit Manager — OANDA v20 API 拡張クライアント
=============================================
live/broker_oanda.py の OandaBroker を継承し、
Exit Manager に必要な追加メソッドを実装する。

追加メソッド:
  - modify_trade()          SL+TP同時変更
  - close_trade_partial()   部分決済
  - get_trade_details()     個別トレード詳細
  - _request_with_retry()   指数バックオフ付きリトライ

認証:
  APIキー/口座IDは環境変数から取得（コード内直書き禁止）:
    OANDA_API_KEY
    OANDA_ACCOUNT_ID
    OANDA_ENVIRONMENT  ('practice' または 'live', デフォルト: 'practice')
"""

import os
import sys
import time
import logging

import oandapyV20
import oandapyV20.endpoints.trades as v20_trades

# live/ ディレクトリから OandaBroker をインポート
# 実行時は sena3fx/ プロジェクトルートから起動する前提
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from live.broker_oanda import OandaBroker

log = logging.getLogger('sena3fx.exit_manager')


class ExitManagerClient(OandaBroker):
    """
    Exit Manager 用 OANDA API クライアント。

    OandaBroker の全機能を継承し、以下を追加:
      - SL/TP 同時変更 (modify_trade)
      - 部分決済 (close_trade_partial)
      - 個別トレード詳細取得 (get_trade_details)
      - 指数バックオフ付きリトライ (_request_with_retry)

    Usage:
        client = ExitManagerClient.from_env()
        trades = client.get_open_trades()
        client.modify_trade('12345', sl_price=2875.50)
        client.close_trade_partial('12345', units=32)
    """

    def __init__(self, account_id: str, api_key: str,
                 environment: str = 'practice',
                 max_retries: int = 3,
                 timeout_sec: int = 10):
        super().__init__(account_id=account_id, api_key=api_key,
                         environment=environment)
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec

    @classmethod
    def from_env(cls) -> 'ExitManagerClient':
        """
        環境変数から認証情報を読み込んでインスタンスを生成する。

        環境変数:
            OANDA_API_KEY     (必須)
            OANDA_ACCOUNT_ID  (必須)
            OANDA_ENVIRONMENT (任意, デフォルト: 'practice')
        """
        api_key = os.environ.get('OANDA_API_KEY')
        account_id = os.environ.get('OANDA_ACCOUNT_ID')
        environment = os.environ.get('OANDA_ENVIRONMENT', 'practice')

        if not api_key:
            raise EnvironmentError(
                "環境変数 OANDA_API_KEY が設定されていません。"
                "\n  export OANDA_API_KEY='your-api-key'"
            )
        if not account_id:
            raise EnvironmentError(
                "環境変数 OANDA_ACCOUNT_ID が設定されていません。"
                "\n  export OANDA_ACCOUNT_ID='your-account-id'"
            )

        return cls(account_id=account_id, api_key=api_key, environment=environment)

    def _request_with_retry(self, endpoint, max_retries: int = None):
        """
        APIリクエストを指数バックオフ付きでリトライする。

        5xx エラー → 1s, 2s, 4s 待機後にリトライ (最大3回)
        4xx エラー → 即座に例外を再発生 (リトライ不可)

        Args:
            endpoint: oandapyV20 エンドポイントオブジェクト
            max_retries: 最大リトライ回数 (None の場合は self.max_retries を使用)

        Returns:
            endpoint.response dict
        """
        retries = max_retries if max_retries is not None else self.max_retries
        wait = 1

        for attempt in range(retries + 1):
            try:
                self.client.request(endpoint)
                return endpoint.response
            except oandapyV20.exceptions.V20Error as e:
                error_str = str(e)
                # 5xx: サーバーエラー → リトライ可能
                if any(code in error_str for code in ('500', '502', '503', '504')):
                    if attempt < retries:
                        log.warning(
                            f"OANDA APIサーバーエラー (試行 {attempt + 1}/{retries + 1}): "
                            f"{e} → {wait}秒後リトライ"
                        )
                        time.sleep(wait)
                        wait *= 2
                        continue
                # 4xx やその他: 即座に再発生
                raise

        raise RuntimeError(f"OANDA API: {retries}回リトライ後も失敗")

    def modify_trade(self, trade_id: str,
                     sl_price: float = None,
                     tp_price: float = None) -> dict:
        """
        既存トレードのSLとTPを同時に変更する。

        Args:
            trade_id: OANDA トレードID
            sl_price: 新しいSL価格 (None の場合は変更しない)
            tp_price: 新しいTP価格 (None の場合は変更しない)

        Returns:
            OANDA APIレスポンス dict
        """
        data = {}
        if sl_price is not None:
            data['stopLoss'] = {
                'price': f'{sl_price:.3f}',
                'timeInForce': 'GTC',
            }
        if tp_price is not None:
            data['takeProfit'] = {
                'price': f'{tp_price:.3f}',
                'timeInForce': 'GTC',
            }

        if not data:
            log.warning(f"modify_trade: sl_price と tp_price が両方 None (trade_id={trade_id})")
            return {}

        endpoint = v20_trades.TradeCRCDO(
            self.account_id, tradeID=str(trade_id), data=data
        )
        return self._request_with_retry(endpoint)

    def close_trade_partial(self, trade_id: str, units: int) -> dict:
        """
        トレードを部分決済する。

        Args:
            trade_id: OANDA トレードID
            units: 決済するユニット数 (正の整数)

        Returns:
            OANDA APIレスポンス dict
        """
        if units <= 0:
            raise ValueError(f"close_trade_partial: units は正の整数でなければなりません (units={units})")

        data = {'units': str(units)}
        endpoint = v20_trades.TradeClose(
            self.account_id, tradeID=str(trade_id), data=data
        )
        return self._request_with_retry(endpoint)

    def close_trade_full(self, trade_id: str) -> dict:
        """
        トレードを全量決済する。

        Args:
            trade_id: OANDA トレードID

        Returns:
            OANDA APIレスポンス dict
        """
        data = {'units': 'ALL'}
        endpoint = v20_trades.TradeClose(
            self.account_id, tradeID=str(trade_id), data=data
        )
        return self._request_with_retry(endpoint)

    def get_trade_details(self, trade_id: str) -> dict:
        """
        個別トレードの詳細情報を取得する。

        Args:
            trade_id: OANDA トレードID

        Returns:
            トレード詳細 dict (unrealizedPL, currentUnits, openTime, etc.)
        """
        endpoint = v20_trades.TradeDetails(
            self.account_id, tradeID=str(trade_id)
        )
        response = self._request_with_retry(endpoint)
        return response.get('trade', {})

    def get_mid_price(self, instrument: str) -> float:
        """
        現在の中値（bid/askの中間）を取得する。

        Args:
            instrument: 'XAU_USD' | 'XAG_USD'

        Returns:
            mid価格 (float)
        """
        bid, ask = self.get_current_price(instrument)
        return (bid + ask) / 2.0

    def get_account_unrealized_pnl_usd(self) -> float:
        """
        口座全体の含み損益合計（USD）を取得する。

        Returns:
            float: 含み損益合計 (USD)
        """
        summary = self.get_account_summary()
        return float(summary.get('unrealizedPL', 0.0))

    def get_account_realized_pnl_today_usd(self) -> float:
        """
        本日の実現損益合計（USD）を取得する。
        注意: OANDA APIでは直接取得できないため、
              NAV - balance - unrealizedPL で近似値を計算する。

        Returns:
            float: 本日の実現損益近似値 (USD)
        """
        summary = self.get_account_summary()
        balance = float(summary.get('balance', 0.0))
        nav = float(summary.get('NAV', 0.0))
        unrealized = float(summary.get('unrealizedPL', 0.0))
        # 実現損益 ≒ NAV - balance - unrealized_pl は必ずしも正確ではないが
        # 口座サマリーの pl フィールドを使う
        return float(summary.get('pl', 0.0))
