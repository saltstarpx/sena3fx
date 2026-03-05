"""
OANDA REST API クライアント
.envファイルまたは環境変数から接続情報を読み込み、発注・決済・ポジション取得を行う。

環境変数の優先順位:
  APIキー  : OANDA_API_KEY（推奨） > OANDA_API_TOKEN（後方互換）
  環境種別 : OANDA_ENVIRONMENT（推奨） > OANDA_ENV（後方互換）

今後の新規設定は OANDA_API_KEY / OANDA_ENVIRONMENT を推奨。
OANDA_API_TOKEN / OANDA_ENV は後方互換として引き続きサポートする。
"""

import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    # python-dotenvが未インストールの場合は.envを手動で読み込む
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

try:
    import oandapyV20
    import oandapyV20.endpoints.orders as orders
    import oandapyV20.endpoints.trades as trades
    import oandapyV20.endpoints.accounts as accounts
    from oandapyV20.exceptions import V20Error
except ImportError:
    raise ImportError(
        "oandapyV20がインストールされていません。"
        "pip install oandapyV20 を実行してください。"
    )


def resolve_oanda_credentials() -> dict:
    """
    OANDA認証情報を環境変数から解決して返す。

    優先順位:
      api_key     : OANDA_API_KEY > OANDA_API_TOKEN
      environment : OANDA_ENVIRONMENT > OANDA_ENV（デフォルト: 'practice'）
      account_id  : OANDA_ACCOUNT_ID

    Returns:
        dict: {'api_key': str, 'environment': str, 'account_id': str}

    Raises:
        ValueError: api_key または account_id が未設定の場合
    """
    # APIキー: OANDA_API_KEY 優先、OANDA_API_TOKEN フォールバック
    api_key = os.environ.get('OANDA_API_KEY') or os.environ.get('OANDA_API_TOKEN', '')

    # 環境種別: OANDA_ENVIRONMENT 優先、OANDA_ENV フォールバック
    environment = (
        os.environ.get('OANDA_ENVIRONMENT')
        or os.environ.get('OANDA_ENV', 'practice')
    )

    account_id = os.environ.get('OANDA_ACCOUNT_ID', '')

    if not api_key:
        raise ValueError(
            "OANDA APIキーが設定されていません。\n"
            "  推奨: OANDA_API_KEY=<your_token> を .env に設定してください。\n"
            "  後方互換: OANDA_API_TOKEN も引き続き使用可能です。"
        )
    if not account_id:
        raise ValueError(
            "OANDA_ACCOUNT_ID が設定されていません。.env ファイルを確認してください。"
        )

    return {
        'api_key': api_key,
        'environment': environment,
        'account_id': account_id,
    }


class OandaClient:
    """OANDA REST API クライアント"""

    def __init__(self):
        creds = resolve_oanda_credentials()
        self.api_key = creds['api_key']
        self.account_id = creds['account_id']
        self.environment = creds['environment']

        self.client = oandapyV20.API(
            access_token=self.api_key,
            environment=self.environment
        )

    def create_order(self, instrument: str, units: int) -> dict:
        """
        成行注文を発注する。
        units > 0 で買い、units < 0 で売り。
        """
        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        r = orders.OrderCreate(self.account_id, data=data)
        response = self.client.request(r)
        return response

    def close_trade(self, trade_id: str) -> dict:
        """指定したトレードIDのポジションを全決済する。"""
        r = trades.TradeClose(self.account_id, tradeID=trade_id)
        response = self.client.request(r)
        return response

    def get_open_trades(self, instrument: str = None) -> list:
        """
        オープンなポジション一覧を取得する。
        instrument を指定すると、その銘柄のみ返す。
        """
        params = {}
        if instrument:
            params["instrument"] = instrument
        r = trades.TradesList(self.account_id, params=params)
        response = self.client.request(r)
        return response.get("trades", [])

    def get_account_summary(self) -> dict:
        """口座サマリー（残高・証拠金等）を取得する。"""
        r = accounts.AccountSummary(self.account_id)
        return self.client.request(r)

    def place_order_with_sl_tp(
        self,
        instrument: str,
        units: int,
        sl_price: float,
        tp_price: float,
    ) -> dict:
        """
        SL/TP付き成行注文を発注する。
        units > 0 で買い、units < 0 で売り。
        """
        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": f"{sl_price:.5f}",
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": f"{tp_price:.5f}",
                    "timeInForce": "GTC",
                },
            }
        }
        r = orders.OrderCreate(self.account_id, data=data)
        return self.client.request(r)

    def retry_request(self, endpoint, max_retries: int = 3, wait: float = 2.0):
        """
        レート制限・一時エラー時のリトライ付きリクエスト。
        """
        for attempt in range(max_retries):
            try:
                return self.client.request(endpoint)
            except V20Error as e:
                if attempt < max_retries - 1:
                    time.sleep(wait * (attempt + 1))
                else:
                    raise
