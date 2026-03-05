"""
OANDA REST API クライアント
.envファイルまたは環境変数から接続情報を読み込み、発注・決済・ポジション取得を行う。
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


class OandaClient:
    """OANDA REST API クライアント"""

    def __init__(self):
        self.api_key = os.environ.get("OANDA_API_KEY")
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID")
        self.environment = os.environ.get("OANDA_ENVIRONMENT", "practice")

        if not self.api_key:
            raise ValueError("OANDA_API_KEY が設定されていません。.envファイルを確認してください。")
        if not self.account_id:
            raise ValueError("OANDA_ACCOUNT_ID が設定されていません。.envファイルを確認してください。")

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
