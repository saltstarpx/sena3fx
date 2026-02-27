"""
OANDA v20 API ブローカーラッパー
=================================
XAU_USD (金) / XAG_USD (銀) の注文・口座管理を担当。

使用ライブラリ: oandapyV20
  pip install oandapyV20

API エンドポイント:
  本番: https://api-fxtrade.oanda.com
  デモ: https://api-fxpractice.oanda.com
"""

import logging
import pandas as pd

import oandapyV20
import oandapyV20.endpoints.accounts as v20_accounts
import oandapyV20.endpoints.instruments as v20_instruments
import oandapyV20.endpoints.orders as v20_orders
import oandapyV20.endpoints.positions as v20_positions
import oandapyV20.endpoints.pricing as v20_pricing
import oandapyV20.endpoints.trades as v20_trades

log = logging.getLogger('sena3fx')


class OandaBroker:
    """OANDA v20 REST API ラッパー"""

    def __init__(self, account_id: str, api_key: str, environment: str = 'practice'):
        """
        Args:
            account_id: OANDA口座ID (例: "001-001-XXXXXXX-001")
            api_key:    APIキー (OANDAダッシュボードで取得)
            environment: 'practice' (デモ) または 'live' (本番)
        """
        self.account_id = account_id
        env = 'practice' if environment.lower() in ('practice', 'demo') else 'live'
        self.client = oandapyV20.API(access_token=api_key, environment=env)
        log.info(f"OANDA ブローカー初期化完了 (環境: {env.upper()}, 口座: {account_id})")

    # ------------------------------------------------------------------ #
    #  口座情報                                                            #
    # ------------------------------------------------------------------ #

    def get_account_summary(self) -> dict:
        """口座サマリー取得"""
        r = v20_accounts.AccountSummary(self.account_id)
        self.client.request(r)
        return r.response['account']

    def get_balance(self) -> float:
        """現在の口座残高 (USD) を取得"""
        summary = self.get_account_summary()
        return float(summary['balance'])

    def get_nav(self) -> float:
        """NAV (純資産価値) を取得 - 含み損益込み残高"""
        summary = self.get_account_summary()
        return float(summary['NAV'])

    # ------------------------------------------------------------------ #
    #  価格・ローソク足データ                                              #
    # ------------------------------------------------------------------ #

    def get_candles(self, instrument: str, granularity: str = 'H4',
                    count: int = 200) -> pd.DataFrame:
        """
        OHLCローソク足データ取得 (完成バーのみ)。

        Args:
            instrument:  'XAU_USD' | 'XAG_USD'
            granularity: 'H4' | 'H1' | 'M15' | 'D' など
            count:       取得バー本数 (最大 5000)

        Returns:
            pd.DataFrame: columns=[open, high, low, close, volume]
                          index=datetime (UTC, TZなし)
        """
        params = {
            'count': min(count, 5000),
            'granularity': granularity,
            'price': 'M',  # mid価格
        }
        r = v20_instruments.InstrumentsCandles(instrument, params=params)
        self.client.request(r)

        rows = []
        for candle in r.response.get('candles', []):
            if not candle.get('complete', False):
                continue  # 未確定バーはスキップ
            rows.append({
                'datetime': pd.to_datetime(candle['time']),
                'open':   float(candle['mid']['o']),
                'high':   float(candle['mid']['h']),
                'low':    float(candle['mid']['l']),
                'close':  float(candle['mid']['c']),
                'volume': int(candle['volume']),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.set_index('datetime')
        # TZを除去 (lib/のコードがTZなしを前提としているため)
        df.index = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
        return df

    def get_current_price(self, instrument: str) -> tuple:
        """
        現在のbid/ask価格を取得。

        Returns:
            (bid: float, ask: float)
        """
        params = {'instruments': instrument}
        r = v20_pricing.PricingInfo(self.account_id, params=params)
        self.client.request(r)
        p = r.response['prices'][0]
        bid = float(p['bids'][0]['price'])
        ask = float(p['asks'][0]['price'])
        return bid, ask

    # ------------------------------------------------------------------ #
    #  ポジション管理                                                      #
    # ------------------------------------------------------------------ #

    def get_position(self, instrument: str) -> tuple:
        """
        指定通貨ペアの現在ポジションを取得。

        Returns:
            (long_units: int, short_units: int)
            ポジションなしの場合は (0, 0)
        """
        try:
            r = v20_positions.PositionDetails(self.account_id, instrument=instrument)
            self.client.request(r)
            pos = r.response['position']
            long_units  = int(pos['long']['units'])
            short_units = int(pos['short']['units'])  # 負数で返ることも
            return long_units, short_units
        except oandapyV20.exceptions.V20Error as e:
            if '404' in str(e) or 'does not exist' in str(e).lower():
                return 0, 0
            raise

    def get_open_positions(self) -> list:
        """全オープンポジション一覧取得"""
        r = v20_positions.OpenPositions(self.account_id)
        self.client.request(r)
        return r.response.get('positions', [])

    def get_open_trades(self) -> list:
        """全オープントレード一覧取得"""
        r = v20_trades.OpenTrades(self.account_id)
        self.client.request(r)
        return r.response.get('trades', [])

    # ------------------------------------------------------------------ #
    #  注文                                                                #
    # ------------------------------------------------------------------ #

    def place_market_order(self, instrument: str, units: int,
                           sl_price: float = None, tp_price: float = None) -> dict:
        """
        成行注文を発注。

        Args:
            instrument: 'XAU_USD' | 'XAG_USD'
            units:      注文ユニット数 (正数=ロング, 負数=ショート)
                        XAU_USD: 1 unit = 1 troy ounce of Gold
                        XAG_USD: 1 unit = 1 troy ounce of Silver
            sl_price:   ストップロス価格 (GTC)
            tp_price:   テイクプロフィット価格 (GTC)

        Returns:
            OANDA APIレスポンス dict
        """
        data = {
            'order': {
                'type': 'MARKET',
                'instrument': instrument,
                'units': str(units),
                'timeInForce': 'FOK',  # Fill Or Kill
                'positionFill': 'DEFAULT',
            }
        }

        if sl_price is not None:
            data['order']['stopLossOnFill'] = {
                'price': f'{sl_price:.3f}',
                'timeInForce': 'GTC',
            }

        if tp_price is not None:
            data['order']['takeProfitOnFill'] = {
                'price': f'{tp_price:.3f}',
                'timeInForce': 'GTC',
            }

        r = v20_orders.Orders(self.account_id, data=data)
        self.client.request(r)
        return r.response

    def close_position(self, instrument: str, side: str = 'long') -> dict:
        """
        ポジションをクローズ。

        Args:
            instrument: 'XAU_USD' | 'XAG_USD'
            side:       'long' または 'short'
        """
        if side == 'long':
            data = {'longUnits': 'ALL'}
        else:
            data = {'shortUnits': 'ALL'}

        r = v20_positions.PositionClose(self.account_id, instrument=instrument, data=data)
        self.client.request(r)
        return r.response

    def modify_trade_sl(self, trade_id: str, sl_price: float) -> dict:
        """既存トレードのSL価格を変更"""
        data = {
            'stopLoss': {
                'price': f'{sl_price:.3f}',
                'timeInForce': 'GTC',
            }
        }
        r = v20_trades.TradeCRCDO(self.account_id, tradeID=trade_id, data=data)
        self.client.request(r)
        return r.response
