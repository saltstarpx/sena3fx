"""
broker_oanda.py - OANDA v20 REST APIブローカー
===============================================
現在は Exness (MetaApi) に移行済み。
BROKER=oanda 指定時のフォールバック用に最低限のインターフェースを保持。
"""
import pandas as pd
import requests
import logging
from broker_base import BrokerBase

logger = logging.getLogger(__name__)


class OandaBroker(BrokerBase):
    """OANDA v20 REST API ブローカー"""

    OANDA_BASE = "https://api-fxpractice.oanda.com"

    # OANDA用シンボルマッピング
    SYMBOL_MAP = {
        "USDJPY": "USD_JPY", "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD",
        "AUDUSD": "AUD_USD", "NZDUSD": "NZD_USD", "USDCAD": "USD_CAD",
        "EURJPY": "EUR_JPY", "GBPJPY": "GBP_JPY", "XAUUSD": "XAU_USD",
        "XAGUSD": "XAG_USD", "US30": "US30_USD", "SPX500": "SPX500_USD",
        "NAS100": "NAS100_USD",
    }

    GRAN_MAP = {
        "M1": "M1", "M5": "M5", "M15": "M15",
        "H1": "H1", "H4": "H4", "D": "D",
    }

    def __init__(self, token: str, account_id: str):
        self.token = token
        self.account_id = account_id
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _oanda_sym(self, symbol: str) -> str:
        return self.SYMBOL_MAP.get(symbol, symbol.replace("/", "_"))

    def get_candles(self, symbol, granularity, count=200):
        inst = self._oanda_sym(symbol)
        gran = self.GRAN_MAP.get(granularity, granularity)
        url = f"{self.OANDA_BASE}/v3/instruments/{inst}/candles"
        params = {"granularity": gran, "count": count, "price": "M"}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=15)
            r.raise_for_status()
            candles = r.json().get("candles", [])
            rows = []
            for c in candles:
                if not c.get("complete", False) and gran != "M1":
                    continue
                m = c["mid"]
                rows.append({
                    "timestamp": pd.Timestamp(c["time"]),
                    "open": float(m["o"]), "high": float(m["h"]),
                    "low": float(m["l"]), "close": float(m["c"]),
                    "volume": int(c.get("volume", 0)),
                })
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df.index = df.index.tz_convert("UTC")
            return df
        except Exception as e:
            logger.error(f"OANDA candles error {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol):
        inst = self._oanda_sym(symbol)
        url = f"{self.OANDA_BASE}/v3/instruments/{inst}/candles"
        params = {"granularity": "M1", "count": 1, "price": "M"}
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=10)
            r.raise_for_status()
            candles = r.json().get("candles", [])
            if candles:
                return float(candles[-1]["mid"]["c"])
        except Exception as e:
            logger.error(f"OANDA price error {symbol}: {e}")
        return 0.0

    def place_order(self, symbol, units, sl, tp):
        inst = self._oanda_sym(symbol)
        url = f"{self.OANDA_BASE}/v3/accounts/{self.account_id}/orders"
        body = {
            "order": {
                "type": "MARKET",
                "instrument": inst,
                "units": str(int(units)),
                "stopLossOnFill": {"price": f"{sl:.5f}"},
                "takeProfitOnFill": {"price": f"{tp:.5f}"},
            }
        }
        try:
            r = requests.post(url, headers=self.headers, json=body, timeout=15)
            r.raise_for_status()
            data = r.json()
            fill = data.get("orderFillTransaction", {})
            return {
                "trade_id": str(fill.get("tradeOpened", {}).get("tradeID", "")),
                "fill_price": float(fill.get("price", 0)),
            }
        except Exception as e:
            logger.error(f"OANDA order error {symbol}: {e}")
            return {}

    def close_trade(self, trade_id):
        url = f"{self.OANDA_BASE}/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        try:
            r = requests.put(url, headers=self.headers, json={"units": "ALL"}, timeout=15)
            r.raise_for_status()
            data = r.json()
            return {"exit_price": float(data.get("orderFillTransaction", {}).get("price", 0))}
        except Exception as e:
            logger.error(f"OANDA close error {trade_id}: {e}")
            return {}

    def get_account_equity(self):
        url = f"{self.OANDA_BASE}/v3/accounts/{self.account_id}/summary"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            r.raise_for_status()
            return float(r.json().get("account", {}).get("balance", 0))
        except Exception as e:
            logger.error(f"OANDA equity error: {e}")
            return 0.0

    def get_open_positions(self):
        url = f"{self.OANDA_BASE}/v3/accounts/{self.account_id}/openTrades"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            r.raise_for_status()
            trades = r.json().get("trades", [])
            result = {}
            for t in trades:
                tid = str(t["id"])
                units = int(t.get("currentUnits", 0))
                result[tid] = {
                    "symbol": t.get("instrument", ""),
                    "type": "buy" if units > 0 else "sell",
                    "volume": abs(units),
                    "profit": float(t.get("unrealizedPL", 0)),
                }
            return result
        except Exception as e:
            logger.error(f"OANDA positions error: {e}")
            return {}
