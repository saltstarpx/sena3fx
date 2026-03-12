"""
broker_oanda.py - OANDA v20 REST API ブローカー実装
====================================================
既存のCloud Run main.pyのOANDA関連ロジックを抽出。
ペーパートレード用（practice API）。
"""
from __future__ import annotations
import logging
import requests
import pandas as pd

from broker_base import BrokerBase

logger = logging.getLogger(__name__)


class OandaBroker(BrokerBase):
    def __init__(self, token: str, account_id: str,
                 base_url: str = "https://api-fxpractice.oanda.com"):
        self.token = token
        self.account_id = account_id
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        # OANDA銘柄名マッピング
        self._sym_map = {
            "USDJPY": "USD_JPY", "EURJPY": "EUR_JPY", "GBPJPY": "GBP_JPY",
            "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "AUDUSD": "AUD_USD",
            "NZDUSD": "NZD_USD", "USDCAD": "USD_CAD", "USDCHF": "USD_CHF",
            "XAUUSD": "XAU_USD", "XAGUSD": "XAG_USD",
            "SPX500": "SPX500_USD", "US30": "US30_USD", "NAS100": "NAS100_USD",
        }
        # OANDA granularity マッピング
        self._gran_map = {
            "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
            "H1": "H1", "H4": "H4", "D": "D", "W": "W",
        }

    def _instrument(self, symbol: str) -> str:
        return self._sym_map.get(symbol, symbol)

    def get_candles(self, symbol: str, granularity: str,
                    count: int = 200) -> pd.DataFrame:
        instr = self._instrument(symbol)
        gran = self._gran_map.get(granularity, granularity)
        try:
            r = requests.get(
                f"{self.base_url}/v3/instruments/{instr}/candles",
                headers=self.headers,
                params={"granularity": gran, "count": count, "price": "M"},
                timeout=15,
            )
            if r.status_code != 200:
                return pd.DataFrame()
            rows = [
                {
                    "timestamp": pd.Timestamp(c["time"]),
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c.get("volume", 0)),
                }
                for c in r.json()["candles"]
                if c.get("complete", True)
            ]
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows).set_index("timestamp").sort_index()
        except Exception as e:
            logger.error(f"OANDA candles {instr}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        instr = self._instrument(symbol)
        try:
            r = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
                headers=self.headers,
                params={"instruments": instr},
                timeout=10,
            )
            if r.status_code == 200:
                p = r.json().get("prices", [])
                if p:
                    return (float(p[0]["asks"][0]["price"]) +
                            float(p[0]["bids"][0]["price"])) / 2
        except Exception as e:
            logger.error(f"OANDA price {instr}: {e}")
        return 0.0

    def place_order(self, symbol: str, units: int,
                    sl: float, tp: float) -> dict:
        instr = self._instrument(symbol)
        try:
            r = requests.post(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                headers=self.headers,
                timeout=10,
                json={"order": {
                    "type": "MARKET",
                    "instrument": instr,
                    "units": str(units),
                    "stopLossOnFill": {"price": f"{sl:.5f}", "timeInForce": "GTC"},
                    "takeProfitOnFill": {"price": f"{tp:.5f}", "timeInForce": "GTC"},
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT",
                }},
            )
            if r.status_code in (200, 201):
                fill = r.json().get("orderFillTransaction", {})
                return {
                    "trade_id": fill.get("tradeOpened", {}).get("tradeID", ""),
                    "fill_price": float(fill.get("price", 0)),
                }
            logger.error(f"OANDA order {instr}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            logger.error(f"OANDA order {instr}: {e}")
        return {}

    def close_trade(self, trade_id: str) -> dict:
        try:
            r = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/close",
                headers=self.headers,
                timeout=10,
            )
            if r.status_code == 200:
                return {
                    "exit_price": float(
                        r.json().get("orderFillTransaction", {}).get("price", 0)
                    )
                }
        except Exception as e:
            logger.error(f"OANDA close {trade_id}: {e}")
        return {}

    def get_account_equity(self) -> float:
        try:
            r = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}",
                headers=self.headers,
                timeout=10,
            )
            if r.status_code == 200:
                acc = r.json().get("account", {})
                return float(acc.get("balance", 1_000_000))
        except Exception as e:
            logger.error(f"OANDA account: {e}")
        return 1_000_000.0
