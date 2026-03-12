"""
broker_metaapi.py - MetaApi経由 Exness MT5 ブローカー実装
=========================================================
MetaApi Cloud (https://metaapi.cloud) を使って、Linux環境から
Exness MT5アカウントにREST APIでアクセスする。

Windows / MT5ターミナル不要。Cloud Run上で動作可能。

【必要な環境変数】
  METAAPI_TOKEN    : MetaApi APIトークン（https://app.metaapi.cloud で取得）
  METAAPI_ACCOUNT_ID : MetaApiに登録したMT5アカウントのID
  EQUITY_JPY       : 口座残高（JPY）※自動取得も可能

【MetaApiセットアップ手順】
  1. https://metaapi.cloud でアカウント作成（無料プランあり）
  2. MT5アカウントを追加（Exness-MT5Real3 / ログイン / パスワード）
  3. APIトークンとアカウントIDを取得
  4. 環境変数に設定
"""
from __future__ import annotations
import logging
import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

from broker_base import BrokerBase

logger = logging.getLogger(__name__)

METAAPI_BASE = "https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"
METAAPI_MARKET = "https://mt-market-data-client-api-v1.agiliumtrade.agiliumtrade.ai"


class MetaApiBroker(BrokerBase):
    """MetaApi REST API経由でExness MT5を操作するブローカー"""

    def __init__(self, token: str, account_id: str, equity_jpy: float = 1_000_000):
        self.token = token
        self.account_id = account_id
        self.equity_jpy = equity_jpy
        self.headers = {
            "auth-token": token,
            "Content-Type": "application/json",
        }
        # Exness MT5銘柄名マッピング（末尾 m 付きの場合あり）
        self._sym_map = {
            "USDJPY": "USDJPYm", "EURJPY": "EURJPYm", "GBPJPY": "GBPJPYm",
            "EURUSD": "EURUSDm", "GBPUSD": "GBPUSDm", "AUDUSD": "AUDUSDm",
            "NZDUSD": "NZDUSDm", "USDCAD": "USDCADm", "USDCHF": "USDCHFm",
            "XAUUSD": "XAUUSDm", "XAGUSD": "XAGUSDm",
            "SPX500": "SP500m",  "US30":   "US30m",   "NAS100": "USTEC",
        }
        # MetaApi timeframe マッピング
        self._tf_map = {
            "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D": "1d", "W": "1w",
        }

    def _mt5_symbol(self, symbol: str) -> str:
        return self._sym_map.get(symbol, symbol)

    def _api_url(self, path: str) -> str:
        return f"{METAAPI_BASE}/users/current/accounts/{self.account_id}{path}"

    def _market_url(self, path: str) -> str:
        return f"{METAAPI_MARKET}/users/current/accounts/{self.account_id}{path}"

    def get_candles(self, symbol: str, granularity: str,
                    count: int = 200) -> pd.DataFrame:
        mt5_sym = self._mt5_symbol(symbol)
        tf = self._tf_map.get(granularity, granularity)
        try:
            # MetaApi candles endpoint
            start_time = (datetime.now(timezone.utc) -
                          timedelta(minutes=count * self._tf_minutes(granularity)))
            r = requests.get(
                self._market_url(
                    f"/historical-market-data/symbols/{mt5_sym}"
                    f"/timeframes/{tf}/candles"
                ),
                headers=self.headers,
                params={
                    "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "limit": count,
                },
                timeout=15,
            )
            if r.status_code != 200:
                logger.error(f"MetaApi candles {mt5_sym}: {r.status_code}")
                return pd.DataFrame()
            candles = r.json()
            if not candles:
                return pd.DataFrame()
            rows = [
                {
                    "timestamp": pd.Timestamp(c["time"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": int(c.get("tickVolume", 0)),
                }
                for c in candles
            ]
            return pd.DataFrame(rows).set_index("timestamp").sort_index()
        except Exception as e:
            logger.error(f"MetaApi candles {mt5_sym}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        mt5_sym = self._mt5_symbol(symbol)
        try:
            r = requests.get(
                self._market_url(
                    f"/symbols/{mt5_sym}/current-price"
                ),
                headers=self.headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                ask = float(data.get("ask", 0))
                bid = float(data.get("bid", 0))
                if ask > 0 and bid > 0:
                    return (ask + bid) / 2
        except Exception as e:
            logger.error(f"MetaApi price {mt5_sym}: {e}")
        return 0.0

    def place_order(self, symbol: str, units: int,
                    sl: float, tp: float) -> dict:
        mt5_sym = self._mt5_symbol(symbol)
        direction = "ORDER_TYPE_BUY" if units > 0 else "ORDER_TYPE_SELL"
        # MetaApiではロット単位（1lot=100,000通貨）
        volume = round(abs(units) / 100_000, 2)
        if volume < 0.01:
            volume = 0.01
        try:
            r = requests.post(
                self._api_url("/trade"),
                headers=self.headers,
                json={
                    "actionType": direction,
                    "symbol": mt5_sym,
                    "volume": volume,
                    "stopLoss": round(sl, 5),
                    "takeProfit": round(tp, 5),
                    "comment": "YAGAMI_GOLD",
                    "clientId": "sena3fx",
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                return {
                    "trade_id": data.get("positionId", data.get("orderId", "")),
                    "fill_price": float(data.get("price", 0)),
                }
            logger.error(f"MetaApi order {mt5_sym}: {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.error(f"MetaApi order {mt5_sym}: {e}")
        return {}

    def close_trade(self, trade_id: str) -> dict:
        try:
            r = requests.post(
                self._api_url("/trade"),
                headers=self.headers,
                json={
                    "actionType": "POSITION_CLOSE_ID",
                    "positionId": trade_id,
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                return {"exit_price": float(data.get("price", 0))}
        except Exception as e:
            logger.error(f"MetaApi close {trade_id}: {e}")
        return {}

    def get_account_equity(self) -> float:
        try:
            r = requests.get(
                self._api_url("/account-information"),
                headers=self.headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                equity = float(data.get("equity", 0))
                currency = data.get("currency", "USD")
                if currency == "JPY":
                    return equity
                # USD→JPY変換（概算）
                usdjpy = self.get_current_price("USDJPY")
                if usdjpy > 0:
                    return equity * usdjpy
                return equity * 150.0
        except Exception as e:
            logger.error(f"MetaApi account: {e}")
        return self.equity_jpy

    @staticmethod
    def _tf_minutes(gran: str) -> int:
        m = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
             "H1": 60, "H4": 240, "D": 1440, "W": 10080}
        return m.get(gran, 60)
